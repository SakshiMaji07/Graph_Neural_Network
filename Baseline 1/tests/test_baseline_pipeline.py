import csv
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from torch_geometric.data import Data

from src.data.pair_dataset import PairMetadata, SLAMGraphPairDataset
from src.data.preprocessing import similarity_targets_from_metadata
from src.evaluation.evaluate import evaluate_dataset
from src.training.train import _build_pair_dataset, _validate_scene_split
from src.training.trainer import Trainer


def _scene_metadata():
    return {
        "ground_truth": {"position": [0.0, 0.0], "rotation": np.pi - 0.1},
        "hypotheses": [
            {"position": [0.0, 0.0], "rotation": -np.pi + 0.1},
            {"position": [1.0, 0.0], "rotation": np.pi - 0.1},
            {"position": [2.0, 0.0], "rotation": np.pi - 0.1},
            {"position": [3.0, 0.0], "rotation": np.pi - 0.1},
            {"position": [4.0, 0.0], "rotation": np.pi - 0.1},
        ],
    }


def _write_graph(path, score):
    node_features = np.zeros((1, 8), dtype=np.float32)
    node_features[0, 0] = score
    np.savez(path, node_features=node_features, edge_index=np.empty((2, 0), dtype=np.int64))


class FeatureScoreModel(nn.Module):
    def forward(self, hypothesis, _ground_truth):
        return hypothesis.x[0, 0].reshape(1)


def test_pose_targets_wrap_rotation_and_are_not_constant():
    targets = similarity_targets_from_metadata(_scene_metadata())

    assert len(targets) == 5
    assert targets[0] == pytest.approx(np.exp(-0.2**2 / 2.0))
    assert targets[1] == pytest.approx(np.exp(-0.5))
    assert targets[0] > targets[1] > targets[2]
    assert all(0.0 <= score <= 1.0 for score in targets)


def test_evaluation_uses_pose_targets_and_writes_scene_predictions(tmp_path):
    scene = tmp_path / "scene_001"
    scene.mkdir()
    _write_graph(scene / "gt.npz", 0.0)
    for index, score in enumerate((0.1, 0.2, 0.3, 0.4, 0.9), start=1):
        _write_graph(scene / f"h{index}.npz", score)
    (scene / "metadata.json").write_text(json.dumps(_scene_metadata()), encoding="utf-8")

    output_path = tmp_path / "predictions.csv"
    metrics = evaluate_dataset(
        model=FeatureScoreModel(),
        data_dir=tmp_path,
        output_csv=output_path,
        device=torch.device("cpu"),
    )

    with output_path.open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    targets = similarity_targets_from_metadata(_scene_metadata())

    assert row["scene_id"] == "scene_001"
    assert [float(row[f"target_h{i}"]) for i in range(1, 6)] == pytest.approx(targets)
    assert [float(row[f"predicted_h{i}"]) for i in range(1, 6)] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.9]
    )
    assert int(row["target_best"]) == 1
    assert int(row["predicted_best"]) == 5
    assert int(row["correct"]) == 0
    assert metrics["random_baseline_accuracy"] == 0.2


class ConstantBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Parameter(torch.tensor(0.0))

    def forward(self, hypothesis, _ground_truth):
        return self.score.expand(hypothesis.shape[0])


def _loss_batch(target_values):
    return {
        "hypothesis": torch.zeros(len(target_values)),
        "ground_truth": torch.zeros(len(target_values)),
        "target": torch.tensor(target_values, dtype=torch.float32),
    }


def test_epoch_losses_are_example_weighted_for_training_and_validation(tmp_path):
    batches = [_loss_batch([0.0, 1.0]), _loss_batch([1.0])]
    model = ConstantBatchModel()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        criterion=nn.MSELoss(),
        train_loader=batches,
        val_loader=batches,
        device="cpu",
        save_dir=tmp_path,
        verbose=False,
    )

    assert trainer.train_epoch() == pytest.approx(2.0 / 3.0)
    assert trainer.validate_epoch() == pytest.approx(2.0 / 3.0)


def _five_pair_dataset(scene_id):
    graph = Data(x=torch.zeros((1, 8)), edge_index=torch.empty((2, 0), dtype=torch.long))
    dataset = SLAMGraphPairDataset()
    for index in range(1, 6):
        dataset.add_pair(
            graph,
            graph,
            0.5,
            metadata=PairMetadata(hypothesis_id=f"H{index}", scenario_id=scene_id),
        )
    return dataset


def test_scene_split_rejects_scene_leakage():
    with pytest.raises(ValueError, match="share scene IDs"):
        _validate_scene_split(_five_pair_dataset("scene_001"), _five_pair_dataset("scene_001"))


def test_training_loads_five_pairs_from_each_scene_folder(tmp_path):
    scene = tmp_path / "train" / "scene_001"
    scene.mkdir(parents=True)
    _write_graph(scene / "gt.npz", 0.0)
    for index in range(1, 6):
        _write_graph(scene / f"h{index}.npz", float(index))
    (scene / "metadata.json").write_text(json.dumps(_scene_metadata()), encoding="utf-8")

    dataset = _build_pair_dataset(
        tmp_path / "train",
        {},
        "train",
        {"sigma_translation": 1.0, "sigma_rotation": 1.0},
    )

    assert len(dataset) == 5
    assert [dataset[index].metadata.hypothesis_id for index in range(5)] == [
        "H1", "H2", "H3", "H4", "H5"
    ]
    assert [dataset[index].similarity_target.item() for index in range(5)] == pytest.approx(
        similarity_targets_from_metadata(_scene_metadata())
    )


def test_baseline_configuration_matches_requested_architecture():
    with open("configs/baseline_siamese.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert config["model"] == {
        "input_dim": 8,
        "hidden_dim": 64,
        "embedding_dim": 64,
        "num_layers": 2,
        "conv_type": "sage",
        "pooling": "mean",
        "dropout": 0.0,
        "activation": "relu",
        "comparison_features": False,
    }
    assert config["similarity_mlp"] == {"hidden_dims": [64, 32], "dropout": 0.0}
    assert config["loss"]["type"] == "mse"
