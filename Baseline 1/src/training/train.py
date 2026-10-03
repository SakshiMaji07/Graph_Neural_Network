"""Train the baseline Siamese GNN from a YAML configuration file."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import collate_graph_pairs
from src.data.graph_loader import GraphLoader
from src.data.pair_dataset import PairMetadata, SLAMGraphPairDataset
from src.data.preprocessing import similarity_targets_from_metadata
from src.losses.combined_loss import CombinedLoss
from src.losses.similarity_loss import SimilarityLoss
from src.models.siamese_gnn import SiameseGNN
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set random generators and deterministic backend options."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return config


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix in {".json", ".jsonl", ".yaml", ".yml"}:
        with manifest_path.open("r", encoding="utf-8") as stream:
            if suffix == ".jsonl":
                rows = [json.loads(line) for line in stream if line.strip()]
            elif suffix in {".yaml", ".yml"}:
                rows = yaml.safe_load(stream)
            else:
                rows = json.load(stream)
        if isinstance(rows, dict):
            rows = rows.get("pairs", rows.get("examples"))
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(
                f"Pair manifest must contain a list of mappings (or a 'pairs'/'examples' list): "
                f"{manifest_path}"
            )
        return rows
    raise ValueError(
        f"Unsupported pair manifest format '{manifest_path.suffix}'. "
        "Use CSV, JSON, JSONL, or YAML."
    )


def _first_value(row: dict[str, Any], names: tuple[str, ...], row_number: int) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    raise ValueError(f"Pair manifest row {row_number} is missing one of: {', '.join(names)}.")


def _build_pair_dataset(
    configured_path: str | Path,
    dataset_config: dict[str, Any],
    split: str,
    target_config: dict[str, Any] | None = None,
) -> SLAMGraphPairDataset:
    """Load pair manifests or five-hypothesis scene folders."""
    manifest_value = dataset_config.get(f"{split}_manifest", configured_path)
    manifest_path = _resolve_path(manifest_value)

    if manifest_path.is_dir():
        manifest_names = (
            f"{split}_pairs.csv",
            f"{split}_manifest.csv",
            f"{split}.csv",
            "pairs.csv",
            "pair_manifest.csv",
            "manifest.csv",
            f"{split}_pairs.json",
            f"{split}_manifest.json",
            f"{split}.json",
            "pairs.json",
            "pair_manifest.json",
            "manifest.json",
            f"{split}_pairs.jsonl",
            "pairs.jsonl",
            f"{split}_pairs.yaml",
            "pairs.yaml",
        )
        candidates = [manifest_path / name for name in manifest_names]
        found_manifest = next((path for path in candidates if path.is_file()), None)
        if found_manifest is None:
            scene_directories = sorted(path for path in manifest_path.iterdir() if path.is_dir())
            if not scene_directories:
                raise FileNotFoundError(
                    f"No {split} pair manifest or scene folders found in '{manifest_path}'."
                )
            return _build_scene_pair_dataset(
                scene_directories,
                dataset_config,
                target_config or {},
            )
        manifest_path = found_manifest

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No {split} pair manifest found at '{manifest_path}'. "
            "Provide a manifest file or a directory containing a split-specific "
            "manifest, pairs.csv, pair_manifest.csv, or manifest.csv."
        )

    rows = _manifest_rows(manifest_path)
    if not rows:
        raise ValueError(f"The {split} pair manifest is empty: {manifest_path}")

    graph_loader = GraphLoader(
        normalize=bool(dataset_config.get("normalize", False)),
        strict=bool(dataset_config.get("strict", True)),
    )
    pair_dataset = SLAMGraphPairDataset(graph_loader=graph_loader)
    base_directory = manifest_path.parent

    for row_number, row in enumerate(rows, start=2):
        hypothesis = _first_value(
            row,
            ("hypothesis_path", "graph_hypothesis", "hypothesis", "graph_a"),
            row_number,
        )
        ground_truth = _first_value(
            row,
            ("ground_truth_path", "graph_ground_truth", "ground_truth", "graph_b"),
            row_number,
        )
        similarity = _first_value(
            row,
            ("similarity", "similarity_target", "target", "score"),
            row_number,
        )

        hypothesis_path = Path(str(hypothesis)).expanduser()
        ground_truth_path = Path(str(ground_truth)).expanduser()
        if not hypothesis_path.is_absolute():
            hypothesis_path = base_directory / hypothesis_path
        if not ground_truth_path.is_absolute():
            ground_truth_path = base_directory / ground_truth_path

        def optional_string(*keys: str) -> str | None:
            for key in keys:
                value = row.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
            return None

        metadata = PairMetadata(
            hypothesis_id=optional_string("hypothesis_id", "id"),
            scenario_id=optional_string("scene_id", "scenario_id", "scenario"),
            ground_truth_id=optional_string("ground_truth_id"),
        )
        pair_dataset.add_from_paths(
            hypothesis_path=hypothesis_path,
            ground_truth_path=ground_truth_path,
            similarity=float(similarity),
            hypothesis_id=metadata.hypothesis_id,
            scenario_id=metadata.scenario_id,
            ground_truth_id=metadata.ground_truth_id,
        )

    return pair_dataset


def _build_scene_pair_dataset(
    scene_directories: list[Path],
    dataset_config: dict[str, Any],
    target_config: dict[str, Any],
) -> SLAMGraphPairDataset:
    """Build five pairwise examples per scene using the scene's pose metadata."""
    graph_loader = GraphLoader(
        normalize=bool(dataset_config.get("normalize", False)),
        strict=bool(dataset_config.get("strict", True)),
    )
    sigma_translation = float(target_config.get("sigma_translation", 1.0))
    sigma_rotation = float(target_config.get("sigma_rotation", 1.0))
    pair_dataset = SLAMGraphPairDataset(graph_loader=graph_loader)

    for scene_directory in scene_directories:
        metadata_path = scene_directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing scene metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            scene_metadata = json.load(stream)
        if not isinstance(scene_metadata, dict):
            raise ValueError(f"Scene metadata must be a JSON object: {metadata_path}")

        targets = similarity_targets_from_metadata(
            scene_metadata,
            sigma_translation=sigma_translation,
            sigma_rotation=sigma_rotation,
        )
        ground_truth_path = scene_directory / "gt.npz"
        if not ground_truth_path.is_file():
            raise FileNotFoundError(f"Missing ground-truth graph: {ground_truth_path}")
        ground_truth_graph = graph_loader.load(ground_truth_path)

        for index, target in enumerate(targets, start=1):
            hypothesis_path = scene_directory / f"h{index}.npz"
            if not hypothesis_path.is_file():
                raise FileNotFoundError(f"Missing hypothesis graph: {hypothesis_path}")
            pair_dataset.add_pair(
                graph_loader.load(hypothesis_path),
                ground_truth_graph,
                target,
                metadata=PairMetadata(
                    hypothesis_id=f"H{index}",
                    scenario_id=scene_directory.name,
                    ground_truth_id=f"{scene_directory.name}:gt",
                ),
            )

    return pair_dataset


def _validate_scene_split(
    train_dataset: SLAMGraphPairDataset,
    val_dataset: SLAMGraphPairDataset,
) -> None:
    """Require scene identifiers and reject scene overlap across training splits."""
    split_ids: list[set[str]] = []
    for split_name, dataset in (("training", train_dataset), ("validation", val_dataset)):
        hypotheses_by_scene: dict[str, list[str]] = {}
        for index in range(len(dataset)):
            metadata = dataset[index].metadata
            if metadata is None or not metadata.scenario_id:
                raise ValueError(
                    f"Every {split_name} pair must have a scene_id/scenario_id so splits can be "
                    "verified at scene level."
                )
            if not metadata.hypothesis_id:
                raise ValueError(
                    f"Every {split_name} pair must have a hypothesis_id so each scene's five "
                    "hypotheses can be verified."
                )
            hypotheses_by_scene.setdefault(metadata.scenario_id, []).append(metadata.hypothesis_id)

        for scene_id, hypothesis_ids in hypotheses_by_scene.items():
            if len(hypothesis_ids) != 5 or len(set(hypothesis_ids)) != 5:
                raise ValueError(
                    f"Scene '{scene_id}' in the {split_name} split must contain exactly five "
                    "distinct hypothesis pairs."
                )
        scene_ids = set(hypotheses_by_scene)
        split_ids.append(scene_ids)

    overlap = split_ids[0] & split_ids[1]
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            f"Training and validation splits share scene IDs ({examples}); split by scene, not pairs."
        )


def _resolve_device(configured_device: str | None, cli_device: str | None) -> torch.device:
    requested = cli_device or configured_device or "auto"
    if requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if cli_device is not None:
            raise RuntimeError("CUDA was requested with --device, but CUDA is not available.")
        print("CUDA configured but unavailable; using CPU.")
        return torch.device("cpu")
    return device


def _build_model(config: dict[str, Any]) -> SiameseGNN:
    model_config = _mapping(config, "model")
    mlp_config = _mapping(config, "similarity_mlp")
    return SiameseGNN(
        input_dim=int(model_config.get("input_dim", 3)),
        hidden_dim=int(model_config.get("hidden_dim", 32)),
        embedding_dim=int(model_config.get("embedding_dim", 16)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.0)),
        activation=str(model_config.get("activation", "relu")),
        conv_type=str(model_config.get("conv_type", "gcn")),
        pooling=str(model_config.get("pooling", "mean")),
        mlp_hidden_dims=list(
            mlp_config.get("hidden_dims", model_config.get("mlp_hidden_dims", [64, 32]))
        ),
        include_raw_features=bool(
            model_config.get("include_raw_features", model_config.get("comparison_features", False))
        ),
    )


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> Optimizer:
    training_config = _mapping(config, "training")
    name = str(training_config.get("optimizer", "adam")).lower()
    learning_rate = float(training_config.get("learning_rate", training_config.get("lr", 1e-3)))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if name == "adam":
        return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'. Supported optimizers are 'adam' and 'adamw'.")


def _build_scheduler(
    optimizer: Optimizer,
    config: dict[str, Any],
) -> Any | None:
    training_config = _mapping(config, "training")
    scheduler_config = training_config.get("scheduler")
    if scheduler_config is None or str(scheduler_config).lower() in {"", "none", "null"}:
        return None

    if isinstance(scheduler_config, dict):
        scheduler_type = str(
            scheduler_config.get("type", scheduler_config.get("name", ""))
        ).lower()
        options = scheduler_config
    else:
        scheduler_type = str(scheduler_config).lower()
        options = {}

    if scheduler_type in {"cosine", "cosine_annealing", "cosineannealing"}:
        return CosineAnnealingLR(
            optimizer,
            T_max=int(options.get("T_max", options.get("t_max", training_config.get("epochs", 100)))),
            eta_min=float(options.get("eta_min", 0.0)),
        )
    if scheduler_type in {"step", "steplr"}:
        return StepLR(
            optimizer,
            step_size=int(options.get("step_size", 10)),
            gamma=float(options.get("gamma", 0.5)),
        )
    if scheduler_type in {"plateau", "reduce_on_plateau", "reduceonplateau", "reducelronplateau"}:
        return ReduceLROnPlateau(
            optimizer,
            mode=str(options.get("mode", "min")),
            factor=float(options.get("factor", 0.5)),
            patience=int(options.get("patience", 3)),
            min_lr=float(options.get("min_lr", 1e-6)),
            threshold=float(options.get("threshold", 1e-4)),
        )
    raise ValueError(
        f"Unsupported scheduler '{scheduler_type}'. Supported schedulers are cosine_annealing, "
        "step, plateau, or null."
    )


def _build_criterion(config: dict[str, Any]) -> torch.nn.Module:
    loss_config = _mapping(config, "loss")
    loss_type = str(loss_config.get("type", "mse")).lower()
    if loss_type in {"combined", "combined_loss"}:
        if bool(loss_config.get("include_ranking", False)):
            raise ValueError(
                "Ranking loss requires five ordered hypotheses per scenario in each batch. "
                "The pairwise Trainer batches examples independently; set include_ranking to false "
                "or use a scenario-aware training loop."
            )
        return CombinedLoss(
            similarity_loss_type=str(loss_config.get("similarity_loss_type", "mse")),
            similarity_beta=float(loss_config.get("similarity_beta", 1.0)),
            lambda_similarity=float(loss_config.get("similarity_weight", 1.0)),
            lambda_ranking=float(loss_config.get("ranking_weight", 0.0)),
            ranking_margin=float(loss_config.get("ranking_margin", 1.0)),
            include_similarity=True,
            include_ranking=False,
        )
    if loss_type in {"mse", "mean_squared_error", "smooth_l1", "smoothl1", "smooth-l1"}:
        return SimilarityLoss(
            loss_type=loss_type,
            beta=float(loss_config.get("beta", 1.0)),
        )
    raise ValueError(f"Unsupported loss type '{loss_type}'. Use 'combined', 'mse', or 'smooth_l1'.")


def _restore_checkpoint(
    resume_path: Path,
    trainer: Trainer,
    loader_generator: torch.Generator,
) -> None:
    if not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=trainer.device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Resume checkpoint must contain a mapping: {resume_path}")
    model_state = checkpoint.get("model_state_dict", checkpoint)
    trainer.model.load_state_dict(model_state)
    if "optimizer_state_dict" in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        if trainer.scheduler is None:
            raise ValueError("Resume checkpoint has scheduler state but no scheduler is configured.")
        trainer.scheduler.load_state_dict(scheduler_state)
    trainer.current_epoch = int(checkpoint.get("epoch", 0))
    trainer.best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
    trainer.best_epoch = int(checkpoint.get("best_epoch", trainer.current_epoch))
    history = checkpoint.get("history")
    if isinstance(history, dict):
        for key in trainer.training_history:
            values = history.get(key)
            if isinstance(values, list):
                trainer.training_history[key] = [float(value) for value in values]

    random_state = checkpoint.get("random_state")
    if isinstance(random_state, dict):
        random.setstate(random_state["python"])
        np.random.set_state(random_state["numpy"])
        torch.set_rng_state(random_state["torch"])
        if torch.cuda.is_available() and random_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(random_state["cuda"])
    generator_state = checkpoint.get("data_loader_generator_state")
    if isinstance(generator_state, torch.Tensor):
        loader_generator.set_state(generator_state)
    print(f"Resumed from {resume_path} at epoch {trainer.current_epoch}.")


def _checkpoint_payload(
    trainer: Trainer,
    epoch: int,
    config: dict[str, Any],
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
        "history": trainer.training_history,
        "config": config,
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "data_loader_generator_state": loader_generator.get_state(),
    }


def _worker_seed(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline Siamese GNN.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the YAML training configuration.")
    parser.add_argument("--device", default=None, help="Override configured device (for example: cpu or cuda:0).")
    parser.add_argument("--resume", default=None, type=Path, help="Checkpoint file to resume training from.")
    arguments = parser.parse_args()

    config_path = arguments.config.expanduser().resolve()
    config = _load_config(config_path)
    dataset_config = _mapping(config, "dataset")
    training_config = _mapping(config, "training")
    hardware_config = _mapping(config, "hardware")
    logging_config = _mapping(config, "logging")
    reproducibility_config = _mapping(config, "reproducibility")

    seed = int(reproducibility_config.get("seed", config.get("seed", 42)))
    set_seed(seed)

    device = _resolve_device(hardware_config.get("device"), arguments.device)
    train_path = dataset_config.get("train_manifest", dataset_config.get("train_path"))
    val_path = dataset_config.get("val_manifest", dataset_config.get("val_path"))
    if train_path is None or val_path is None:
        raise ValueError("Configuration dataset section must define train_path and val_path.")

    target_config = _mapping(config, "similarity_target")
    train_dataset = _build_pair_dataset(train_path, dataset_config, "train", target_config)
    val_dataset = _build_pair_dataset(val_path, dataset_config, "val", target_config)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Both training and validation datasets must contain at least one pair.")
    _validate_scene_split(train_dataset, val_dataset)

    batch_size = int(training_config.get("batch_size", 32))
    if batch_size <= 0:
        raise ValueError(f"training.batch_size must be positive, got {batch_size}.")
    num_workers = int(training_config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError(f"training.num_workers must be non-negative, got {num_workers}.")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_graph_pairs,
        "worker_init_fn": _worker_seed,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(training_config.get("persistent_workers", False))

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = _build_model(config)
    criterion = _build_criterion(config)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    checkpoint_directory = _resolve_path(logging_config.get("checkpoint_dir", "checkpoints"))
    patience = int(training_config.get("early_stopping_patience", 10))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scheduler=scheduler,
        save_dir=checkpoint_directory,
        checkpoint_name=str(logging_config.get("checkpoint_name", "best_model.pt")),
        max_epochs=int(training_config.get("epochs", 100)),
        patience=patience,
        min_delta=float(training_config.get("min_delta", 1e-6)),
        early_stopping=bool(training_config.get("early_stopping", True)),
        seed=None,
    )
    trainer.checkpoint_config = config

    if arguments.resume is not None:
        _restore_checkpoint(arguments.resume.expanduser().resolve(), trainer, loader_generator)

    loss_config = _mapping(config, "loss")
    if (
        str(loss_config.get("type", "mse")).lower() in {"combined", "combined_loss"}
        and float(loss_config.get("ranking_weight", 0.0)) != 0.0
        and not bool(loss_config.get("include_ranking", False))
    ):
        print(
            "Ranking loss is disabled because the configured pairwise Trainer does not "
            "group examples into five-hypothesis scenarios."
        )

    model_configuration = {
        "model": _mapping(config, "model"),
        "similarity_mlp": _mapping(config, "similarity_mlp"),
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Number of parameters: {parameter_count:,}")
    print(f"Device: {device}")
    print("Model configuration:")
    print(json.dumps(model_configuration, indent=2, sort_keys=True, default=str))

    history = trainer.fit()
    final_epoch = trainer.current_epoch
    torch.save(
        _checkpoint_payload(trainer, final_epoch, config, loader_generator),
        checkpoint_directory / "final_model.pt",
    )
    with (checkpoint_directory / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    with (checkpoint_directory / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        history_writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "train_loss", "val_loss", "learning_rate"),
        )
        history_writer.writeheader()
        history_writer.writerows(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
            for epoch, (train_loss, val_loss, learning_rate) in enumerate(
                zip(
                    history["train_loss"],
                    history["val_loss"],
                    history["lr"],
                ),
                start=1,
            )
        )

    print(f"Best model: {trainer.checkpoint_path}")
    print(f"Final model: {checkpoint_directory / 'final_model.pt'}")
    print(f"Training history: {checkpoint_directory / 'training_history.json'}")
    print(f"Epoch history: {checkpoint_directory / 'training_history.csv'}")


if __name__ == "__main__":
    main()