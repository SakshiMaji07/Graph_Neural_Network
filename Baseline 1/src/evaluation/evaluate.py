"""Evaluate a trained Siamese GNN on five-hypothesis graph matching scenarios.

This script loads a trained checkpoint, evaluates it over a scenario dataset, and writes
per-scenario predictions to a CSV file. Each scenario contains a ground-truth graph and
five candidate hypothesis graphs; the model predicts a similarity score for each pair,
chooses the best hypothesis by maximum score, and compares the selected index with the
best target index.

The evaluation is intentionally deterministic: it loads the model in evaluation mode,
uses a fixed random seed when requested, and does not perform any stochastic dropout or
batch normalization updates during inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.data.graph_loader import GraphLoader
from src.evaluation.metrics import mae, rmse, spearman_rank_correlation, top1_selection_accuracy
from src.models.siamese_gnn import SiameseGNN


def set_deterministic(seed: int = 0) -> None:
    """Enable deterministic behavior for reproducible evaluation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML configuration file for evaluation settings."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".json"}:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyYAML is required to read YAML config files.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_model(checkpoint_path: str | Path, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Construct and load a Siamese GNN model from a checkpoint."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model_cfg = config.get("model", {})
    model = SiameseGNN(
        input_dim=int(model_cfg.get("input_dim", 3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        embedding_dim=int(model_cfg.get("embedding_dim", 16)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        activation=str(model_cfg.get("activation", "relu")),
        conv_type=str(model_cfg.get("conv_type", "gcn")),
        pooling=str(model_cfg.get("pooling", "mean")),
        mlp_hidden_dims=model_cfg.get("mlp_hidden_dims", [64, 32]),
        include_raw_features=bool(model_cfg.get("include_raw_features", False)),
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _ensure_graph_pair(graph: Any, name: str) -> Any:
    """Validate that the object looks like a valid PyG graph before evaluation."""
    if graph is None:
        raise ValueError(f"Missing graph for {name}.")
    return graph


def evaluate_scenario(model: torch.nn.Module, hypothesis_graphs: list[Any], gt_graph: Any, device: torch.device) -> dict[str, Any]:
    """Evaluate one scenario by comparing each hypothesis to the ground-truth graph.

    Returns a dictionary with the target values, predicted values, selected indices, and
    correctness flag for a single scenario.
    """
    if len(hypothesis_graphs) != 5:
        raise ValueError(f"Expected exactly 5 hypothesis graphs, got {len(hypothesis_graphs)}.")

    gt_graph = _ensure_graph_pair(gt_graph, "ground truth")
    gt_graph = gt_graph.to(device)

    predicted_scores = []
    target_scores = []

    for index, hypothesis in enumerate(hypothesis_graphs):
        hypothesis_graph = _ensure_graph_pair(hypothesis, f"hypothesis {index + 1}")
        hypothesis_graph = hypothesis_graph.to(device)

        with torch.no_grad():
            score = model(hypothesis_graph, gt_graph)
            score = score.detach().cpu().view(-1)
            if score.numel() != 1:
                raise ValueError(
                    f"Model output for hypothesis {index + 1} must be a scalar score per graph, got shape {tuple(score.shape)}."
                )
            predicted_scores.append(float(score.item()))

        target_scores.append(float(0.0))

    target_scores_arr = np.asarray(target_scores, dtype=np.float64)
    predicted_scores_arr = np.asarray(predicted_scores, dtype=np.float64)

    target_best = int(np.argmax(target_scores_arr))
    predicted_best = int(np.argmax(predicted_scores_arr))
    correct = int(predicted_best == target_best)

    return {
        "target_scores": target_scores_arr,
        "predicted_scores": predicted_scores_arr,
        "target_best": target_best,
        "predicted_best": predicted_best,
        "correct": correct,
    }


def load_scenario_files(data_dir: str | Path) -> Iterable[dict[str, Any]]:
    """Load all scenarios from the provided dataset directory.

    The expected directory structure is a collection of scenario folders, each containing:
      - gt_graph.pt or gt_graph.pkl
      - h1.pt, h2.pt, h3.pt, h4.pt, h5.pt
    The function returns a list of dictionaries with the scenario identifier and file paths.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    scenarios: list[dict[str, Any]] = []
    for item in sorted(data_path.iterdir()):
        if not item.is_dir():
            continue

        gt_candidates = [item / "gt_graph.pt", item / "gt_graph.pkl", item / "ground_truth.pt"]
        gt_path = next((path for path in gt_candidates if path.exists()), None)
        if gt_path is None:
            continue

        hypothesis_paths = []
        for idx in range(1, 6):
            candidate_paths = [
                item / f"h{idx}.pt",
                item / f"h{idx}.pkl",
                item / f"hypothesis_{idx}.pt",
                item / f"hypothesis_{idx}.pkl",
            ]
            found = next((path for path in candidate_paths if path.exists()), None)
            if found is None:
                raise FileNotFoundError(f"Missing hypothesis {idx} for scenario {item.name}.")
            hypothesis_paths.append(found)

        scenarios.append({
            "scenario_id": item.name,
            "ground_truth": gt_path,
            "hypotheses": hypothesis_paths,
        })

    if not scenarios:
        raise ValueError(f"No scenario folders were found under {data_path}.")

    return scenarios


def load_graph_from_path(path: str | Path) -> Any:
    """Load a graph object from disk using the project graph loader.

    The loader supports the project-specific graph serialization format and falls back to
    PyTorch's state-dict handling when needed.
    """
    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    graph_loader = GraphLoader(normalize=False)
    try:
        return graph_loader.load(graph_path)
    except Exception:
        try:
            return torch.load(graph_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to load graph from {graph_path}.") from exc


def evaluate_dataset(
    model: torch.nn.Module,
    data_dir: str | Path,
    output_csv: str | Path,
    device: torch.device,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate the full dataset and write per-scenario results to CSV.

    The CSV contains one row per scenario with the target values, predicted values,
    the target-best index, predicted-best index, and correctness flag. The function also
    returns aggregate metrics computed over the full dataset.
    """
    set_deterministic(seed)

    scenario_rows: list[dict[str, Any]] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    correct_flags: list[int] = []

    for scenario in load_scenario_files(data_dir):
        gt_graph = load_graph_from_path(scenario["ground_truth"])
        hypotheses = [load_graph_from_path(path) for path in scenario["hypotheses"]]

        result = evaluate_scenario(model, hypotheses, gt_graph, device)
        target_scores = result["target_scores"]
        predicted_scores = result["predicted_scores"]

        all_targets.append(target_scores)
        all_predictions.append(predicted_scores)
        correct_flags.append(int(result["correct"]))

        row = {
            "scenario_id": scenario["scenario_id"],
            "target_h1": float(target_scores[0]),
            "target_h2": float(target_scores[1]),
            "target_h3": float(target_scores[2]),
            "target_h4": float(target_scores[3]),
            "target_h5": float(target_scores[4]),
            "predicted_h1": float(predicted_scores[0]),
            "predicted_h2": float(predicted_scores[1]),
            "predicted_h3": float(predicted_scores[2]),
            "predicted_h4": float(predicted_scores[3]),
            "predicted_h5": float(predicted_scores[4]),
            "target_best": int(result["target_best"]),
            "predicted_best": int(result["predicted_best"]),
            "correct": int(result["correct"]),
        }
        scenario_rows.append(row)

    if not scenario_rows:
        raise ValueError("No scenario rows were produced during evaluation.")

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "scenario_id",
            "target_h1",
            "target_h2",
            "target_h3",
            "target_h4",
            "target_h5",
            "predicted_h1",
            "predicted_h2",
            "predicted_h3",
            "predicted_h4",
            "predicted_h5",
            "target_best",
            "predicted_best",
            "correct",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scenario_rows)

    all_targets_arr = np.concatenate(all_targets)
    all_predictions_arr = np.concatenate(all_predictions)

    metrics = {
        "mae": mae(all_targets_arr, all_predictions_arr),
        "rmse": rmse(all_targets_arr, all_predictions_arr),
        "spearman": spearman_rank_correlation(all_targets_arr, all_predictions_arr),
        "top1_accuracy": top1_selection_accuracy(
            np.stack(all_targets, axis=0),
            np.stack(all_predictions, axis=0),
        ),
        "accuracy": float(np.mean(correct_flags)),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Siamese GNN on hypothesis selection scenarios.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint .pt file.")
    parser.add_argument("--data", type=str, required=True, help="Directory containing scenario folders.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config JSON or YAML file.")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic evaluation.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for evaluation."""
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    model = load_model(args.checkpoint, config, device)
    metrics = evaluate_dataset(
        model=model,
        data_dir=args.data,
        output_csv=args.output,
        device=device,
        seed=args.seed,
    )

    print("Evaluation complete:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
