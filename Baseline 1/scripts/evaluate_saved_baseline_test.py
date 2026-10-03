"""Evaluate a saved baseline checkpoint on the test split and save its metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import similarity_targets_from_metadata
from src.evaluation.evaluate import (
    load_config,
    load_graph_from_path,
    load_model,
    load_scenario_files,
    set_deterministic,
)
from src.evaluation.metrics import (
    mae,
    rmse,
    spearman_rank_correlation,
    top1_selection_accuracy,
)


CSV_FIELDS = (
    "scene_id",
    "target_h1",
    "target_h2",
    "target_h3",
    "target_h4",
    "target_h5",
    "pred_h1",
    "pred_h2",
    "pred_h3",
    "pred_h4",
    "pred_h5",
    "target_best",
    "predicted_best",
    "correct",
)


def evaluate_test_split(
    checkpoint_path: Path,
    config_path: Path,
    test_path: Path,
    output_directory: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate only the supplied test scene directory and write result artifacts."""
    config = load_config(config_path)
    target_config = config.get("similarity_target", {})
    if not isinstance(target_config, dict):
        raise ValueError("Configuration section 'similarity_target' must be a mapping.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_deterministic(seed)
    model = load_model(checkpoint_path, config, device)

    sigma_translation = float(target_config.get("sigma_translation", 1.0))
    sigma_rotation = float(target_config.get("sigma_rotation", 1.0))
    rows: list[dict[str, Any]] = []
    scene_targets: list[np.ndarray] = []
    scene_predictions: list[np.ndarray] = []
    target_best_counts = Counter({f"H{index}": 0 for index in range(1, 6)})

    for scenario in load_scenario_files(test_path):
        with Path(scenario["metadata"]).open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError(f"Scene metadata must be a JSON object: {scenario['metadata']}")

        targets = np.asarray(
            similarity_targets_from_metadata(
                metadata,
                sigma_translation=sigma_translation,
                sigma_rotation=sigma_rotation,
            ),
            dtype=np.float64,
        )
        if targets.shape != (5,) or not np.isfinite(targets).all():
            raise ValueError(f"Invalid five-hypothesis target similarities in {scenario['metadata']}.")
        if np.any((targets < 0.0) | (targets > 1.0)):
            raise ValueError(f"Target similarities outside [0, 1] in {scenario['metadata']}.")

        ground_truth = load_graph_from_path(scenario["ground_truth"]).to(device)
        predictions: list[float] = []
        for hypothesis_path in scenario["hypotheses"]:
            hypothesis = load_graph_from_path(hypothesis_path).to(device)
            with torch.no_grad():
                score = model(hypothesis, ground_truth).detach().cpu().reshape(-1)
            if score.numel() != 1:
                raise ValueError(
                    f"Expected one prediction per hypothesis in {scenario['scenario_id']}, "
                    f"got shape {tuple(score.shape)}."
                )
            predictions.append(float(score.item()))

        predicted = np.asarray(predictions, dtype=np.float64)
        if not np.isfinite(predicted).all() or np.any((predicted < 0.0) | (predicted > 1.0)):
            raise ValueError(f"Invalid predicted similarities in {scenario['scenario_id']}.")

        target_best = int(np.argmax(targets)) + 1
        predicted_best = int(np.argmax(predicted)) + 1
        correct = int(target_best == predicted_best)
        target_best_counts[f"H{target_best}"] += 1
        scene_targets.append(targets)
        scene_predictions.append(predicted)
        rows.append(
            {
                "scene_id": scenario["scenario_id"],
                **{f"target_h{index + 1}": float(value) for index, value in enumerate(targets)},
                **{f"pred_h{index + 1}": float(value) for index, value in enumerate(predicted)},
                "target_best": target_best,
                "predicted_best": predicted_best,
                "correct": correct,
            }
        )

    if not rows:
        raise ValueError(f"No test scenes found under {test_path}.")

    flat_targets = np.concatenate(scene_targets)
    flat_predictions = np.concatenate(scene_predictions)
    metrics: dict[str, Any] = {
        "MAE": mae(flat_targets, flat_predictions),
        "RMSE": rmse(flat_targets, flat_predictions),
        "Spearman": spearman_rank_correlation(flat_targets, flat_predictions),
        "Top1_accuracy": top1_selection_accuracy(
            np.stack(scene_targets),
            np.stack(scene_predictions),
        ),
        "random_baseline": 1 / 5,
        "num_scenes": len(rows),
        "target_best_distribution": {
            f"H{index}": target_best_counts[f"H{index}"]
            for index in range(1, 6)
        },
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = output_directory / "test_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = output_directory / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test scenes: {len(rows)}")
    for name in ("MAE", "RMSE", "Spearman", "Top1_accuracy", "random_baseline"):
        print(f"{name}: {metrics[name]}")
    print(f"Target-best distribution: {metrics['target_best_distribution']}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved baseline checkpoint on test data only."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/baseline_best.pt"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full_baseline.yaml"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/full_synthetic_benchmark/test"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        expanded = path.expanduser()
        return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()

    evaluate_test_split(
        checkpoint_path=resolve(args.checkpoint),
        config_path=resolve(args.config),
        test_path=resolve(args.test),
        output_directory=resolve(args.output),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
