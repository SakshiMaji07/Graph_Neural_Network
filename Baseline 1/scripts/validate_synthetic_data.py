"""Validate synthetic SLAM graph datasets before baseline training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_loader import GraphLoader
from src.data.graph_schema import NODE_FEATURE_DIM
from src.data.preprocessing import (
    _wrap_angle,
    pose_similarity,
    rotation_error,
    similarity_targets_from_metadata,
    translation_error,
)

SPLITS = ("train", "val", "test")
HYPOTHESIS_NAMES = tuple(f"h{index}" for index in range(1, 6))
MANIFEST_FIELDS = {
    "scene_id",
    "hypothesis_path",
    "ground_truth_path",
    "hypothesis_id",
    "similarity",
}
SIMILARITY_TOLERANCE = 1e-9


def _record(errors: list[str], location: str | Path, message: str) -> None:
    errors.append(f"{location}: {message}")


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        _record(errors, path, f"unable to read valid JSON ({exc})")
        return None
    if not isinstance(metadata, dict):
        _record(errors, path, "metadata must be a JSON object")
        return None
    return metadata


def _finite_number(value: Any, location: str, errors: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _record(errors, location, f"expected a numeric value, got {value!r}")
        return None
    if not math.isfinite(number):
        _record(errors, location, f"value must be finite, got {number!r}")
        return None
    return number


def _validate_pose_labels(
    scene_directory: Path,
    scene_id: str,
    metadata: dict[str, Any],
    errors: list[str],
) -> tuple[list[float], int] | None:
    if metadata.get("scene_id") != scene_id:
        _record(errors, scene_directory / "metadata.json", "scene_id does not match the directory name")
    if not isinstance(metadata.get("random_seed"), int):
        _record(errors, scene_directory / "metadata.json", "random_seed must be an integer")

    sigmas = metadata.get("similarity_sigmas")
    if not isinstance(sigmas, dict):
        _record(errors, scene_directory / "metadata.json", "similarity_sigmas mapping is missing")
        return None
    sigma_translation = _finite_number(
        sigmas.get("translation"),
        scene_directory / "metadata.json:similarity_sigmas.translation",
        errors,
    )
    sigma_rotation = _finite_number(
        sigmas.get("rotation"),
        scene_directory / "metadata.json:similarity_sigmas.rotation",
        errors,
    )
    if sigma_translation is None or sigma_rotation is None:
        return None
    if sigma_translation <= 0.0 or sigma_rotation <= 0.0:
        _record(errors, scene_directory / "metadata.json", "similarity sigmas must be positive")
        return None

    hypotheses = metadata.get("hypotheses")
    if not isinstance(hypotheses, dict):
        _record(errors, scene_directory / "metadata.json", "hypotheses must be a mapping with h1 through h5")
        return None
    if set(hypotheses) != set(HYPOTHESIS_NAMES):
        _record(
            errors,
            scene_directory / "metadata.json",
            f"hypotheses must contain exactly {list(HYPOTHESIS_NAMES)}, got {sorted(hypotheses)}",
        )
        return None

    try:
        recomputed_scores = similarity_targets_from_metadata(
            metadata,
            sigma_translation=sigma_translation,
            sigma_rotation=sigma_rotation,
        )
    except (TypeError, ValueError, KeyError) as exc:
        _record(errors, scene_directory / "metadata.json", f"invalid pose metadata ({exc})")
        return None

    gt_pose = metadata.get("gt_pose")
    if not isinstance(gt_pose, dict):
        _record(errors, scene_directory / "metadata.json", "gt_pose must be a mapping")
        return None

    for index, hypothesis_name in enumerate(HYPOTHESIS_NAMES):
        hypothesis = hypotheses[hypothesis_name]
        if not isinstance(hypothesis, dict):
            _record(errors, scene_directory / "metadata.json", f"{hypothesis_name} must be a mapping")
            continue
        pose = hypothesis.get("pose")
        if not isinstance(pose, dict):
            _record(errors, scene_directory / "metadata.json", f"{hypothesis_name}.pose must be a mapping")
            continue
        try:
            translation = translation_error(pose["position"], gt_pose["position"])
            wrapped_rotation = float(
                _wrap_angle(float(pose["rotation"]) - float(gt_pose["rotation"]))
            )
            angular_error = rotation_error(pose["rotation"], gt_pose["rotation"])
        except (KeyError, TypeError, ValueError) as exc:
            _record(errors, scene_directory / "metadata.json", f"invalid {hypothesis_name} pose ({exc})")
            continue

        stored_translation = _finite_number(
            hypothesis.get("translation_error"),
            f"{scene_directory.name}/{hypothesis_name}.translation_error",
            errors,
        )
        stored_rotation = _finite_number(
            hypothesis.get("rotation_error"),
            f"{scene_directory.name}/{hypothesis_name}.rotation_error",
            errors,
        )
        stored_similarity = _finite_number(
            hypothesis.get("similarity"),
            f"{scene_directory.name}/{hypothesis_name}.similarity",
            errors,
        )
        if stored_translation is not None and not math.isclose(
            stored_translation, translation, rel_tol=0.0, abs_tol=SIMILARITY_TOLERANCE
        ):
            _record(
                errors,
                scene_directory / "metadata.json",
                f"{hypothesis_name} translation_error does not match its pose",
            )
        if stored_rotation is not None and not math.isclose(
            stored_rotation, wrapped_rotation, rel_tol=0.0, abs_tol=SIMILARITY_TOLERANCE
        ):
            _record(
                errors,
                scene_directory / "metadata.json",
                f"{hypothesis_name} rotation_error is not the wrapped pose difference",
            )
        if stored_similarity is not None:
            if not 0.0 <= stored_similarity <= 1.0:
                _record(errors, scene_directory / "metadata.json", f"{hypothesis_name} similarity is outside [0, 1]")
            if not math.isclose(
                stored_similarity,
                recomputed_scores[index],
                rel_tol=0.0,
                abs_tol=SIMILARITY_TOLERANCE,
            ):
                _record(
                    errors,
                    scene_directory / "metadata.json",
                    f"{hypothesis_name} similarity does not match recomputed pose similarity",
                )
        expected_similarity = pose_similarity(
            translation,
            angular_error,
            sigma_translation=sigma_translation,
            sigma_rotation=sigma_rotation,
        )
        if not math.isclose(
            expected_similarity,
            recomputed_scores[index],
            rel_tol=0.0,
            abs_tol=SIMILARITY_TOLERANCE,
        ):
            _record(
                errors,
                scene_directory / "metadata.json",
                f"{hypothesis_name} recomputed similarity is inconsistent",
            )

    target_best = int(np.argmax(np.asarray(recomputed_scores, dtype=np.float64))) + 1
    recorded_best = metadata.get("target_best")
    if recorded_best is not None and recorded_best != target_best:
        _record(
            errors,
            scene_directory / "metadata.json",
            f"target_best H{recorded_best} does not equal similarity argmax H{target_best}",
        )
    return recomputed_scores, target_best


def _validate_graph(
    graph_path: Path,
    graph_loader: GraphLoader,
    errors: list[str],
) -> tuple[int, int] | None:
    try:
        with np.load(graph_path, allow_pickle=False) as archive:
            missing = {"node_features", "edge_index"} - set(archive.files)
            if missing:
                _record(errors, graph_path, f"missing required arrays: {', '.join(sorted(missing))}")
                return None
            node_features = np.asarray(archive["node_features"])
            edge_index = np.asarray(archive["edge_index"])

            if node_features.ndim != 2:
                _record(errors, graph_path, f"node_features must be rank 2, got {node_features.shape}")
                return None
            if node_features.shape[0] <= 0:
                _record(errors, graph_path, "node count must be greater than zero")
                return None
            if node_features.shape[1] != NODE_FEATURE_DIM:
                _record(
                    errors,
                    graph_path,
                    f"expected {NODE_FEATURE_DIM} node features, got {node_features.shape[1]}",
                )
            if not np.isfinite(node_features).all():
                _record(errors, graph_path, "node_features contain NaN or Inf")

            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                _record(errors, graph_path, f"edge_index must have shape [2, E], got {edge_index.shape}")
                return None
            if not np.issubdtype(edge_index.dtype, np.integer):
                _record(errors, graph_path, f"edge_index must have an integer dtype, got {edge_index.dtype}")
            elif edge_index.size and (
                edge_index.min() < 0 or edge_index.max() >= node_features.shape[0]
            ):
                _record(
                    errors,
                    graph_path,
                    f"edge indices must be in [0, {node_features.shape[0] - 1}]",
                )
            if not np.isfinite(edge_index).all():
                _record(errors, graph_path, "edge_index contains NaN or Inf")
    except (OSError, ValueError, TypeError) as exc:
        _record(errors, graph_path, f"unable to read valid NPZ arrays ({exc})")
        return None

    try:
        graph = graph_loader.load(graph_path)
    except Exception as exc:
        _record(errors, graph_path, f"project GraphLoader could not load graph ({exc})")
        return None

    if graph.x.dim() != 2 or graph.x.size(1) != NODE_FEATURE_DIM:
        _record(errors, graph_path, f"GraphLoader returned invalid x shape {tuple(graph.x.shape)}")
        return None
    if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
        _record(
            errors,
            graph_path,
            f"GraphLoader returned invalid edge_index shape {tuple(graph.edge_index.shape)}",
        )
        return None
    if not torch_is_finite(graph.x) or not torch_is_finite(graph.edge_index):
        _record(errors, graph_path, "GraphLoader returned values containing NaN or Inf")
        return None
    if graph.edge_index.numel() and (
        int(graph.edge_index.min()) < 0
        or int(graph.edge_index.max()) >= int(graph.x.size(0))
    ):
        _record(errors, graph_path, "GraphLoader returned edge indices outside the node range")
        return None
    return int(graph.x.size(0)), int(graph.edge_index.size(1))


def torch_is_finite(tensor: Any) -> bool:
    """Use torch's tensor checks without exposing a second graph-loading path."""
    import torch

    return bool(torch.isfinite(tensor).all().item())


def _resolve_manifest_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def _validate_manifest(
    dataset_root: Path,
    split: str,
    manifest_path: Path,
    scene_data: dict[str, dict[str, Any]],
    errors: list[str],
) -> int:
    if not manifest_path.is_file():
        _record(errors, manifest_path, "pair manifest is missing")
        return 0

    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or [])
            missing_fields = MANIFEST_FIELDS - fieldnames
            if missing_fields:
                _record(
                    errors,
                    manifest_path,
                    f"missing CSV columns: {', '.join(sorted(missing_fields))}",
                )
                return 0
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        _record(errors, manifest_path, f"unable to read CSV ({exc})")
        return 0

    rows_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows_by_pair: dict[tuple[str, str], int] = Counter()
    for row_number, row in enumerate(rows, start=2):
        missing = [field for field in MANIFEST_FIELDS if not (row.get(field) or "").strip()]
        if missing:
            _record(errors, f"{manifest_path}:{row_number}", f"missing values for {', '.join(sorted(missing))}")
            continue

        scene_id = row["scene_id"].strip()
        hypothesis_id = row["hypothesis_id"].strip().lower()
        if hypothesis_id.startswith("h") and hypothesis_id[1:].isdigit():
            hypothesis_name = hypothesis_id
        else:
            _record(errors, f"{manifest_path}:{row_number}", f"invalid hypothesis_id {row['hypothesis_id']!r}")
            continue
        if hypothesis_name not in HYPOTHESIS_NAMES:
            _record(errors, f"{manifest_path}:{row_number}", f"unexpected hypothesis_id {hypothesis_name!r}")
            continue

        hypothesis_path = _resolve_manifest_path(dataset_root, row["hypothesis_path"].strip())
        ground_truth_path = _resolve_manifest_path(dataset_root, row["ground_truth_path"].strip())
        if not hypothesis_path.is_file():
            _record(errors, f"{manifest_path}:{row_number}", f"hypothesis path does not exist: {hypothesis_path}")
        if not ground_truth_path.is_file():
            _record(errors, f"{manifest_path}:{row_number}", f"ground-truth path does not exist: {ground_truth_path}")

        manifest_score = _finite_number(
            row["similarity"],
            f"{manifest_path}:{row_number}:similarity",
            errors,
        )
        if manifest_score is not None and not 0.0 <= manifest_score <= 1.0:
            _record(errors, f"{manifest_path}:{row_number}", "similarity is outside [0, 1]")

        expected_scene = scene_data.get(scene_id)
        if expected_scene is None:
            _record(errors, f"{manifest_path}:{row_number}", f"scene_id {scene_id!r} is not in {split}")
        else:
            expected_hypothesis_path = (expected_scene["directory"] / f"{hypothesis_name}.npz").resolve()
            expected_gt_path = (expected_scene["directory"] / "gt.npz").resolve()
            if hypothesis_path != expected_hypothesis_path:
                _record(
                    errors,
                    f"{manifest_path}:{row_number}",
                    f"hypothesis path does not match {scene_id}/{hypothesis_name}.npz",
                )
            if ground_truth_path != expected_gt_path:
                _record(
                    errors,
                    f"{manifest_path}:{row_number}",
                    f"ground_truth_path does not point to {scene_id}/gt.npz",
                )
            scores = expected_scene.get("scores")
            if scores is not None and manifest_score is not None:
                expected_score = scores[int(hypothesis_name[1:]) - 1]
                if not math.isclose(
                    manifest_score,
                    expected_score,
                    rel_tol=0.0,
                    abs_tol=SIMILARITY_TOLERANCE,
                ):
                    _record(errors, f"{manifest_path}:{row_number}", "similarity does not match scene metadata")

        rows_by_scene[scene_id].append(row)
        pair_key = (scene_id, hypothesis_name)
        rows_by_pair[pair_key] += 1
        if rows_by_pair[pair_key] > 1:
            _record(errors, f"{manifest_path}:{row_number}", f"duplicate pair {scene_id}/{hypothesis_name}")

    for scene_id in scene_data:
        scene_rows = rows_by_scene.get(scene_id, [])
        if len(scene_rows) != 5:
            _record(
                errors,
                manifest_path,
                f"scene {scene_id} contributes {len(scene_rows)} pairs; expected exactly 5",
            )
        found_hypotheses = {row.get("hypothesis_id", "").strip().lower() for row in scene_rows}
        if found_hypotheses != set(HYPOTHESIS_NAMES):
            _record(
                errors,
                manifest_path,
                f"scene {scene_id} must reference h1 through h5 exactly once",
            )
    return len(rows)


def validate_dataset(
    dataset_root: str | Path,
    graph_loader: GraphLoader | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    errors: list[str] = []
    loader = graph_loader or GraphLoader(normalize=False, strict=True)
    scenes_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    scene_ids_by_split: dict[str, set[str]] = {}
    node_counts: list[int] = []
    edge_counts: list[int] = []
    similarities_all: list[float] = []
    best_distribution: Counter[str] = Counter({name.upper(): 0 for name in HYPOTHESIS_NAMES})
    best_hypotheses_seen: set[int] = set()

    for split in SPLITS:
        split_directory = root / split
        if not split_directory.is_dir():
            _record(errors, split_directory, "split directory is missing")
            scenes_by_split[split] = {}
            scene_ids_by_split[split] = set()
            continue

        scene_directories = sorted(path for path in split_directory.iterdir() if path.is_dir())
        if not scene_directories:
            _record(errors, split_directory, "split contains no scene directories")

        split_scenes: dict[str, dict[str, Any]] = {}
        for scene_directory in scene_directories:
            scene_id = scene_directory.name
            expected_files = {
                "gt.npz",
                "metadata.json",
                *(f"{name}.npz" for name in HYPOTHESIS_NAMES),
            }
            actual_entries = {path.name for path in scene_directory.iterdir()}
            if actual_entries != expected_files:
                _record(
                    errors,
                    scene_directory,
                    f"expected exactly {sorted(expected_files)}, found entries {sorted(actual_entries)}",
                )

            scene_entry: dict[str, Any] = {"directory": scene_directory, "scores": None}
            metadata_path = scene_directory / "metadata.json"
            metadata = _read_json(metadata_path, errors) if metadata_path.is_file() else None
            if metadata is None:
                if not metadata_path.exists():
                    _record(errors, metadata_path, "required metadata file is missing")
            else:
                labels = _validate_pose_labels(scene_directory, scene_id, metadata, errors)
                if labels is not None:
                    scene_entry["scores"], target_best = labels
                    similarities_all.extend(labels[0])
                    best_distribution[f"H{target_best}"] += 1
                    best_hypotheses_seen.add(target_best)
                elif isinstance(metadata.get("hypotheses"), dict):
                    for name in HYPOTHESIS_NAMES:
                        hypothesis = metadata["hypotheses"].get(name)
                        if isinstance(hypothesis, dict):
                            value = _finite_number(
                                hypothesis.get("similarity"),
                                f"{scene_directory.name}/{name}.similarity",
                                errors,
                            )
                            if value is not None:
                                similarities_all.append(value)

            for graph_name in ("gt.npz", *(f"{name}.npz" for name in HYPOTHESIS_NAMES)):
                graph_path = scene_directory / graph_name
                if not graph_path.is_file():
                    _record(errors, graph_path, "required graph file is missing")
                    continue
                graph_stats = _validate_graph(graph_path, loader, errors)
                if graph_stats is not None:
                    node_counts.append(graph_stats[0])
                    edge_counts.append(graph_stats[1])

            if scene_id in split_scenes:
                _record(errors, scene_directory, f"duplicate scene directory ID {scene_id!r}")
            split_scenes[scene_id] = scene_entry

        scenes_by_split[split] = split_scenes
        scene_ids_by_split[split] = set(split_scenes)

    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            overlap = scene_ids_by_split.get(left_split, set()) & scene_ids_by_split.get(right_split, set())
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                _record(
                    errors,
                    root,
                    f"scene leakage between {left_split} and {right_split}: {examples}",
                )

    pair_counts: dict[str, int] = {}
    for split in SPLITS:
        pair_counts[split] = _validate_manifest(
            root,
            split,
            root / f"{split}_pairs.csv",
            scenes_by_split.get(split, {}),
            errors,
        )

    if not best_hypotheses_seen:
        _record(errors, root, "no target-best hypotheses could be verified")
    elif best_hypotheses_seen == {1}:
        _record(errors, root, "target-best is H1 for every scene; hypothesis ordering is not randomized")

    summary = {
        "scenes_per_split": {split: len(scenes_by_split.get(split, {})) for split in SPLITS},
        "pairs_per_split": pair_counts,
        "nodes": _statistics(node_counts),
        "edges": _statistics(edge_counts),
        "similarities": _statistics(similarities_all),
        "target_best": {f"H{index}": best_distribution[f"H{index}"] for index in range(1, 6)},
        "errors": errors,
    }
    return summary


def _statistics(values: list[int] | list[float]) -> dict[str, int | float] | None:
    if not values:
        return None
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _print_statistics(name: str, values: dict[str, int | float] | None) -> None:
    if values is None:
        print(f"{name}: unavailable (no valid values)")
        return
    print(
        f"{name}: min={values['min']}, mean={values['mean']:.3f}, max={values['max']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the synthetic SLAM graph dataset before training."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Dataset root containing train/, val/, test/, and split pair CSVs.",
    )
    args = parser.parse_args()

    try:
        summary = validate_dataset(args.data)
    except Exception as exc:
        print(f"DATASET VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("Dataset validation summary:")
    for split in SPLITS:
        print(
            f"{split}: scenes={summary['scenes_per_split'][split]}, "
            f"pairs={summary['pairs_per_split'][split]}"
        )
    _print_statistics("Node count", summary["nodes"])
    _print_statistics("Edge count", summary["edges"])
    _print_statistics("Similarity", summary["similarities"])
    print(
        "Target-best distribution: "
        + ", ".join(f"{name}={count}" for name, count in summary["target_best"].items())
    )

    if summary["errors"]:
        print(f"DATASET VALIDATION FAILED ({len(summary['errors'])} issue(s)):", file=sys.stderr)
        for error in summary["errors"]:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("DATASET VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
