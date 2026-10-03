"""Generate reproducible synthetic SLAM-like graph scenes for the Siamese baseline."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_loader import GraphLoader
from src.data.graph_schema import NODE_FEATURE_DIM
from src.data.preprocessing import _wrap_angle, pose_similarity, rotation_error, translation_error

HYPOTHESIS_COUNT = 5
NEIGHBOR_COUNT = 3
NEIGHBOR_RADIUS = 2.0
GENERATOR_MARKER = "synthetic_slam_baseline_v1"
QUALITY_PROFILES = (
    (0.12, 0.04, 0.015),
    (0.32, 0.12, 0.025),
    (0.58, 0.22, 0.040),
    (0.88, 0.34, 0.060),
    (1.15, 0.48, 0.080),
)
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}
SPLIT_NAMES = ("train", "val", "test")


def _load_configured_sigmas() -> tuple[float, float]:
    """Read the two scalar target parameters from the existing baseline YAML."""
    config_path = PROJECT_ROOT / "configs" / "baseline_siamese.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Baseline configuration not found: {config_path}")
    config_text = config_path.read_text(encoding="utf-8")
    section_match = re.search(
        r"(?m)^similarity_target:[ \t]*\r?\n((?:(?:[ \t]+.*)?\r?\n)*)",
        config_text,
    )
    if section_match is None:
        return 1.0, 1.0

    values: dict[str, float] = {}
    for name in ("sigma_translation", "sigma_rotation"):
        value_match = re.search(rf"(?m)^[ \t]+{name}:[ \t]*(.*?)\s*$", section_match.group(1))
        if value_match is None:
            continue
        value_text = value_match.group(1).split("#", maxsplit=1)[0].strip()
        try:
            values[name] = float(value_text)
        except ValueError as exc:
            raise ValueError(
                f"Configuration value '{name}' must be a number, got {value_text!r}."
            ) from exc
    sigma_translation = values.get("sigma_translation", 1.0)
    sigma_rotation = values.get("sigma_rotation", 1.0)
    if not math.isfinite(sigma_translation) or not math.isfinite(sigma_rotation):
        raise ValueError("Configured similarity sigmas must be finite.")
    if sigma_translation <= 0.0 or sigma_rotation <= 0.0:
        raise ValueError("Configured similarity sigmas must both be positive.")
    return sigma_translation, sigma_rotation


def _rotation_matrix(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _generate_landmark_layout(rng: np.random.Generator, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a room-boundary layout with a small interior landmark cluster."""
    width = float(rng.uniform(8.0, 14.0))
    height = float(rng.uniform(7.0, 12.0))
    perimeter_count = node_count - max(3, int(round(node_count * 0.18)))
    perimeter = 2.0 * (width + height)

    arc_positions = (
        np.arange(perimeter_count, dtype=np.float64) + rng.uniform(-0.18, 0.18, perimeter_count)
    ) * (perimeter / perimeter_count)
    arc_positions %= perimeter
    boundary_points: list[tuple[float, float]] = []
    for distance in arc_positions:
        if distance < width:
            point = (distance, 0.0)
        elif distance < width + height:
            point = (width, distance - width)
        elif distance < 2.0 * width + height:
            point = (2.0 * width + height - distance, height)
        else:
            point = (0.0, perimeter - distance)

        tangent_jitter = float(rng.normal(0.0, 0.06))
        normal_jitter = float(rng.normal(0.0, 0.04))
        if distance < width or distance >= 2.0 * width + height:
            boundary_points.append((point[0] + tangent_jitter + normal_jitter, point[1] + tangent_jitter))
        else:
            boundary_points.append((point[0] + tangent_jitter, point[1] + tangent_jitter + normal_jitter))

    interior_count = node_count - perimeter_count
    cluster_centers = np.asarray(
        [
            [width * 0.35, height * 0.38],
            [width * 0.68, height * 0.62],
        ],
        dtype=np.float64,
    )
    cluster_indices = rng.integers(0, len(cluster_centers), size=interior_count)
    interior_points = cluster_centers[cluster_indices] + rng.normal(
        loc=0.0,
        scale=0.35,
        size=(interior_count, 2),
    )

    points = np.concatenate(
        [np.asarray(boundary_points, dtype=np.float64), interior_points],
        axis=0,
    )
    point_types = np.concatenate(
        [np.zeros(perimeter_count, dtype=np.float32), np.ones(interior_count, dtype=np.float32)]
    )
    return points, point_types


def _graph_arrays(points: np.ndarray, point_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create schema-ordered 8D node features and a symmetric kNN COO edge index."""
    node_count = points.shape[0]
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    features = np.zeros((node_count, NODE_FEATURE_DIM), dtype=np.float32)
    features[:, 0] = point_types
    features[:, 1] = np.arctan2(points[:, 1], points[:, 0]).astype(np.float32)
    features[:, 2] = pairwise.min(axis=1).astype(np.float32)
    features[:, 3] = ((pairwise <= NEIGHBOR_RADIUS).sum(axis=1)).astype(np.float32)

    edges: set[tuple[int, int]] = set()
    for node_index in range(node_count):
        nearest = np.lexsort((np.arange(node_count), pairwise[node_index]))[:NEIGHBOR_COUNT]
        for neighbor_index in nearest:
            neighbor = int(neighbor_index)
            edges.add((node_index, neighbor))
            edges.add((neighbor, node_index))

        local_points = points[np.concatenate(([node_index], nearest))]
        covariance = np.cov(local_points.T, bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
        lambda_1 = float(eigenvalues[0])
        lambda_2 = float(eigenvalues[1])
        features[node_index, 4:7] = (lambda_1, lambda_2, 0.0)
        features[node_index, 7] = lambda_2 / lambda_1 if lambda_1 > 0.0 else 0.0

    edge_index = np.asarray(sorted(edges), dtype=np.int64).T
    if edge_index.shape[0] != 2:
        raise RuntimeError(f"Generated edge_index must have shape [2, E], got {edge_index.shape}.")
    if not np.isfinite(features).all():
        raise RuntimeError("Generated node features contain NaN or Inf values.")
    return features, edge_index


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write NPZ arrays with fixed ZIP metadata for byte-reproducible output."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def _pose(position: np.ndarray, rotation: float) -> dict[str, Any]:
    return {
        "position": [float(position[0]), float(position[1])],
        "rotation": float(rotation),
    }


def _scene_seed(seed: int, split_index: int, scene_index: int) -> int:
    sequence = np.random.SeedSequence([seed, split_index, scene_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _assert_output_is_managed(split_directory: Path, requested_scene_ids: set[str]) -> None:
    """Avoid overwriting unrelated files and detect stale generated scene folders."""
    if not split_directory.exists():
        return

    for entry in split_directory.iterdir():
        if not entry.is_dir() or not entry.name.startswith("scene_"):
            raise FileExistsError(
                f"Refusing to overwrite unrecognized data in '{split_directory}': {entry.name}"
            )
        metadata_path = entry / "metadata.json"
        if not metadata_path.is_file():
            raise FileExistsError(
                f"Refusing to overwrite an unmarked scene directory: {entry}"
            )
        with metadata_path.open("r", encoding="utf-8") as stream:
            existing_metadata = json.load(stream)
        if not isinstance(existing_metadata, dict) or existing_metadata.get("generator") != GENERATOR_MARKER:
            raise FileExistsError(
                f"Refusing to overwrite a scene not generated by this script: {entry}"
            )
        if entry.name not in requested_scene_ids:
            raise FileExistsError(
                f"Stale generated scene '{entry}' is outside this run's requested scene range. "
                "Choose a new --output directory to preserve existing data."
            )


def _generate_scene(
    scene_directory: Path,
    scene_id: str,
    split: str,
    seed: int,
    sigma_translation: float,
    sigma_rotation: float,
) -> int:
    scene_seed = _scene_seed(seed, SPLIT_INDEX[split], int(scene_id.removeprefix("scene_")))
    rng = np.random.default_rng(scene_seed)
    node_count = int(rng.integers(20, 51))
    local_points, point_types = _generate_landmark_layout(rng, node_count)

    gt_position = rng.uniform(-2.0, 2.0, size=2)
    gt_rotation = float(rng.uniform(-math.pi, math.pi))
    world_points = local_points @ _rotation_matrix(gt_rotation).T + gt_position
    gt_features, gt_edges = _graph_arrays(world_points, point_types)
    _write_deterministic_npz(
        scene_directory / "gt.npz",
        {"node_features": gt_features, "edge_index": gt_edges},
    )

    profile_order = rng.permutation(len(QUALITY_PROFILES))
    hypothesis_metadata: dict[str, Any] = {}
    ground_truth_pose = _pose(gt_position, gt_rotation)

    for hypothesis_number, profile_index in enumerate(profile_order, start=1):
        translation_magnitude, rotation_magnitude, landmark_noise_std = QUALITY_PROFILES[int(profile_index)]
        translation_direction = float(rng.uniform(-math.pi, math.pi))
        translation_offset = translation_magnitude * np.asarray(
            [math.cos(translation_direction), math.sin(translation_direction)],
            dtype=np.float64,
        )
        rotation_offset = float(rng.choice((-1.0, 1.0)) * rotation_magnitude)
        hypothesis_position = gt_position + translation_offset
        hypothesis_rotation = float(_wrap_angle(gt_rotation + rotation_offset))
        transformed_points = (
            local_points @ _rotation_matrix(hypothesis_rotation).T
            + hypothesis_position
            + rng.normal(0.0, landmark_noise_std, size=local_points.shape)
        )
        hypothesis_features, hypothesis_edges = _graph_arrays(transformed_points, point_types)
        _write_deterministic_npz(
            scene_directory / f"h{hypothesis_number}.npz",
            {"node_features": hypothesis_features, "edge_index": hypothesis_edges},
        )

        dt = translation_error(hypothesis_position, gt_position)
        dr_signed = float(_wrap_angle(hypothesis_rotation - gt_rotation))
        dr = rotation_error(hypothesis_rotation, gt_rotation)
        similarity = pose_similarity(
            dt,
            dr,
            sigma_translation=sigma_translation,
            sigma_rotation=sigma_rotation,
        )
        hypothesis_metadata[f"h{hypothesis_number}"] = {
            "pose": _pose(hypothesis_position, hypothesis_rotation),
            "translation_error": dt,
            "rotation_error": dr_signed,
            "similarity": similarity,
            "landmark_noise_std": float(landmark_noise_std),
        }

    metadata = {
        "generator": GENERATOR_MARKER,
        "scene_id": scene_id,
        "split": split,
        "random_seed": scene_seed,
        "gt_pose": ground_truth_pose,
        "hypotheses": hypothesis_metadata,
        "similarity_sigmas": {
            "translation": float(sigma_translation),
            "rotation": float(sigma_rotation),
        },
    }
    with (scene_directory / "metadata.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    ordered_similarities = [
        float(hypothesis_metadata[f"h{index}"]["similarity"])
        for index in range(1, HYPOTHESIS_COUNT + 1)
    ]
    return int(np.argmax(ordered_similarities))


def _write_pair_manifest(output_root: Path, split: str, rows: list[dict[str, Any]]) -> None:
    path = output_root / f"{split}_pairs.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "scene_id",
            "hypothesis_path",
            "ground_truth_path",
            "hypothesis_id",
            "similarity",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(
    output: str | Path,
    train_scenes: int = 800,
    val_scenes: int = 200,
    test_scenes: int = 200,
    seed: int = 42,
    sigma_translation: float | None = None,
    sigma_rotation: float | None = None,
) -> dict[str, Any]:
    if min(train_scenes, val_scenes, test_scenes) < 1:
        raise ValueError("Each split must request at least one scene.")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")

    configured_sigma_translation, configured_sigma_rotation = _load_configured_sigmas()
    sigma_translation = (
        configured_sigma_translation if sigma_translation is None else float(sigma_translation)
    )
    sigma_rotation = configured_sigma_rotation if sigma_rotation is None else float(sigma_rotation)
    if not math.isfinite(sigma_translation) or not math.isfinite(sigma_rotation):
        raise ValueError("Similarity sigmas must be finite.")
    if sigma_translation <= 0.0 or sigma_rotation <= 0.0:
        raise ValueError("Similarity sigmas must both be positive.")

    output_root = Path(output).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scene_counts = {
        "train": train_scenes,
        "val": val_scenes,
        "test": test_scenes,
    }
    starts: dict[str, int] = {}
    next_index = 0
    for split in SPLIT_NAMES:
        starts[split] = next_index
        next_index += scene_counts[split]
    requested_ids = {
        split: {
            f"scene_{index:04d}"
            for index in range(starts[split], starts[split] + scene_counts[split])
        }
        for split in SPLIT_NAMES
    }
    for split in SPLIT_NAMES:
        _assert_output_is_managed(output_root / split, requested_ids[split])

    graph_loader = GraphLoader(normalize=False, strict=True)
    node_counts_all: list[int] = []
    similarities_all: list[float] = []
    best_counts = {f"h{index}": 0 for index in range(1, HYPOTHESIS_COUNT + 1)}
    pair_count = 0
    graph_count = 0
    split_summaries: dict[str, dict[str, int]] = {}

    for split in SPLIT_NAMES:
        split_directory = output_root / split
        split_directory.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []
        split_scene_count = scene_counts[split]

        for local_index in range(split_scene_count):
            scene_id = f"scene_{starts[split] + local_index:04d}"
            scene_directory = split_directory / scene_id
            scene_directory.mkdir(parents=True, exist_ok=True)
            target_best = _generate_scene(
                scene_directory,
                scene_id,
                split,
                seed,
                sigma_translation,
                sigma_rotation,
            )

            expected_names = {
                "gt.npz",
                "metadata.json",
                *(f"h{index}.npz" for index in range(1, HYPOTHESIS_COUNT + 1)),
            }
            actual_names = {path.name for path in scene_directory.iterdir()}
            if actual_names != expected_names:
                raise RuntimeError(
                    f"Scene '{scene_id}' must contain exactly {sorted(expected_names)}, "
                    f"found {sorted(actual_names)}."
                )

            for graph_name in ("gt.npz", *(f"h{index}.npz" for index in range(1, 6))):
                graph = graph_loader.load(scene_directory / graph_name)
                if graph.x.size(1) != NODE_FEATURE_DIM:
                    raise RuntimeError(
                        f"{scene_id}/{graph_name}: expected {NODE_FEATURE_DIM} node features, "
                        f"found {graph.x.size(1)}."
                    )
                node_counts_all.append(int(graph.x.size(0)))
                graph_count += 1

            with (scene_directory / "metadata.json").open("r", encoding="utf-8") as stream:
                scene_metadata = json.load(stream)
            if len(scene_metadata["hypotheses"]) != HYPOTHESIS_COUNT:
                raise RuntimeError(f"Scene '{scene_id}' does not have exactly five hypotheses.")

            for index in range(1, HYPOTHESIS_COUNT + 1):
                hypothesis_id = f"H{index}"
                hypothesis_data = scene_metadata["hypotheses"][f"h{index}"]
                manifest_rows.append(
                    {
                        "scene_id": scene_id,
                        "hypothesis_path": f"{split}/{scene_id}/h{index}.npz",
                        "ground_truth_path": f"{split}/{scene_id}/gt.npz",
                        "hypothesis_id": hypothesis_id,
                        "similarity": f"{float(hypothesis_data['similarity']):.17g}",
                    }
                )
                similarities_all.append(float(hypothesis_data["similarity"]))
            best_counts[f"h{target_best + 1}"] += 1
            pair_count += HYPOTHESIS_COUNT

        _write_pair_manifest(output_root, split, manifest_rows)
        split_summaries[split] = {
            "scenes": split_scene_count,
            "pairs": len(manifest_rows),
        }

    if graph_count != sum(scene_counts.values()) * (HYPOTHESIS_COUNT + 1):
        raise RuntimeError("Generated graph file count does not match the expected scene layout.")

    summary = {
        "output": str(output_root),
        "splits": split_summaries,
        "scenes": sum(scene_counts.values()),
        "graph_files": graph_count,
        "nodes_per_graph": {
            "min": int(np.min(node_counts_all)),
            "max": int(np.max(node_counts_all)),
            "mean": float(np.mean(node_counts_all)),
        },
        "similarity": {
            "min": float(np.min(similarities_all)),
            "max": float(np.max(similarities_all)),
            "mean": float(np.mean(similarities_all)),
        },
        "target_best_scenes": best_counts,
        "pair_examples": pair_count,
        "similarity_sigmas": {
            "translation": sigma_translation,
            "rotation": sigma_rotation,
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic SLAM-like graphs using the project graph schema."
    )
    parser.add_argument("--train-scenes", type=int, default=800)
    parser.add_argument("--val-scenes", type=int, default=200)
    parser.add_argument("--test-scenes", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-translation", type=float, default=None)
    parser.add_argument("--sigma-rotation", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_dataset(
        output=args.output,
        train_scenes=args.train_scenes,
        val_scenes=args.val_scenes,
        test_scenes=args.test_scenes,
        seed=args.seed,
        sigma_translation=args.sigma_translation,
        sigma_rotation=args.sigma_rotation,
    )
    print("Synthetic SLAM-like dataset generated:")
    print(f"Output: {summary['output']}")
    print(f"Scenes: {summary['scenes']}")
    print(f"Graph files: {summary['graph_files']}")
    print(
        "Nodes per graph: min={min}, max={max}, mean={mean:.2f}".format(
            **summary["nodes_per_graph"]
        )
    )
    print(
        "Similarity: min={min:.6f}, max={max:.6f}, mean={mean:.6f}".format(
            **summary["similarity"]
        )
    )
    print("Target-best scene counts: " + ", ".join(
        f"{hypothesis}={count}" for hypothesis, count in summary["target_best_scenes"].items()
    ))
    print(f"Pair examples: {summary['pair_examples']}")
    for split, counts in summary["splits"].items():
        print(f"{split}: scenes={counts['scenes']}, pairs={counts['pairs']}")


if __name__ == "__main__":
    main()
