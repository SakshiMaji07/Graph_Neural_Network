"""Generate and validate the full synthetic benchmark without training it.

The benchmark uses seed 42 for repeatability. Generated scene IDs are assigned to
one split only, and the existing validator checks graph files, labels, manifests,
and scene leakage before success is reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_data import generate_dataset
from scripts.validate_synthetic_data import validate_dataset

FIXED_SEED = 42
SCENE_COUNTS = {"train": 800, "val": 200, "test": 200}
EXPECTED_PAIR_COUNTS = {"train": 4000, "val": 1000, "test": 1000}


def _print_statistics(summary: dict[str, Any]) -> None:
    print("Full synthetic benchmark statistics:")
    for split in ("train", "val", "test"):
        print(
            f"{split}: scenes={summary['scenes_per_split'][split]}, "
            f"pairs={summary['pairs_per_split'][split]}"
        )
    for label, key in (
        ("Node count", "nodes"),
        ("Edge count", "edges"),
        ("Similarity", "similarities"),
    ):
        values = summary[key]
        if values is None:
            raise RuntimeError(f"Validation returned no statistics for {label.lower()}.")
        print(
            f"{label}: min={values['min']}, mean={values['mean']:.3f}, "
            f"max={values['max']}"
        )
    print(
        "Target-best distribution: "
        + ", ".join(
            f"{name}={count}"
            for name, count in summary["target_best"].items()
        )
    )


def generate_and_validate(output: Path) -> dict[str, Any]:
    """Generate the fixed-size benchmark once and validate the resulting files."""
    output = output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()

    print(f"Generating benchmark with fixed seed {FIXED_SEED} at {output}")
    generated = generate_dataset(
        output=output,
        train_scenes=SCENE_COUNTS["train"],
        val_scenes=SCENE_COUNTS["val"],
        test_scenes=SCENE_COUNTS["test"],
        seed=FIXED_SEED,
    )
    print(
        "Generated: "
        f"{generated['scenes']} scenes, {generated['graph_files']} graph files, "
        f"{generated['pair_examples']} pairs"
    )

    summary = validate_dataset(output)
    _print_statistics(summary)

    if summary["errors"]:
        errors = "\n".join(f"- {error}" for error in summary["errors"])
        raise RuntimeError(f"Dataset validation found issues:\n{errors}")
    if summary["scenes_per_split"] != SCENE_COUNTS:
        raise RuntimeError(
            f"Unexpected scene counts: {summary['scenes_per_split']}; "
            f"expected {SCENE_COUNTS}."
        )
    if summary["pairs_per_split"] != EXPECTED_PAIR_COUNTS:
        raise RuntimeError(
            f"Unexpected pair counts: {summary['pairs_per_split']}; "
            f"expected {EXPECTED_PAIR_COUNTS}."
        )
    if sum(summary["target_best"].values()) != sum(SCENE_COUNTS.values()):
        raise RuntimeError("Target-best distribution does not cover all 1,200 scenes.")
    if sum(count > 0 for count in summary["target_best"].values()) < 2:
        raise RuntimeError(
            "Target-best is not varied across hypotheses; expected at least two "
            "hypotheses to be best for one or more scenes."
        )

    print(f"Output: {output}")
    print("FULL DATASET VALIDATION PASSED")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and validate the full 800/200/200 synthetic benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/full_synthetic_benchmark"),
        help="Output dataset root (default: data/full_synthetic_benchmark).",
    )
    args = parser.parse_args()
    generate_and_validate(args.output)


if __name__ == "__main__":
    main()
