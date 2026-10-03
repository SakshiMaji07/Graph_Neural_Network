"""Preprocessing utilities for continuous graph similarity supervision.

This module defines a supervision target for the SGMH-SLAM hypothesis-selection
problem. It does not learn similarity itself; instead, it converts a geometric
pose error between a hypothesis and a ground-truth pose into a continuous scalar
similarity score in the range ``[0, 1]``.

The output is used as a regression target for a graph neural network trained to
predict similarity between a hypothesis graph and a ground-truth graph.

The supervision function is intentionally independent from any graph encoder or
model logic.
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import Iterable, Sequence

import numpy as np


def _wrap_angle(angle: float) -> float:
    """Wrap an angle into the range ``[-pi, pi]``.

    This handles angular periodicity and ensures that rotation errors are computed
    consistently even for values near the wrapping boundary.

    Args:
        angle: Input angle in radians.

    Returns:
        The wrapped angle in ``[-pi, pi]``.
    """

    wrapped = (angle + pi) % (2.0 * pi) - pi
    return wrapped


def translation_error(
    hypothesis_position: Sequence[float] | np.ndarray,
    ground_truth_position: Sequence[float] | np.ndarray,
) -> float:
    """Compute Euclidean translation error between hypothesis and ground truth.

    Args:
        hypothesis_position: Position vector of the hypothesis, e.g. ``[x, y, z]``.
        ground_truth_position: Position vector of the ground truth, e.g. ``[x, y, z]``.

    Returns:
        The Euclidean norm of the difference between the two position vectors.
    """

    h = np.asarray(hypothesis_position, dtype=np.float64)
    g = np.asarray(ground_truth_position, dtype=np.float64)

    if h.shape != g.shape:
        raise ValueError(
            "Hypothesis and ground-truth position vectors must have identical shapes; "
            f"got {h.shape} and {g.shape}."
        )
    if h.size == 0:
        raise ValueError("Position vectors must be non-empty.")

    diff = h - g
    return float(np.linalg.norm(diff))


def rotation_error(
    hypothesis_rotation: Sequence[float] | np.ndarray,
    ground_truth_rotation: Sequence[float] | np.ndarray,
) -> float:
    """Compute the smallest angular difference between two rotations.

    This function supports a scalar angle or an angle representation that can be
    converted to a single angular difference. It correctly handles angle wrapping
    by comparing the signed difference modulo ``2*pi`` and taking the minimum of the
    two possible directions.

    Args:
        hypothesis_rotation: Rotation value or vector for the hypothesis.
        ground_truth_rotation: Rotation value or vector for the ground truth.

    Returns:
        The smallest angular difference in radians, bounded by ``[0, pi]``.
    """

    h = np.asarray(hypothesis_rotation, dtype=np.float64)
    g = np.asarray(ground_truth_rotation, dtype=np.float64)

    if h.shape != g.shape:
        raise ValueError(
            "Hypothesis and ground-truth rotation values must have identical shapes; "
            f"got {h.shape} and {g.shape}."
        )

    if h.size == 0:
        raise ValueError("Rotation arrays must be non-empty.")

    if h.size != 1:
        if h.size != 1:
            raise ValueError(
                "rotation_error() expects scalar rotations for the baseline implementation; "
                f"got shape {h.shape}."
            )

    angle_diff = float(_wrap_angle(float(h.reshape(-1)[0]) - float(g.reshape(-1)[0])))
    return abs(angle_diff)


def pose_similarity(
    d_translation: float,
    d_rotation: float,
    sigma_translation: float = 1.0,
    sigma_rotation: float = 1.0,
) -> float:
    """Convert geometric pose error into a continuous similarity target.

    The supervision score is defined as:

    ``exp(-(d_translation^2)/(2*sigma_translation^2) - (d_rotation^2)/(2*sigma_rotation^2))``

    with the final value clamped to the interval ``[0, 1]``.

    This function is a supervision definition and must not be confused with a
    learned similarity component or a graph-embedding model.

    Args:
        d_translation: Euclidean translation error.
        d_rotation: Smallest angular error in radians.
        sigma_translation: Translation scale parameter controlling how quickly the
            score decays with translation error.
        sigma_rotation: Rotation scale parameter controlling how quickly the score
            decays with angular error.

    Returns:
        A scalar similarity score in ``[0, 1]``.
    """

    if sigma_translation <= 0:
        raise ValueError(f"sigma_translation must be positive, got {sigma_translation}.")
    if sigma_rotation <= 0:
        raise ValueError(f"sigma_rotation must be positive, got {sigma_rotation}.")

    score = float(
        np.exp(
            -((d_translation ** 2) / (2.0 * sigma_translation ** 2))
            - ((d_rotation ** 2) / (2.0 * sigma_rotation ** 2))
        )
    )
    return float(np.clip(score, 0.0, 1.0))


def similarity_targets_for_hypotheses(
    ground_truth_position: Sequence[float] | np.ndarray,
    ground_truth_rotation: float | Sequence[float] | np.ndarray,
    hypothesis_positions: Iterable[Sequence[float] | np.ndarray],
    hypothesis_rotations: Iterable[float | Sequence[float] | np.ndarray],
    sigma_translation: float = 1.0,
    sigma_rotation: float = 1.0,
) -> list[float]:
    """Compute similarity targets for five hypotheses relative to one ground truth.

    This helper converts a set of hypothesis-vs-ground-truth pose errors into a list
    of continuous supervision values.

    Args:
        ground_truth_position: Ground-truth position vector.
        ground_truth_rotation: Ground-truth rotation value or vector.
        hypothesis_positions: Iterable of hypothesis position vectors.
        hypothesis_rotations: Iterable of hypothesis rotation values.
        sigma_translation: Translation scale parameter.
        sigma_rotation: Rotation scale parameter.

    Returns:
        A list of similarity scores, one for each hypothesis.
    """

    pos_list = list(hypothesis_positions)
    rot_list = list(hypothesis_rotations)

    if len(pos_list) != len(rot_list):
        raise ValueError(
            "hypothesis_positions and hypothesis_rotations must have the same length; "
            f"got {len(pos_list)} and {len(rot_list)}."
        )

    scores: list[float] = []
    for hypothesis_position, hypothesis_rotation in zip(pos_list, rot_list):
        d_translation = translation_error(hypothesis_position, ground_truth_position)
        d_rotation = rotation_error(hypothesis_rotation, ground_truth_rotation)
        scores.append(
            pose_similarity(
                d_translation=d_translation,
                d_rotation=d_rotation,
                sigma_translation=sigma_translation,
                sigma_rotation=sigma_rotation,
            )
        )
    return scores


__all__ = [
    "_wrap_angle",
    "translation_error",
    "rotation_error",
    "pose_similarity",
    "similarity_targets_for_hypotheses",
]
