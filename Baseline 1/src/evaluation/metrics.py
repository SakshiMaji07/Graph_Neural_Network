"""Evaluation metrics for continuous graph similarity and five-way hypothesis selection.

This module provides regression and ranking metrics for Siamese graph-matching models.
The functions intentionally return plain Python floats so they are easy to log, serialize,
report, and compare across training and evaluation runs.

The continuous metrics operate on 1D arrays of paired predictions and targets.
The selection metrics operate on 2D arrays of shape ``[B, 5]`` where each row contains
five hypothesis scores for a single sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_numpy_1d(values: Any, *, name: str) -> np.ndarray:
    """Convert an input tensor/array/list into a 1D NumPy float64 array."""
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 0:
        array = array.reshape(1)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor/array, got shape {array.shape}.")

    return np.asarray(array, dtype=np.float64)


def _as_numpy_2d(values: Any, *, name: str) -> np.ndarray:
    """Convert an input tensor/array/list into a 2D NumPy float64 array."""
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor/array, got shape {array.shape}.")

    return np.asarray(array, dtype=np.float64)


def _validate_same_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Ensure regression targets and predictions have matching lengths."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length; "
            f"got {y_true.shape[0]} and {y_pred.shape[0]}."
        )

    if y_true.size == 0:
        raise ValueError("Regression metrics cannot be computed on an empty array.")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks for a 1D array, preserving tie handling.

    Ties are assigned the average of the tied positions, which is the standard approach
    for Spearman rank correlation and ranking metrics under equal scores.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        raise ValueError("Cannot rank an empty array.")

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    start = 0

    while start < n:
        end = start + 1
        while end < n and values[order[end]] == values[order[start]]:
            end += 1

        avg_rank = 0.5 * (start + 1 + end)
        for idx in range(start, end):
            ranks[order[idx]] = avg_rank
        start = end

    return ranks


def mae(y_true: Any, y_pred: Any) -> float:
    """Compute the mean absolute error (MAE).

    MAE is the average absolute difference between the predicted and target values:

        MAE = (1 / N) * sum_i |y_i - yhat_i|

    Returns a plain Python float.
    """
    true = _as_numpy_1d(y_true, name="y_true")
    pred = _as_numpy_1d(y_pred, name="y_pred")
    _validate_same_length(true, pred)
    return float(np.mean(np.abs(true - pred)))


def mse(y_true: Any, y_pred: Any) -> float:
    """Compute the mean squared error (MSE).

    MSE is the average squared difference between the predicted and target values:

        MSE = (1 / N) * sum_i (y_i - yhat_i)^2

    Returns a plain Python float.
    """
    true = _as_numpy_1d(y_true, name="y_true")
    pred = _as_numpy_1d(y_pred, name="y_pred")
    _validate_same_length(true, pred)
    return float(np.mean((true - pred) ** 2))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Compute the root mean squared error (RMSE).

    RMSE is the square root of the MSE and is useful because it is on the same scale as
    the target variable:

        RMSE = sqrt((1 / N) * sum_i (y_i - yhat_i)^2)

    Returns a plain Python float.
    """
    true = _as_numpy_1d(y_true, name="y_true")
    pred = _as_numpy_1d(y_pred, name="y_pred")
    _validate_same_length(true, pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def pearson_correlation(y_true: Any, y_pred: Any) -> float:
    """Compute Pearson correlation if it is statistically valid.

    Pearson correlation is the linear correlation coefficient between two variables:

        corr = sum((x - x̄)(y - ȳ)) / sqrt(sum((x - x̄)^2) * sum((y - ȳ)^2))

    This metric is undefined when either sequence is constant or when the variance is
    zero. In those cases, the function returns ``float('nan')`` instead of producing an
    invalid numeric result.

    Returns a plain Python float or ``nan`` when the correlation is undefined.
    """
    true = _as_numpy_1d(y_true, name="y_true")
    pred = _as_numpy_1d(y_pred, name="y_pred")
    _validate_same_length(true, pred)

    if true.size < 2:
        return float("nan")

    x_centered = true - np.mean(true)
    y_centered = pred - np.mean(pred)

    if np.allclose(x_centered, 0.0) or np.allclose(y_centered, 0.0):
        return float("nan")

    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if np.isclose(denom, 0.0):
        return float("nan")

    corr = float(np.dot(x_centered, y_centered) / denom)
    return corr


def spearman_rank_correlation(y_true: Any, y_pred: Any) -> float:
    """Compute Spearman rank correlation with tie-aware ranking.

    Spearman correlation is the Pearson correlation of the rank-transformed values. It
    measures monotonic agreement rather than linear agreement and is more robust to
    nonlinearity or small monotonic changes.

    Ties are handled using the average-rank convention, so equal scores receive the same
    average rank before applying the Pearson correlation. If the rank vectors are
    constant (e.g., all scores are identical), the result is undefined and this function
    returns ``float('nan')``.

    Returns a plain Python float or ``nan`` when undefined.
    """
    true = _as_numpy_1d(y_true, name="y_true")
    pred = _as_numpy_1d(y_pred, name="y_pred")
    _validate_same_length(true, pred)

    ranks_true = _average_ranks(true)
    ranks_pred = _average_ranks(pred)
    return pearson_correlation(ranks_true, ranks_pred)


def top1_selection_accuracy(target_scores: Any, predicted_scores: Any) -> float:
    """Compute top-1 selection accuracy for five-hypothesis scoring.

    For each sample, the selected hypothesis is the index of the maximum score:

        argmax(predicted_scores[i])

    The target-best hypothesis is similarly determined by

        argmax(target_scores[i])

    The metric is then the mean of the indicator that the model selected the same
    hypothesis as the target-best one:

        mean(argmax(predicted_scores) == argmax(target_scores))

    When there are ties, ``np.argmax`` chooses the first occurrence in the row, which is
    a deterministic and reasonable tie-breaking rule. This matches the usual semantics of
    ``argmax`` in NumPy and PyTorch.

    Returns a plain Python float in the range ``[0.0, 1.0]``.
    """
    target = _as_numpy_2d(target_scores, name="target_scores")
    pred = _as_numpy_2d(predicted_scores, name="predicted_scores")

    if target.shape != pred.shape:
        raise ValueError(
            "target_scores and predicted_scores must have identical shapes; "
            f"got {target.shape} and {pred.shape}."
        )

    if target.size == 0:
        raise ValueError("Selection metrics cannot be computed on an empty batch.")

    target_best = np.argmax(target, axis=1)
    pred_best = np.argmax(pred, axis=1)
    return float(np.mean(pred_best == target_best))


def _average_ranks_descending(row: np.ndarray) -> np.ndarray:
    """Return average rank positions for a single row in descending score order.

    A larger score is ranked as a more preferred hypothesis. In the presence of ties,
    equal scores receive the average rank of their tied positions. For example, scores
    ``[0.9, 0.9, 0.3]`` receive ranks ``[1.5, 1.5, 3.0]`` when ranked in descending order.
    """
    n = row.size
    order = np.argsort(-row, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    start = 0

    while start < n:
        end = start + 1
        while end < n and row[order[end]] == row[order[start]]:
            end += 1

        avg_rank = 0.5 * (start + 1 + end)
        for idx in range(start, end):
            ranks[order[idx]] = avg_rank
        start = end

    return ranks


def mean_rank_of_best_hypothesis(target_scores: Any, predicted_scores: Any) -> float:
    """Compute the average rank of the ground-truth-best hypothesis.

    For each sample, the ground-truth-best hypothesis is the index of the maximum value in
    the target score row:

        argmax(target_scores[i])

    The predicted ranking is formed by sorting the predicted scores in descending order,
    with average ranks assigned to tied scores. The rank of the target-best hypothesis is
    then measured among the predicted hypotheses, and the final metric is the mean across
    all samples.

    This metric is useful for evaluating whether the most relevant hypothesis is also the
    best-ranked hypothesis under the model's predictions.

    Returns a plain Python float representing the mean rank position, where smaller is
    better. The rank is 1-indexed (first hypothesis is rank 1).
    """
    target = _as_numpy_2d(target_scores, name="target_scores")
    pred = _as_numpy_2d(predicted_scores, name="predicted_scores")

    if target.shape != pred.shape:
        raise ValueError(
            "target_scores and predicted_scores must have identical shapes; "
            f"got {target.shape} and {pred.shape}."
        )

    if target.size == 0:
        raise ValueError("Selection metrics cannot be computed on an empty batch.")

    if target.shape[1] == 0:
        raise ValueError("Each sample must contain at least one hypothesis score.")

    ranks = []
    for i in range(target.shape[0]):
        gt_best_index = int(np.argmax(target[i]))
        row_ranks = _average_ranks_descending(pred[i])
        rank_of_best = float(row_ranks[gt_best_index])
        ranks.append(rank_of_best)

    return float(np.mean(ranks))


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return a dictionary of standard regression metrics.

    This helper is a convenience wrapper for evaluation pipelines that want a single
    dictionary containing MAE, MSE, RMSE, and Pearson correlation.
    """
    return {
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson": pearson_correlation(y_true, y_pred),
    }


def selection_metrics(target_scores: Any, predicted_scores: Any) -> dict[str, float]:
    """Return a dictionary of five-hypothesis selection metrics.

    The dictionary includes the top-1 selection accuracy and the mean rank of the
    ground-truth-best hypothesis. These metrics are computed row-wise for batches of
    candidate hypothesis scores of shape ``[B, 5]``.
    """
    return {
        "top1_accuracy": top1_selection_accuracy(target_scores, predicted_scores),
        "mean_rank_of_best_hypothesis": mean_rank_of_best_hypothesis(target_scores, predicted_scores),
    }


__all__ = [
    "mae",
    "mse",
    "rmse",
    "pearson_correlation",
    "spearman_rank_correlation",
    "top1_selection_accuracy",
    "mean_rank_of_best_hypothesis",
    "regression_metrics",
    "selection_metrics",
]
