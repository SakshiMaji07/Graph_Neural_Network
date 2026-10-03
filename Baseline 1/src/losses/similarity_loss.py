"""Continuous graph similarity losses.

This module implements regression losses for scalar graph similarity scores. In the
training setup, both the predicted similarity and the supervision target live in the
closed interval [0, 1]. The losses are therefore defined over a batch of scalar
values and return a single scalar summary value computed by mean reduction.

For a batch of size B with predictions p_i and targets t_i:

    MSE(p, t) = (1 / B) * sum_i (p_i - t_i)^2

    SmoothL1(p, t) = (1 / B) * sum_i L_smooth(p_i - t_i)

where

    L_smooth(x) = 0.5 * x^2 / beta              if |x| < beta
                 |x| - 0.5 * beta               otherwise

The implementation accepts both [B] and [B, 1] inputs and normalizes them to a
common 1D representation before computing the loss. This keeps the behavior
consistent for single-score batches and column-vector batches.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


LossType = Literal["mse", "smooth_l1", "smoothl1"]


def _reshape_batch_like(tensor: Tensor, name: str) -> Tensor:
    """Return a flat batch view of shape [B].

    Accepted inputs are:
    - [B]
    - [B, 1]

    A 2D tensor with a non-singleton second dimension is rejected to prevent silent
    broadcasting mistakes.
    """
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have shape [B] or [B, 1], but received scalar shape {tuple(tensor.shape)}.")
    if tensor.ndim > 2:
        raise ValueError(f"{name} must have shape [B] or [B, 1], but received {tuple(tensor.shape)}.")
    if tensor.ndim == 2 and tensor.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [B] or [B, 1], but received {tuple(tensor.shape)}. "
            "If you have a column vector, use shape [B, 1]."
        )
    return tensor.reshape(-1)


def mse_similarity_loss(similarity_pred: Tensor, similarity_target: Tensor) -> Tensor:
    """Compute the mean squared error for continuous similarity regression.

    Args:
        similarity_pred: Predicted similarity values in [0, 1], shape [B] or [B, 1].
        similarity_target: Ground-truth similarity values in [0, 1], shape [B] or [B, 1].

    Returns:
        Scalar tensor equal to (1 / B) * sum_i (pred_i - target_i)^2.
    """
    return SimilarityLoss(loss_type="mse")(similarity_pred, similarity_target)


def smooth_l1_similarity_loss(
    similarity_pred: Tensor,
    similarity_target: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Compute the SmoothL1 loss for similarity regression.

    Args:
        similarity_pred: Predicted similarity values in [0, 1], shape [B] or [B, 1].
        similarity_target: Ground-truth similarity values in [0, 1], shape [B] or [B, 1].
        beta: SmoothL1 parameter controlling the transition point between quadratic and
            linear regimes. Larger beta makes the loss more linear; smaller beta makes it
            more quadratic near zero.

    Returns:
        Scalar tensor equal to the mean SmoothL1 residual over the batch.
    """
    return SimilarityLoss(loss_type="smooth_l1", beta=beta)(similarity_pred, similarity_target)


class SimilarityLoss(nn.Module):
    """Configurable regression loss for graph similarity scores.

    The model output is expected to be a scalar similarity in [0, 1]. The target is
    also in [0, 1]. This loss is intended for continuous similarity-label regression
    and does not implement ranking or pairwise margin objectives.

    Supported loss types:
        - "mse": mean squared error
        - "smooth_l1" / "smoothl1": SmoothL1 regression loss

    Input handling:
        - Accepts [B] and [B, 1] tensors.
        - Reinterprets both as a 1D batch of length B.
        - Raises a descriptive error if the batch dimensions are incompatible.
        - Returns a single scalar loss tensor with no remaining batch dimension.
    """

    def __init__(self, loss_type: LossType = "mse", beta: float = 1.0) -> None:
        super().__init__()

        normalized_type = str(loss_type).lower()
        if normalized_type in {"mse", "mean_squared_error"}:
            self.loss_type: str = "mse"
        elif normalized_type in {"smooth_l1", "smoothl1", "smooth-l1"}:
            self.loss_type = "smooth_l1"
        else:
            raise ValueError(
                f"Unsupported similarity loss type '{loss_type}'. "
                "Expected one of: 'mse', 'smooth_l1', or 'smoothl1'."
            )

        beta = float(beta)
        if beta <= 0.0:
            raise ValueError(f"beta must be positive for SmoothL1 loss, but received beta={beta}.")
        self.beta = beta

    def _validate_batch(self, similarity_pred: Tensor, similarity_target: Tensor) -> tuple[Tensor, Tensor]:
        """Validate shapes and value ranges for predictions and targets."""
        if not isinstance(similarity_pred, Tensor):
            raise TypeError(f"similarity_pred must be a torch.Tensor, but received {type(similarity_pred).__name__}.")
        if not isinstance(similarity_target, Tensor):
            raise TypeError(f"similarity_target must be a torch.Tensor, but received {type(similarity_target).__name__}.")

        if similarity_pred.device != similarity_target.device:
            raise ValueError(
                "similarity_pred and similarity_target must be on the same device. "
                f"Got pred.device={similarity_pred.device}, target.device={similarity_target.device}."
            )

        pred_1d = _reshape_batch_like(similarity_pred, "similarity_pred")
        target_1d = _reshape_batch_like(similarity_target, "similarity_target")

        if pred_1d.shape != target_1d.shape:
            raise ValueError(
                "Shape mismatch between similarity_pred and similarity_target. "
                f"Received pred.shape={tuple(similarity_pred.shape)} and target.shape={tuple(similarity_target.shape)}. "
                "Expected both to be [B] or [B, 1] with matching batch sizes."
            )

        if not torch.is_floating_point(pred_1d):
            pred_1d = pred_1d.float()
        if not torch.is_floating_point(target_1d):
            target_1d = target_1d.float()

        if not torch.all(torch.isfinite(pred_1d)):
            raise ValueError("similarity_pred contains non-finite values (NaN or Inf).")
        if not torch.all(torch.isfinite(target_1d)):
            raise ValueError("similarity_target contains non-finite values (NaN or Inf).")

        pred_min = pred_1d.min().item()
        pred_max = pred_1d.max().item()
        target_min = target_1d.min().item()
        target_max = target_1d.max().item()

        if not (0.0 <= pred_min and pred_max <= 1.0):
            raise ValueError(
                f"similarity_pred must be in the range [0, 1], but received min={pred_min:.6f}, max={pred_max:.6f}."
            )
        if not (0.0 <= target_min and target_max <= 1.0):
            raise ValueError(
                f"similarity_target must be in the range [0, 1], but received min={target_min:.6f}, max={target_max:.6f}."
            )

        return pred_1d, target_1d

    def _mse_loss(self, similarity_pred: Tensor, similarity_target: Tensor) -> Tensor:
        """Mean squared error: MSE = (1 / B) * sum_i (p_i - t_i)^2."""
        residual = similarity_pred - similarity_target
        return torch.mean(residual.pow(2))

    def _smooth_l1_loss(self, similarity_pred: Tensor, similarity_target: Tensor) -> Tensor:
        """SmoothL1 loss: quadratic near zero and linear elsewhere.

        For a residual x = p - t,

            L(x) = 0.5 * x^2 / beta         if |x| < beta
                   |x| - 0.5 * beta         otherwise

        The final value is the mean across the batch.
        """
        residual = similarity_pred - similarity_target
        abs_residual = torch.abs(residual)
        loss = torch.where(
            abs_residual < self.beta,
            0.5 * residual.pow(2) / self.beta,
            abs_residual - 0.5 * self.beta,
        )
        return torch.mean(loss)

    def forward(self, similarity_pred: Tensor, similarity_target: Tensor) -> Tensor:
        """Compute the scalar similarity regression loss.

        Args:
            similarity_pred: Predicted similarity values. Shape may be [B] or [B, 1].
            similarity_target: Ground truth similarity values. Shape may be [B] or [B, 1].

        Returns:
            A single scalar tensor representing the mean loss over the batch.
        """
        pred_1d, target_1d = self._validate_batch(similarity_pred, similarity_target)

        if self.loss_type == "mse":
            loss = self._mse_loss(pred_1d, target_1d)
        elif self.loss_type == "smooth_l1":
            loss = self._smooth_l1_loss(pred_1d, target_1d)
        else:
            raise RuntimeError(f"Unsupported loss type '{self.loss_type}' in SimilarityLoss.")

        if not torch.isfinite(loss):
            raise ValueError(
                "Computed similarity loss is non-finite. This usually indicates invalid input values or an unstable loss configuration."
            )

        return loss


__all__ = ["SimilarityLoss", "mse_similarity_loss", "smooth_l1_similarity_loss"]
