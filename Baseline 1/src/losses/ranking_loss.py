"""Pairwise margin ranking loss for hypothesis selection in SGMH-SLAM.

Problem setup
-------------
For each scenario, there are five candidate hypotheses:

    H1, H2, H3, H4, H5

with corresponding continuous ground-truth similarity values:

    y1, y2, y3, y4, y5

The desired behavior is that if a hypothesis i is more correct than hypothesis j,
then its predicted score should be higher. Formally, for all pairs (i, j):

    if yi > yj, then score_i > score_j

To enforce this ordering, we use the pairwise margin ranking objective:

    L(i, j) = max(0, margin - (score_i - score_j)),  for yi > yj

This is a hinge loss: if the score difference exceeds the margin, the pair is
satisfied and contributes zero; otherwise it is penalized. The total loss is the
mean over all valid ordered pairs in the batch.

The implementation below:
- accepts predicted_scores with shape [B, 5]
- accepts target_scores with shape [B, 5]
--generates all valid pairs only for hypotheses with yi > yj
- avoids useless identical comparisons
- ignores ties by requiring strict inequality yi > yj
- returns a single scalar loss tensor
- can be combined with a continuous regression loss later because it only scores
  the relative ordering of hypotheses and does not collapse the five scores into a
  single scalar.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PairwiseRankingLoss(nn.Module):
    """Margin ranking loss for ordered hypothesis pairs.

    The model predicts five scores per scenario, one for each hypothesis. This class
    enforces that for every pair of hypotheses where the ground-truth similarity of
    hypothesis i is greater than that of hypothesis j, the model must satisfy:

        predicted_score_i >= predicted_score_j + margin.

    The loss is computed only for valid pairs with strict target ordering, so ties are
    ignored. This is important because a tie should not create a ranking preference.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()

        if not isinstance(margin, (int, float)):
            raise TypeError(f"margin must be a real number, but received {type(margin).__name__}.")
        margin = float(margin)
        if margin < 0.0:
            raise ValueError(f"margin must be non-negative, but received {margin}.")
        self.margin = margin

        valid_reductions = {"mean", "sum", "none"}
        if reduction not in valid_reductions:
            raise ValueError(
                f"Unsupported reduction '{reduction}'. Expected one of: {sorted(valid_reductions)}."
            )
        self.reduction = reduction

    def _validate_inputs(self, predicted_scores: Tensor, target_scores: Tensor) -> tuple[Tensor, Tensor]:
        """Validate tensor type, shape, and value constraints."""
        if not isinstance(predicted_scores, Tensor):
            raise TypeError(
                "predicted_scores must be a torch.Tensor, "
                f"but received {type(predicted_scores).__name__}."
            )
        if not isinstance(target_scores, Tensor):
            raise TypeError(
                "target_scores must be a torch.Tensor, "
                f"but received {type(target_scores).__name__}."
            )

        if predicted_scores.shape != target_scores.shape:
            raise ValueError(
                "Shape mismatch between predicted_scores and target_scores. "
                f"Received predicted_scores.shape={tuple(predicted_scores.shape)} and "
                f"target_scores.shape={tuple(target_scores.shape)}. "
                "Expected both to have shape [B, 5]."
            )

        if predicted_scores.ndim != 2 or predicted_scores.shape[1] != 5:
            raise ValueError(
                "predicted_scores must have shape [B, 5], but received "
                f"{tuple(predicted_scores.shape)}."
            )
        if target_scores.ndim != 2 or target_scores.shape[1] != 5:
            raise ValueError(
                "target_scores must have shape [B, 5], but received "
                f"{tuple(target_scores.shape)}."
            )

        if predicted_scores.device != target_scores.device:
            raise ValueError(
                "predicted_scores and target_scores must be on the same device. "
                f"Got predicted_scores.device={predicted_scores.device} and "
                f"target_scores.device={target_scores.device}."
            )

        if not torch.is_floating_point(predicted_scores):
            predicted_scores = predicted_scores.float()
        if not torch.is_floating_point(target_scores):
            target_scores = target_scores.float()

        if not torch.all(torch.isfinite(predicted_scores)):
            raise ValueError("predicted_scores contains NaN or Inf values.")
        if not torch.all(torch.isfinite(target_scores)):
            raise ValueError("target_scores contains NaN or Inf values.")

        target_min = target_scores.min().item()
        target_max = target_scores.max().item()
        if not (0.0 <= target_min and target_max <= 1.0):
            raise ValueError(
                "target_scores should represent continuous similarity values in [0, 1], "
                f"but received min={target_min}, max={target_max}."
            )

        return predicted_scores, target_scores

    def _generate_valid_pairs(self, target_scores: Tensor) -> tuple[Tensor, Tensor]:
        """Create ordered pairs (i, j) where target_scores[:, i] > target_scores[:, j].

        We generate all unique pairs with i < j and then filter by strict target ordering.
        This avoids unnecessary identical-pair comparisons and correctly ignores ties.
        """
        batch_size = target_scores.shape[0]
        idx_i, idx_j = torch.triu_indices(5, 5, offset=1)
        # idx_i, idx_j are length 10, representing unique pairs (0,1), (0,2), ..., (4,5)
        # but since we only have 5 hypotheses, the pair set is all unique unordered pairs.
        pair_i = idx_i.to(target_scores.device)
        pair_j = idx_j.to(target_scores.device)

        # For each sample, we need a strict ordering: yi > yj.
        valid_mask = target_scores[:, pair_i] > target_scores[:, pair_j]
        valid_pairs_i = pair_i[None, :].expand(batch_size, -1)
        valid_pairs_j = pair_j[None, :].expand(batch_size, -1)

        valid_mask = valid_mask & (target_scores[:, pair_i] > target_scores[:, pair_j])

        pair_i = valid_pairs_i[valid_mask]
        pair_j = valid_pairs_j[valid_mask]
        return pair_i, pair_j

    def forward(self, predicted_scores: Tensor, target_scores: Tensor) -> Tensor:
        """Compute the scalar pairwise ranking loss.

        Args:
            predicted_scores: Tensor of shape [B, 5], representing the model's score for
                each hypothesis.
            target_scores: Tensor of shape [B, 5], representing the ground-truth
                continuous similarity values for each hypothesis.

        Returns:
            A scalar tensor summarizing the ranking loss for the batch.
        """
        predicted_scores, target_scores = self._validate_inputs(predicted_scores, target_scores)

        pair_i, pair_j = self._generate_valid_pairs(target_scores)
        if pair_i.numel() == 0:
            return predicted_scores.new_zeros(())

        score_i = predicted_scores[:, pair_i]
        score_j = predicted_scores[:, pair_j]

        # For a valid pair, the target ordering is yi > yj, so we want score_i > score_j + margin.
        # F.margin_ranking_loss with target=1.0 enforces x1 > x2 + margin.
        loss = F.margin_ranking_loss(
            score_i,
            score_j,
            torch.ones_like(score_i),
            margin=self.margin,
            reduction="none",
        )

        if self.reduction == "none":
            return loss

        if self.reduction == "sum":
            return loss.sum()

        return loss.mean()


def pairwise_ranking_loss(
    predicted_scores: Tensor,
    target_scores: Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Convenience wrapper around PairwiseRankingLoss."""
    return PairwiseRankingLoss(margin=margin, reduction=reduction)(predicted_scores, target_scores)


__all__ = ["PairwiseRankingLoss", "pairwise_ranking_loss"]
