"""Combined objective for graph similarity regression and hypothesis ranking.

This module defines a configurable combined loss for the GNN SLAM hypothesis-selection
system. The total objective is:

    L_total = lambda_similarity * L_similarity + lambda_ranking * L_ranking

The two components serve different purposes:

1. Continuous similarity regression
   - Use when the model should predict a calibrated score in [0, 1] for each hypothesis.
   - This is useful for learning the absolute similarity value of a candidate match.
   - It is typically supervised by a regression target such as a continuous similarity
     annotation, and is implemented via SimilarityLoss.

2. Pairwise ranking loss
   - Use when the model should learn relative ordering between candidate hypotheses.
   - This is useful when the objective is to rank the best hypothesis above weaker ones,
     even if the individual absolute scores are not perfectly calibrated.
   - It is especially relevant for the five-hypothesis selection problem where the goal is
     to prefer the correct hypothesis over alternatives.

The module supports three training modes:

- similarity-only training: compute only L_similarity
- ranking-only training: compute only L_ranking
- combined training: compute both and combine them through configurable lambdas

The output contains both the total loss and a dictionary with individual terms so that
training logs or diagnostics can inspect each component separately.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .ranking_loss import PairwiseRankingLoss
from .similarity_loss import SimilarityLoss


class CombinedLoss(nn.Module):
    """Weighted combination of continuous similarity regression and ranking losses.

    The total loss is

        L_total = lambda_similarity * L_similarity + lambda_ranking * L_ranking

    The selected components are configurable, which allows similarity-only, ranking-only,
    or combined training.
    """

    def __init__(
        self,
        similarity_loss_type: str = "mse",
        similarity_beta: float = 1.0,
        lambda_similarity: float = 1.0,
        lambda_ranking: float = 1.0,
        ranking_margin: float = 1.0,
        include_similarity: bool = True,
        include_ranking: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(lambda_similarity, (int, float)):
            raise TypeError(
                f"lambda_similarity must be a real number, but received {type(lambda_similarity).__name__}."
            )
        if not isinstance(lambda_ranking, (int, float)):
            raise TypeError(
                f"lambda_ranking must be a real number, but received {type(lambda_ranking).__name__}."
            )

        self.lambda_similarity = float(lambda_similarity)
        self.lambda_ranking = float(lambda_ranking)

        if not isinstance(include_similarity, bool):
            raise TypeError(f"include_similarity must be a bool, but received {type(include_similarity).__name__}.")
        if not isinstance(include_ranking, bool):
            raise TypeError(f"include_ranking must be a bool, but received {type(include_ranking).__name__}.")

        self.include_similarity = include_similarity
        self.include_ranking = include_ranking

        self.similarity_loss_fn = SimilarityLoss(loss_type=similarity_loss_type, beta=similarity_beta)
        self.ranking_loss_fn = PairwiseRankingLoss(margin=ranking_margin, reduction="mean")

    def _validate_similarity_inputs(
        self,
        similarity_pred: Tensor,
        similarity_target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Validate scalar similarity regression inputs."""
        if not isinstance(similarity_pred, Tensor):
            raise TypeError(
                "similarity_pred must be a torch.Tensor, "
                f"but received {type(similarity_pred).__name__}."
            )
        if not isinstance(similarity_target, Tensor):
            raise TypeError(
                "similarity_target must be a torch.Tensor, "
                f"but received {type(similarity_target).__name__}."
            )

        return similarity_pred, similarity_target

    def _validate_ranking_inputs(
        self,
        predicted_scores: Tensor,
        target_scores: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Validate hypothesis-ranking inputs."""
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

        return predicted_scores, target_scores

    def forward(
        self,
        similarity_pred: Optional[Tensor] = None,
        similarity_target: Optional[Tensor] = None,
        predicted_scores: Optional[Tensor] = None,
        target_scores: Optional[Tensor] = None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Compute the combined objective and the individual components.

        Args:
            similarity_pred: Predicted scalar similarity values, shape [B] or [B, 1].
                Required when include_similarity is True.
            similarity_target: Ground-truth similarity values, shape [B] or [B, 1].
                Required when include_similarity is True.
            predicted_scores: Hypothesis scores with shape [B, 5]. Required when
                include_ranking is True.
            target_scores: Hypothesis target similarity values with shape [B, 5].
                Required when include_ranking is True.

        Returns:
            total_loss: Scalar tensor equal to the weighted sum of active components.
            losses: Dictionary with the individual component values. Keys are:
                - "similarity"
                - "ranking"
                - "total"
        """
        losses: Dict[str, Tensor] = {}

        if self.include_similarity:
            if similarity_pred is None or similarity_target is None:
                raise ValueError(
                    "Similarity loss is enabled, but similarity_pred and similarity_target were not provided."
                )
            similarity_pred, similarity_target = self._validate_similarity_inputs(
                similarity_pred, similarity_target
            )
            similarity_loss = self.similarity_loss_fn(similarity_pred, similarity_target)
            losses["similarity"] = similarity_loss
        else:
            losses["similarity"] = torch.zeros((), device=self.similarity_loss_fn.beta.__class__ if False else None)

        if self.include_ranking:
            if predicted_scores is None or target_scores is None:
                raise ValueError(
                    "Ranking loss is enabled, but predicted_scores and target_scores were not provided."
                )
            predicted_scores, target_scores = self._validate_ranking_inputs(predicted_scores, target_scores)
            ranking_loss = self.ranking_loss_fn(predicted_scores, target_scores)
            losses["ranking"] = ranking_loss
        else:
            losses["ranking"] = torch.tensor(0.0, dtype=torch.float32)

        if not self.include_similarity and not self.include_ranking:
            raise ValueError(
                "CombinedLoss requires at least one active component. "
                "Set include_similarity=True or include_ranking=True."
            )

        total_loss = torch.zeros((), dtype=torch.float32)
        if self.include_similarity:
            total_loss = total_loss + self.lambda_similarity * losses["similarity"]
        if self.include_ranking:
            total_loss = total_loss + self.lambda_ranking * losses["ranking"]

        losses["total"] = total_loss
        return total_loss, losses


def combined_loss(
    similarity_pred: Optional[Tensor] = None,
    similarity_target: Optional[Tensor] = None,
    predicted_scores: Optional[Tensor] = None,
    target_scores: Optional[Tensor] = None,
    similarity_loss_type: str = "mse",
    similarity_beta: float = 1.0,
    lambda_similarity: float = 1.0,
    lambda_ranking: float = 1.0,
    ranking_margin: float = 1.0,
    include_similarity: bool = True,
    include_ranking: bool = False,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Convenience wrapper for the combined loss module."""
    return CombinedLoss(
        similarity_loss_type=similarity_loss_type,
        similarity_beta=similarity_beta,
        lambda_similarity=lambda_similarity,
        lambda_ranking=lambda_ranking,
        ranking_margin=ranking_margin,
        include_similarity=include_similarity,
        include_ranking=include_ranking,
    )(
        similarity_pred=similarity_pred,
        similarity_target=similarity_target,
        predicted_scores=predicted_scores,
        target_scores=target_scores,
    )


__all__ = ["CombinedLoss", "combined_loss"]
