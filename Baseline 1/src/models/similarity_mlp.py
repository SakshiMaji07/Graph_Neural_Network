"""MLP for predicting continuous graph similarity from two embedding vectors.

The model receives two graph-level embeddings,

- ``A`` with shape ``[B, D]``
- ``B`` with shape ``[B, D]``

and predicts a scalar similarity score in the range ``[0, 1]`` for each pair.

Why use absolute difference and elementwise product?
----------------------------------------------------
The comparison vector is designed to capture complementary notions of similarity:

1. ``abs(A - B)`` measures how far the two graph embeddings are apart in feature
   space. Large absolute differences indicate that the two graphs are structurally
   different.
2. ``A * B`` measures aligned activation strength: if two embeddings agree on the
   same latent dimensions, their elementwise product is large. This reveals
   co-activation patterns and encourages the model to capture shared structure.

Together, these two operators provide a compact and informative comparison signal
without requiring explicit node correspondence. They allow the network to learn one
scalar score from two graph embeddings while keeping the architecture simple and
interpretable.

The network is intentionally separate from the graph encoder and graph pooling. It
receives fixed-dimensional graph representations and outputs a continuous similarity
score that can be used as a regression target for the Siamese model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SimilarityMLP(nn.Module):
    """MLP that compares two graph embeddings and predicts a similarity score.

    Args:
        input_dim: Dimensionality of the comparison vector after concatenating the
            selected comparison features.
        hidden_dims: Hidden layer widths for the MLP.
        dropout: Dropout probability applied after each hidden layer.
        include_raw: Whether to include the raw embeddings ``A`` and ``B`` in the
            comparison vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.0,
        include_raw: bool = False,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        if hidden_dims is None:
            hidden_dims = [64, 32]

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer width.")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"All hidden dimensions must be positive, got {hidden_dims}.")

        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.include_raw = bool(include_raw)

        current_dim = self.input_dim
        layers: list[nn.Module] = []

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def build_comparison_vector(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the comparison feature vector for two graph embeddings.

        The default comparison vector includes:

        1. ``abs(A - B)``
        2. ``A * B``

        Optional raw embeddings may also be appended.

        Args:
            embedding_a: Graph embedding A with shape ``[B, D]``.
            embedding_b: Graph embedding B with shape ``[B, D]``.

        Returns:
            Comparison tensor with shape ``[B, comparison_dim]``.
        """

        if embedding_a.shape != embedding_b.shape:
            raise ValueError(
                "embedding_a and embedding_b must have the same shape; "
                f"got {tuple(embedding_a.shape)} and {tuple(embedding_b.shape)}."
            )
        if embedding_a.dim() != 2:
            raise ValueError(
                "Each embedding must have shape [batch_size, feature_dim]; "
                f"got {tuple(embedding_a.shape)}."
            )

        features = [torch.abs(embedding_a - embedding_b), embedding_a * embedding_b]
        if self.include_raw:
            features.extend([embedding_a, embedding_b])

        comparison_vector = torch.cat(features, dim=-1)
        return comparison_vector

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        """Predict a similarity score for each pair of graph embeddings.

        Args:
            embedding_a: Tensor of shape ``[B, D]``.
            embedding_b: Tensor of shape ``[B, D]``.

        Returns:
            Tensor of shape ``[B]`` containing similarity scores in ``[0, 1]``.
        """

        comparison_vector = self.build_comparison_vector(embedding_a, embedding_b)
        logits = self.network(comparison_vector)
        return logits.squeeze(-1)


__all__ = ["SimilarityMLP"]
