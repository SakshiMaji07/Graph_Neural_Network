"""Baseline Siamese GNN for graph similarity learning in SGMH-SLAM.

The model handles one hypothesis graph and one ground-truth graph and learns a
continuous similarity score in the interval ``[0, 1]``.

Computational graph
-------------------
For a hypothesis graph ``H`` and ground-truth graph ``G``:

    H ──> Shared GraphEncoder ──> Node embeddings ──> GraphPooling ──> z_H
    G ──> Shared GraphEncoder ──> Node embeddings ──> GraphPooling ──> z_GT

    z_H, z_GT ──> comparison vector ──> SimilarityMLP ──> similarity score

The key architectural constraint is that the same encoder instance is used for both
branches. There is exactly one set of encoder parameters shared across the two graphs.
This is the Siamese design: the same feature extractor is applied to both inputs,
allowing the model to learn a metric consistent across the hypothesis and ground-truth
space.

The graph pooling step converts variable-size node sets into fixed-dimensional graph
embeddings, which are then compared by a small MLP that predicts similarity.

This file intentionally focuses on the architecture and computational flow; it does
not implement training loops or loss functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from .gnn_encoder import GraphEncoder
from .graph_pooling import GraphPooling
from .similarity_mlp import SimilarityMLP


@dataclass
class SiameseOutputs:
    """Container for the outputs of a Siamese graph comparison.

    Attributes:
        similarity_score: Scalar similarity prediction for each pair in the batch.
        graph_embedding_h: Graph embedding for the hypothesis branch.
        graph_embedding_gt: Graph embedding for the ground-truth branch.
    """

    similarity_score: torch.Tensor
    graph_embedding_h: torch.Tensor
    graph_embedding_gt: torch.Tensor


class SiameseGNN(nn.Module):
    """Baseline Siamese graph neural network for hypothesis-vs-ground-truth scoring.

    This model accepts two PyTorch Geometric graphs or batched graphs, encodes them
    with the same ``GraphEncoder`` instance, pools node embeddings to graph embeddings,
    compares the graph embeddings, and produces a continuous similarity score in the
    range ``[0, 1]``.

    Args:
        input_dim: Input node feature dimensionality.
        hidden_dim: Hidden dimensionality inside the shared encoder.
        embedding_dim: Node embedding size output by the shared encoder.
        num_layers: Number of message-passing layers in the encoder.
        dropout: Dropout probability inside the encoder and similarity MLP.
        activation: Activation function used by the encoder.
        conv_type: Convolution type used by the encoder (``"gcn"`` or ``"sage"``).
        pooling: Graph pooling mode used after encoding.
        mlp_hidden_dims: Hidden dimensions of the comparison MLP.
        include_raw_features: Whether the comparison MLP includes raw embedding values.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        conv_type: str = "gcn",
        pooling: str = "mean",
        mlp_hidden_dims: Optional[list[int]] = None,
        include_raw_features: bool = False,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.activation = activation.lower()
        self.conv_type = conv_type.lower()
        self.pooling = pooling

        self.encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            conv_type=conv_type,
        )

        self.pool = GraphPooling(pooling=pooling)

        comparison_input_dim = 2 * embedding_dim
        if include_raw_features:
            comparison_input_dim += 2 * embedding_dim

        self.similarity_mlp = SimilarityMLP(
            input_dim=comparison_input_dim,
            hidden_dims=mlp_hidden_dims or [64, 32],
            dropout=dropout,
            include_raw=include_raw_features,
        )

        self.include_raw_features = include_raw_features

    def encode_graph(self, graph: Any) -> torch.Tensor:
        """Return the graph embedding for a single graph.

        This method applies the same shared encoder and pooling to a single graph.

        Args:
            graph: A PyTorch Geometric ``Data`` or ``Batch`` object.

        Returns:
            Graph embedding tensor of shape ``[num_graphs_in_batch, embedding_dim]``.
        """

        node_embeddings = self.encoder.encode_nodes(graph)

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)

        graph_embedding = self.pool(node_embeddings, batch=batch)
        return graph_embedding

    def compare_graphs(
        self,
        graph_embedding_h: torch.Tensor,
        graph_embedding_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compare two graph embeddings and predict a continuous similarity score.

        Args:
            graph_embedding_h: Embedding tensor for hypothesis graphs.
            graph_embedding_gt: Embedding tensor for ground-truth graphs.

        Returns:
            Predicted similarity scores with shape ``[B]`` in the range ``[0, 1]``.
        """

        if graph_embedding_h.shape != graph_embedding_gt.shape:
            raise ValueError(
                "Graph embeddings must have the same shape for comparison; "
                f"got {tuple(graph_embedding_h.shape)} and {tuple(graph_embedding_gt.shape)}."
            )

        similarity_score = self.similarity_mlp(graph_embedding_h, graph_embedding_gt)
        return torch.clamp(similarity_score, min=0.0, max=1.0)

    def forward(
        self,
        hypothesis_graph: Any,
        ground_truth_graph: Any,
        *,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the similarity score between a hypothesis graph and a ground truth.

        Args:
            hypothesis_graph: PyG ``Data`` or ``Batch`` object for the hypothesis.
            ground_truth_graph: PyG ``Data`` or ``Batch`` object for the ground truth.
            return_embeddings: If ``True``, also return the graph embeddings for both
                branches.

        Returns:
            If ``return_embeddings`` is ``False``:
                similarity score tensor with shape ``[B]`` or ``[num_graphs]``.
            If ``return_embeddings`` is ``True``:
                tuple ``(similarity_score, z_h, z_gt)``.
        """

        z_h = self.encode_graph(hypothesis_graph)
        z_gt = self.encode_graph(ground_truth_graph)

        similarity_score = self.compare_graphs(z_h, z_gt)

        if return_embeddings:
            return similarity_score, z_h, z_gt
        return similarity_score

    def parameter_count(self) -> int:
        """Return the total number of trainable parameters in the model."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["SiameseGNN", "SiameseOutputs"]
