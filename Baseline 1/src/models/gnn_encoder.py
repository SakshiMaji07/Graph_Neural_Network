"""Shared graph encoder for Siamese graph similarity learning.

This module defines the encoder used to process both a SLAM hypothesis graph and a
ground-truth graph with the same set of parameters. The encoder is shared by design:

    z_H = Encoder(H)
    z_GT = Encoder(GT)

Both graphs are passed through the same encoder instance, meaning that there are no
separate branches or independent weights for the hypothesis and ground-truth inputs.

The encoder works with PyTorch Geometric graphs and returns node embeddings before
pooling. A separate pooling step in ``graph_pooling.py`` handles graph-level summary
vectors.

Message passing interpretation
-----------------------------
At layer ``l``, each node updates itself by aggregating information from its neighbors:

    h_i^(l+1) = UPDATE(h_i^(l), AGGREGATE({h_j^(l): j in N(i)}))

This is the standard node update used by message-passing GNNs. The encoder may use
GCN-like normalization or GraphSAGE-style aggregation depending on the configured
convolution operator.

The implementation supports variable graph sizes because it operates on each graph
individually and uses per-graph adjacency patterns. It is compatible with PyG batch
objects, where nodes from different graphs are concatenated and the batch vector is
used to keep each graph's nodes together.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, SAGEConv


ActivationFactory = Callable[[int], nn.Module]


class GraphEncoder(nn.Module):
    """Shared graph encoder for Siamese graph comparison.

    The encoder is configurable and supports a standard message-passing stack of
    configurable convolution layers. It returns node-level embeddings by default,
    and can optionally return intermediate representations or a pooled graph-level
    encoding if needed by other components.

    Args:
        input_dim: Dimensionality of each node feature vector.
        hidden_dim: Hidden feature width for convolution layers.
        embedding_dim: Output feature width of the final node embedding.
        num_layers: Number of message-passing layers.
        dropout: Dropout probability applied after each layer.
        activation: Name of the activation function to use. Supported values are
            ``"relu"``, ``"elu"``, ``"gelu"``, ``"leaky_relu"``, and ``"tanh"``.
        conv_type: Convolution type to use. Supported values are ``"gcn"`` and
            ``"sage"``.
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
        self.activation_name = activation.lower()
        self.conv_type = conv_type.lower()

        if self.conv_type not in {"gcn", "sage"}:
            raise ValueError(f"Unsupported conv_type '{conv_type}'. Supported: 'gcn', 'sage'.")

        self.activation = self._build_activation(self.activation_name)
        self.convs = self._build_convs()

    def _build_activation(self, activation_name: str) -> nn.Module:
        """Create the configured activation function."""

        activations: Dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
            "tanh": nn.Tanh(),
        }

        if activation_name not in activations:
            raise ValueError(
                f"Unsupported activation '{activation_name}'. Supported: {sorted(activations.keys())}."
            )
        return activations[activation_name]

    def _build_convs(self) -> nn.ModuleList:
        """Build the message-passing layers.

        The layer widths are:

        - first layer: input_dim -> hidden_dim
        - middle layers: hidden_dim -> hidden_dim
        - final layer: hidden_dim -> embedding_dim

        If ``num_layers == 1``, the encoder uses a single projection from input_dim to
        embedding_dim.
        """

        layers: list[nn.Module] = []

        if self.num_layers == 1:
            in_dim, out_dim = self.input_dim, self.embedding_dim
            layers.append(self._make_conv_layer(in_dim, out_dim))
            return nn.ModuleList(layers)

        layers.append(self._make_conv_layer(self.input_dim, self.hidden_dim))
        for _ in range(self.num_layers - 2):
            layers.append(self._make_conv_layer(self.hidden_dim, self.hidden_dim))
        layers.append(self._make_conv_layer(self.hidden_dim, self.embedding_dim))
        return nn.ModuleList(layers)

    def _make_conv_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create a configured convolution layer."""

        if self.conv_type == "gcn":
            return GCNConv(in_channels=in_dim, out_channels=out_dim)
        if self.conv_type == "sage":
            return SAGEConv(in_channels=in_dim, out_channels=out_dim)
        raise ValueError(f"Unsupported conv_type '{self.conv_type}'.")

    def encode_nodes(self, data: Data) -> torch.Tensor:
        """Encode node features and return node embeddings.

        Args:
            data: A PyG ``Data`` or ``Batch`` object with node features in ``data.x`` and
                adjacency information in ``data.edge_index``.

        Returns:
            Tensor of node embeddings with shape ``[num_nodes, embedding_dim]``.
        """

        if not hasattr(data, "x") or data.x is None:
            raise ValueError("Graph data must contain an 'x' tensor with node features.")
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("Graph data must contain an 'edge_index' tensor.")

        x = data.x
        if x.dim() != 2:
            raise ValueError(f"Expected x to have shape [num_nodes, num_features], got {tuple(x.shape)}.")
        if x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim} but got feature dimension {x.size(1)}."
            )

        edge_index = data.edge_index
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "Expected edge_index to have shape [2, num_edges], "
                f"got {tuple(edge_index.shape)}."
            )

        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if layer_idx < len(self.convs) - 1:
                x = self.activation(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass returning node embeddings by default.

        The model is intentionally designed to return node embeddings first. Graph-level
        pooling is handled in ``graph_pooling.py`` so that the same encoder can be used
        for graphs with different sizes and shapes.

        Args:
            data: PyG ``Data`` or ``Batch`` object.

        Returns:
            Tensor of node embeddings with shape ``[num_nodes, embedding_dim]``.
        """

        return self.encode_nodes(data)


__all__ = ["GraphEncoder"]
