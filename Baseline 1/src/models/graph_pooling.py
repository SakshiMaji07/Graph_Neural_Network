"""Graph-level pooling for the Siamese GNN.

A message-passing encoder produces node embeddings for each graph. However, the
similarity network needs one fixed-dimensional representation per graph, not one
embedding per node. Graph pooling is therefore required to aggregate node-level
information into a single graph embedding.

This is essential for graph-level similarity because different graphs may have
different numbers of nodes and edges, yet the downstream similarity network expects
an input of fixed dimension. Pooling maps a variable-size set of node embeddings to
one vector per graph, enabling direct comparison of graphs of different sizes.

In PyTorch Geometric, the batch vector tracks which nodes belong to which graph. The
pooling layer must respect these group boundaries so that nodes from different graphs
are never mixed together during aggregation.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.data import Batch


PoolingType = Literal["mean", "sum", "max", "mean_max"]


class GraphPooling(nn.Module):
    """Aggregate node embeddings into fixed-size graph embeddings.

    The module accepts node embeddings of shape ``[N_total, D]`` together with a
    PyG batch vector of shape ``[N_total]`` and returns a graph embedding of shape
    ``[num_graphs, D]`` or ``[num_graphs, 2 * D]`` for combined mean+max pooling.

    Args:
        pooling: Pooling mode. Supported values are:
            - ``"mean"``: average over nodes in each graph
            - ``"sum"``: sum over nodes in each graph
            - ``"max"``: elementwise max over nodes in each graph
            - ``"mean_max"``: concatenate mean and max pooled outputs
    """

    def __init__(self, pooling: PoolingType = "mean") -> None:
        super().__init__()

        supported = {"mean", "sum", "max", "mean_max"}
        if pooling not in supported:
            raise ValueError(f"Unsupported pooling mode '{pooling}'. Supported: {sorted(supported)}.")

        self.pooling = pooling

    def _validate_inputs(self, x: torch.Tensor, batch: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate that the node embeddings and batch vector are well-formed."""

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}.")
        if x.dim() != 2:
            raise ValueError(f"x must have shape [N_total, D], got {tuple(x.shape)}.")
        if x.size(0) == 0:
            raise ValueError("x must contain at least one node embedding.")

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if not isinstance(batch, torch.Tensor):
            raise TypeError(f"batch must be a torch.Tensor or None, got {type(batch).__name__}.")
        if batch.dim() != 1:
            raise ValueError(f"batch must have shape [N_total], got {tuple(batch.shape)}.")
        if batch.size(0) != x.size(0):
            raise ValueError(
                f"batch length must match x.shape[0]; got {batch.size(0)} and {x.size(0)}."
            )

        return x, batch

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Pool node embeddings into graph-level embeddings.

        Args:
            x: Node embeddings with shape ``[N_total, D]``.
            batch: Optional PyG batch vector of shape ``[N_total]``. If omitted, all
                nodes are treated as belonging to a single graph.

        Returns:
            Graph embeddings with shape ``[num_graphs, D]`` or ``[num_graphs, 2 * D]``
            depending on the chosen pooling mode.
        """

        x, batch = self._validate_inputs(x, batch)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        if self.pooling == "mean":
            return scatter_mean(x, batch, dim_size=num_graphs)

        if self.pooling == "sum":
            return scatter_sum(x, batch, dim_size=num_graphs)

        if self.pooling == "max":
            return scatter_max(x, batch, dim_size=num_graphs)

        if self.pooling == "mean_max":
            mean_pool = scatter_mean(x, batch, dim_size=num_graphs)
            max_pool = scatter_max(x, batch, dim_size=num_graphs)
            return torch.cat([mean_pool, max_pool], dim=-1)

        raise ValueError(f"Unsupported pooling mode '{self.pooling}'.")


def scatter_mean(x: torch.Tensor, batch: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Compute mean aggregation within each graph in a batch.

    This respects the PyG batch vector so that aggregation happens only within each
    graph's node set, never across distinct graphs in the same batch.
    """

    out = x.new_zeros(dim_size, x.size(-1))
    count = x.new_zeros(dim_size, x.size(-1))
    count.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)), torch.ones_like(x))
    out.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)), x)
    return out / torch.clamp(count, min=1.0)


def scatter_sum(x: torch.Tensor, batch: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Compute sum aggregation within each graph in a batch."""

    out = x.new_zeros(dim_size, x.size(-1))
    out.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)), x)
    return out


def scatter_max(x: torch.Tensor, batch: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Compute max aggregation within each graph in a batch."""

    out = x.new_full((dim_size, x.size(-1)), float("-inf"), device=x.device, dtype=x.dtype)
    out = out.index_scatter(0, batch, x, reduce="amax")
    return out


__all__ = ["GraphPooling", "PoolingType", "scatter_mean", "scatter_sum", "scatter_max"]
