"""Graph schema for the Baseline 1 graph similarity learning pipeline.

This module defines the expected representation for structural graphs used by the
SGMH-SLAM system. Each graph is a PyTorch Geometric ``Data`` object and stores:

- node features in ``x``
- adjacency information in ``edge_index``
- optional per-edge features in ``edge_attr``
- provenance metadata in ``graph_metadata``

The schema intentionally keeps metadata separate from neural-network inputs so
that it is never confused with model features. The model should consume only the
feature tensors ``x``, ``edge_index``, and optionally ``edge_attr``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

import torch
from torch_geometric.data import Batch, Data

NODE_FEATURE_DIM: int = 8
EDGE_ATTR_DIM: int = 3


@dataclass(frozen=True)
class GraphMetadata:
    """Provenance metadata for a graph instance.

    This structure describes a graph's identity and acquisition context. It is
    intentionally stored as a dedicated attribute and is not part of the feature
    tensor used by the model. All attributes are metadata only and should be used
    for logging, filtering, and evaluation, not as network inputs.

    Attributes:
        graph_id: Unique identifier for the graph instance.
        hypothesis_id: Optional identifier for a candidate hypothesis. This is
            ``None`` for ground-truth graphs.
        timestamp: Optional timestamp associated with the graph.
        is_ground_truth: ``True`` when the graph represents the anchor/ground
            truth graph, otherwise ``False``.
    """

    graph_id: str
    hypothesis_id: Optional[str] = None
    timestamp: Optional[datetime | str] = None
    is_ground_truth: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the metadata.

        Returns:
            A dictionary containing all metadata values, including the timestamp
            when present.
        """

        payload: dict[str, Any] = {
            "graph_id": self.graph_id,
            "hypothesis_id": self.hypothesis_id,
            "timestamp": self.timestamp,
            "is_ground_truth": self.is_ground_truth,
        }
        return payload


class GraphData(Data):
    """PyTorch Geometric graph representation for the Baseline 1 pipeline.

    This object is designed to be directly compatible with ``torch_geometric.data.Data``.
    It represents a single graph and stores the feature tensor ``x`` for nodes, the
    edge connectivity tensor ``edge_index``, optional edge attributes ``edge_attr``,
    and a ``graph_metadata`` attribute that contains provenance information.

    The class is intentionally lightweight and does not contain any model logic.
    It only validates the contract expected by the project.

    Example:
        >>> x = torch.randn(10, 8)
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        >>> graph = GraphData(x=x, edge_index=edge_index)
        >>> graph.x.shape
        torch.Size([10, 8])
    """

    def __init__(
        self,
        x: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: Optional[torch.Tensor] = None,
        graph_metadata: Optional[GraphMetadata] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_metadata=graph_metadata,
            **kwargs,
        )
        if isinstance(self, Batch) and x is None and edge_index is None:
            return
        validate_graph_data(self)


def check_node_feature_dim(
    x: torch.Tensor,
    min_dim: int = NODE_FEATURE_DIM,
) -> None:
    """Validate the dimensionality of the node feature tensor ``x``.

    The baseline graph schema expects each node to carry the same vector length,
    and the minimum feature dimensionality for the initial implementation is 8.

    Required minimum feature ordering for each node:

    - ``x[:, 0]``: point type indicator (0 = constellation, 1 = cone)
    - ``x[:, 1]``: bearing
    - ``x[:, 2]``: nearest-neighbour distance
    - ``x[:, 3]``: number of neighbours in radius ``r``
    - ``x[:, 4:7]``: local covariance eigenvalues ``(lambda_1, lambda_2, lambda_3)``
    - ``x[:, 7]``: eigenvalue ratio ``lambda_2 / lambda_1`` (or 0.0 when unavailable)

    For constellation points, missing geometric statistics may be padded with
    zeros while keeping the same feature length.

    Args:
        x: Tensor with shape ``[num_nodes, num_node_features]``.
        min_dim: Minimum valid feature dimension for the baseline schema.

    Raises:
        TypeError: If ``x`` is not a torch tensor.
        ValueError: If the tensor is not rank-2, has too few features, or has
            zero nodes.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}.")

    if x.dim() != 2:
        raise ValueError(f"x must have shape [num_nodes, num_node_features]; got {tuple(x.shape)}.")

    if x.size(0) < 1:
        raise ValueError("Graph must contain at least one node.")

    if x.size(1) < min_dim:
        raise ValueError(
            "Node feature dimension is too small for the baseline schema: "
            f"expected at least {min_dim}, got {x.size(1)}."
        )


def check_edge_index_dim(edge_index: torch.Tensor) -> None:
    """Validate the edge connectivity tensor ``edge_index``.

    In PyTorch Geometric, ``edge_index`` is typically stored in COO format as a
    tensor of shape ``[2, num_edges]`` where:

    - ``edge_index[0]`` contains source node indices
    - ``edge_index[1]`` contains destination node indices

    Each entry denotes a directed edge from source to destination. For an undirected
    graph, the reverse edge is generally included explicitly unless the message-passing
    layer handles it internally.

    Args:
        edge_index: Edge connectivity tensor with shape ``[2, num_edges]``.

    Raises:
        TypeError: If ``edge_index`` is not a tensor.
        ValueError: If the tensor does not have the expected rank or orientation.
    """

    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"edge_index must be a torch.Tensor, got {type(edge_index).__name__}.")

    if edge_index.dim() != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges]; got {tuple(edge_index.shape)}.")

    if edge_index.size(0) != 2:
        raise ValueError(
            "edge_index must have two rows corresponding to source and target node indices; "
            f"got shape {tuple(edge_index.shape)}."
        )


def check_invalid_node_indices(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> None:
    """Verify that all indices in ``edge_index`` are valid node IDs.

    This check prevents invalid graph connectivity, such as references to nodes
    outside the valid range ``[0, num_nodes - 1]``.

    Args:
        edge_index: Edge connectivity tensor of shape ``[2, num_edges]``.
        num_nodes: Total number of nodes in the graph.

    Raises:
        ValueError: If any edge endpoint is negative or exceeds the number of nodes.
    """

    if edge_index.numel() == 0:
        return

    if not torch.is_floating_point(edge_index) and not torch.is_complex(edge_index):
        if edge_index.min().item() < 0:
            raise ValueError("edge_index contains negative node indices.")

    max_index = int(edge_index.max().item())
    if max_index >= num_nodes:
        raise ValueError(
            f"edge_index contains node indices outside the valid range [0, {num_nodes - 1}]. "
            f"Maximum index observed: {max_index}."
        )


def check_no_nan_or_inf(*tensors: Optional[torch.Tensor], names: Optional[Iterable[str]] = None) -> None:
    """Ensures tensors do not contain ``NaN`` or ``Inf`` values.

    This validation is important for numerical stability and for ensuring that the
    graph structure is valid before training or evaluation.

    Args:
        *tensors: One or more tensors to validate.
        names: Optional names associated to each tensor. If omitted, defaults to
            positional names such as ``tensor_0``.

    Raises:
        TypeError: If a provided object is not a tensor.
        ValueError: If any tensor contains non-finite values.
    """

    if names is None:
        names = [f"tensor_{idx}" for idx in range(len(tensors))]
    else:
        names = list(names)

    if len(names) != len(tensors):
        raise ValueError("The number of tensor names must match the number of tensors.")

    for tensor, name in zip(tensors, names):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

        if tensor.dtype.is_floating_point or tensor.dtype == torch.bfloat16:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"{name} contains NaN or Inf values.")


def validate_graph_data(data: Data) -> None:
    """Validate the full graph object used in the Baseline 1 project.

    This is the central validation entry point. It validates the node feature tensor,
    adjacency connectivity, edge attributes, and finiteness constraints. It should be
    called after graph construction and before passing a graph to a model or dataloader.

    Args:
        data: A ``torch_geometric.data.Data`` object.

    Raises:
        TypeError: If ``data`` is not a ``Data`` instance.
        ValueError: If the graph violates the schema contract.
    """

    if not isinstance(data, Data):
        raise TypeError(f"Expected a torch_geometric.data.Data instance, got {type(data).__name__}.")

    if not hasattr(data, "x") or data.x is None:
        raise ValueError("Graph must define x: node feature tensor with shape [num_nodes, num_node_features].")

    check_node_feature_dim(data.x)
    check_no_nan_or_inf(data.x, names=["x"])

    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Graph must define edge_index: tensor with shape [2, num_edges].")

    check_edge_index_dim(data.edge_index)
    check_invalid_node_indices(data.edge_index, num_nodes=int(data.x.size(0)))
    check_no_nan_or_inf(data.edge_index, names=["edge_index"])

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr
        if edge_attr.dim() != 2:
            raise ValueError(
                f"edge_attr must have shape [num_edges, num_edge_features]; got {tuple(edge_attr.shape)}."
            )
        if edge_attr.size(0) not in (0, data.edge_index.size(1)):
            raise ValueError(
                "edge_attr must have the same number of rows as edge_index; "
                f"expected {data.edge_index.size(1)} rows, got {edge_attr.size(0)}."
            )
        if edge_attr.size(1) < EDGE_ATTR_DIM:
            raise ValueError(
                "edge_attr is too small for the baseline schema: "
                f"expected at least {EDGE_ATTR_DIM} columns, got {edge_attr.size(1)}."
            )
        check_no_nan_or_inf(edge_attr, names=["edge_attr"])


def build_graph_data(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    graph_metadata: Optional[GraphMetadata] = None,
    **kwargs: Any,
) -> GraphData:
    """Construct a validated graph object for the Baseline 1 pipeline.

    This helper mirrors the standard PyG construction pattern while enforcing the
    project-specific schema before returning the object.

    Args:
        x: Node feature tensor with shape ``[num_nodes, num_node_features]``.
        edge_index: Edge connectivity tensor with shape ``[2, num_edges]``.
        edge_attr: Optional edge feature tensor with shape ``[num_edges, num_edge_features]``.
        graph_metadata: Optional provenance metadata.
        **kwargs: Additional fields to attach to the graph object.

    Returns:
        A validated ``GraphData`` instance.
    """

    data = GraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_metadata=graph_metadata,
        **kwargs,
    )
    return data


__all__ = [
    "GraphMetadata",
    "GraphData",
    "NODE_FEATURE_DIM",
    "EDGE_ATTR_DIM",
    "check_node_feature_dim",
    "check_edge_index_dim",
    "check_invalid_node_indices",
    "check_no_nan_or_inf",
    "validate_graph_data",
    "build_graph_data",
]


"""Additional schema documentation.

Node feature contract
---------------------
The feature tensor ``x`` represents one node per row. The baseline schema expects a
feature matrix of shape ``[num_nodes, num_node_features]``. The minimum supported
feature vector is length 8, with the following ordering:

1. ``point_type``: a numeric indicator, typically ``0`` for constellation and ``1``
   for cone points.
2. ``bearing``: the local bearing angle for the point, generally in radians.
3. ``nearest_neighbour_distance``: the distance to the nearest nearby point.
4. ``num_neighbours_in_radius_r``: number of neighbouring points within radius ``r``.
5. ``local_covariance_eigenvalues``: three values corresponding to covariance eigenvalues
   ``(lambda_1, lambda_2, lambda_3)`` for a local neighbourhood.
6. ``eigenvalue_ratio``: a ratio such as ``lambda_2 / lambda_1``.

For points that do not have all statistics available (for example, constellation points
that do not carry local covariance information), the absent values may be zero-filled
while preserving a fixed feature width across all nodes in the graph.

Edge representation
-------------------
``edge_index`` is a tensor of shape ``[2, num_edges]`` in COO format. Each column is
one directed edge: ``edge_index[0, k]`` is the source node, and ``edge_index[1, k]`` is
its destination. This is the standard representation used by PyTorch Geometric. For
undirected graphs, the reverse edge is typically added explicitly.

``edge_attr`` is optional and may store per-edge information such as:

- edge type: ``0`` = cone-cone, ``1`` = cone-constellation, ``2`` = constellation-constellation
- squared Euclidean distance between the connected nodes
- relative orientation with respect to neighbouring edges

The baseline schema expects a matrix of shape ``[num_edges, 3]`` when edge attributes are
present, and each feature should be finite.

Tensor dimensions
-----------------
The expected shapes are:

- ``x``: ``[num_nodes, num_node_features]``
- ``edge_index``: ``[2, num_edges]``
- ``edge_attr``: ``[num_edges, num_edge_features]`` when present

The actual number of node features may grow in future work, but the baseline contract
requires at least 8 features per node.

Batching behavior in PyTorch Geometric
-------------------------------------
PyTorch Geometric datasets are usually combined with ``DataLoader``. When graphs are
batched, the library concatenates node features and reindexes edges so that each graph
remains valid under the same batch. The internal ``batch`` tensor tracks which graph each
node belongs to, but the model should operate on the batched tensors rather than on the
raw graph metadata.

This module does not add any model-specific operation; it only defines the data schema
used by the project. ``graph_metadata`` must remain separate from the feature tensors so
that it is never mistakenly passed into the encoder or similarity network.
"""
