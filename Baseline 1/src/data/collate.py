"""Collation utilities for paired PyTorch Geometric graph samples.

Each dataset item represents a pair of graphs:

    - graph_hypothesis: candidate hypothesis graph
    - graph_ground_truth: corresponding ground-truth graph
    - similarity_target: scalar similarity target in the range [0, 1]

The collate function batches each graph family independently so that hypothesis graphs
and ground-truth graphs are never concatenated into the same graph. This is essential
because the two graph types may have different node counts and different feature
statistics, and they should remain separate inputs for a Siamese or paired encoder.

The function returns a dictionary with the following keys:

    {
        "hypothesis": Batch(...),
        "ground_truth": Batch(...),
        "target": tensor([B]),
        "metadata": [...],  # optional
    }
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data


def _coerce_similarity_target(value: Any, index: int) -> Tensor:
    """Convert a target similarity value into a 1D float tensor."""
    if isinstance(value, Tensor):
        tensor = value.detach().float()
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)

    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor.float()

    raise ValueError(
        f"similarity_target at index {index} must be scalar or 1D, got shape {tuple(tensor.shape)}."
    )


def collate_graph_pairs(batch: Sequence[Mapping[str, Any] | Any]) -> dict[str, Any]:
    """Batch hypothesis and ground-truth graphs independently.

    Args:
        batch: A sequence of items. Each item must provide:
            - ``graph_hypothesis``
            - ``graph_ground_truth``
            - ``similarity_target``

        Optional metadata may be present under a ``metadata`` key or as an attribute
        named ``metadata`` on the item.

    Returns:
        A dictionary containing:
            - ``"hypothesis"``: PyG ``Batch`` for hypothesis graphs
            - ``"ground_truth"``: PyG ``Batch`` for ground-truth graphs
            - ``"target"``: 1D tensor of similarity targets
            - ``"metadata"``: list of item metadata when present
    """
    if len(batch) == 0:
        raise ValueError("collate_graph_pairs received an empty batch.")

    hypothesis_graphs: list[Data] = []
    ground_truth_graphs: list[Data] = []
    similarity_targets: list[Tensor] = []
    metadata: list[Any] = []

    for idx, item in enumerate(batch):
        if isinstance(item, Mapping):
            hypothesis_graph = item.get("graph_hypothesis", item.get("graph_a"))
            ground_truth_graph = item.get("graph_ground_truth", item.get("graph_b"))
            target = item.get("similarity_target", item.get("target"))
            item_metadata = item.get("metadata")
        else:
            hypothesis_graph = getattr(
                item,
                "graph_hypothesis",
                getattr(item, "graph_a", None),
            )
            ground_truth_graph = getattr(
                item,
                "graph_ground_truth",
                getattr(item, "graph_b", None),
            )
            target = getattr(item, "similarity_target", None)
            item_metadata = getattr(item, "metadata", None)

        if hypothesis_graph is None:
            raise ValueError(f"Batch item {idx} is missing the required 'graph_hypothesis' field.")
        if ground_truth_graph is None:
            raise ValueError(f"Batch item {idx} is missing the required 'graph_ground_truth' field.")
        if target is None:
            raise ValueError(f"Batch item {idx} is missing the required 'similarity_target' field.")

        if not isinstance(hypothesis_graph, Data):
            raise TypeError(
                f"graph_hypothesis at index {idx} must be a torch_geometric.data.Data object, "
                f"but received {type(hypothesis_graph).__name__}."
            )
        if not isinstance(ground_truth_graph, Data):
            raise TypeError(
                f"graph_ground_truth at index {idx} must be a torch_geometric.data.Data object, "
                f"but received {type(ground_truth_graph).__name__}."
            )

        hypothesis_graphs.append(hypothesis_graph)
        ground_truth_graphs.append(ground_truth_graph)
        similarity_targets.append(_coerce_similarity_target(target, idx))

        if item_metadata is not None:
            metadata.append(item_metadata)

    hypothesis_batch = Batch.from_data_list(hypothesis_graphs)
    ground_truth_batch = Batch.from_data_list(ground_truth_graphs)
    target_tensor = torch.cat(similarity_targets, dim=0) if len(similarity_targets) > 0 else torch.empty(0)

    assert hypothesis_batch.num_graphs == len(batch), (
        "Hypothesis batch size mismatch: expected one batched graph per item. "
        f"Got hypothesis_batch.num_graphs={hypothesis_batch.num_graphs}, batch_size={len(batch)}."
    )
    assert ground_truth_batch.num_graphs == len(batch), (
        "Ground-truth batch size mismatch: expected one batched graph per item. "
        f"Got ground_truth_batch.num_graphs={ground_truth_batch.num_graphs}, batch_size={len(batch)}."
    )
    assert target_tensor.shape[0] == len(batch), (
        "Target tensor length mismatch: expected one target per batch item. "
        f"Got target.shape={tuple(target_tensor.shape)}, batch_size={len(batch)}."
    )

    result: dict[str, Any] = {
        "hypothesis": hypothesis_batch,
        "ground_truth": ground_truth_batch,
        "target": target_tensor,
    }

    if metadata:
        result["metadata"] = metadata

    return result


__all__ = ["collate_graph_pairs"]
