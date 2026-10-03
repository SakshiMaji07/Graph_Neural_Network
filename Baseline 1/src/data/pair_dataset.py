"""Pair dataset construction for Siamese graph similarity learning.

This module builds training examples of the form:

- ``(graph_a, graph_b, similarity_target)``

for a Siamese graph neural network. In the SGMH-SLAM setting, each sample is the
pair ``(H_i, G_GT)`` where ``H_i`` is one candidate SLAM hypothesis and ``G_GT`` is
its corresponding ground-truth structural graph.

The dataset intentionally does not generate node-level correspondence labels. The
model focuses on learning a similarity score between two graphs, not a dense node
matching assignment.

Why Siamese graph networks can compare graphs of different sizes
----------------------------------------------------------------
A Siamese graph network processes each graph with the same encoder, producing a
graph-level embedding. This is possible even when the graphs have different numbers
of nodes, because the encoder is applied independently to each graph and the
resulting representations are pooled into fixed-length vectors. The pooling step
(or graph readout) maps variable-size graph structure into a fixed-dimensional
embedding, allowing valid comparison across different graph sizes.

The dataset therefore stores graphs as independent ``torch_geometric.data.Data``
objects without requiring equal node counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch_geometric.data import Data

from .graph_loader import GraphLoader


@dataclass(frozen=True)
class PairMetadata:
    """Optional ranking metadata attached to a graph pair.

    This metadata is used for evaluation and analysis; it is not used as model
    input unless the training code explicitly chooses to include it.

    Attributes:
        hypothesis_id: Identifier of the candidate hypothesis graph.
        scenario_id: Identifier of the full sample or scenario.
        ground_truth_id: Identifier of the ground-truth graph.
    """

    hypothesis_id: Optional[str] = None
    scenario_id: Optional[str] = None
    ground_truth_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the pair metadata."""

        return {
            "hypothesis_id": self.hypothesis_id,
            "scenario_id": self.scenario_id,
            "ground_truth_id": self.ground_truth_id,
        }


@dataclass(frozen=True)
class PairExample:
    """A single graph-pair training example.

    Attributes:
        graph_a: The first graph in the pair, typically a hypothesis graph.
        graph_b: The second graph in the pair, typically the ground-truth graph.
        similarity_target: Continuous similarity target in the range ``[0, 1]``.
        metadata: Optional metadata describing the ranking context of the pair.
    """

    graph_a: Data
    graph_b: Data
    similarity_target: torch.Tensor
    metadata: Optional[PairMetadata] = None


class SLAMGraphPairDataset:
    """Dataset of hypothesis-ground-truth graph pairs for Siamese learning.

    Each dataset item is a pair ``(graph_a, graph_b, similarity_target)``. The pair is
    designed for a Siamese encoder that uses the same graph encoder on both inputs and
    then compares their graph-level embeddings.

    The dataset supports constructing pairs from a list of tuples:

    ``(hypothesis_path, ground_truth_path, similarity)``

    with optional metadata fields for ranking and scenario information.
    """

    def __init__(
        self,
        graph_loader: Optional[GraphLoader] = None,
        normalize: bool = False,
    ) -> None:
        self.graph_loader = graph_loader or GraphLoader(normalize=normalize)
        self.normalize = normalize
        self._items: list[PairExample] = []

    def __len__(self) -> int:
        """Return the number of pair examples in the dataset."""

        return len(self._items)

    def __getitem__(self, index: int) -> PairExample:
        """Return the pair example at the specified index."""

        if index < 0 or index >= len(self._items):
            raise IndexError(f"PairDataset index out of range: {index}")
        return self._items[index]

    def add_pair(
        self,
        graph_a: Data,
        graph_b: Data,
        similarity: float | torch.Tensor,
        metadata: Optional[PairMetadata] = None,
    ) -> None:
        """Append a single pair example to the dataset.

        Args:
            graph_a: The first graph in the pair.
            graph_b: The second graph in the pair.
            similarity: Similarity target in the range ``[0, 1]``.
            metadata: Optional ranking metadata.
        """

        target = _validate_similarity_target(similarity)
        self._items.append(
            PairExample(
                graph_a=graph_a,
                graph_b=graph_b,
                similarity_target=target,
                metadata=metadata,
            )
        )

    def add_from_paths(
        self,
        hypothesis_path: str | Path,
        ground_truth_path: str | Path,
        similarity: float | torch.Tensor,
        hypothesis_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        ground_truth_id: Optional[str] = None,
    ) -> None:
        """Load a graph pair from disk using the project graph loader.

        Args:
            hypothesis_path: Path to the hypothesis graph file.
            ground_truth_path: Path to the ground-truth graph file.
            similarity: Target similarity in ``[0, 1]``.
            hypothesis_id: Optional hypothesis identifier.
            scenario_id: Optional scenario identifier.
            ground_truth_id: Optional ground-truth graph identifier.
        """

        graph_a = self.graph_loader.load(hypothesis_path)
        graph_b = self.graph_loader.load(ground_truth_path)
        metadata = PairMetadata(
            hypothesis_id=str(hypothesis_id) if hypothesis_id is not None else None,
            scenario_id=str(scenario_id) if scenario_id is not None else None,
            ground_truth_id=str(ground_truth_id) if ground_truth_id is not None else None,
        )
        self.add_pair(graph_a, graph_b, similarity, metadata=metadata)

    def from_pairs(
        self,
        pairs: Iterable[tuple[str | Path, str | Path, float | torch.Tensor]],
        *,
        hypothesis_ids: Optional[Iterable[Optional[str]]] = None,
        scenario_ids: Optional[Iterable[Optional[str]]] = None,
        ground_truth_ids: Optional[Iterable[Optional[str]]] = None,
    ) -> "SLAMGraphPairDataset":
        """Construct a dataset from a list of ``(hypothesis_path, gt_path, similarity)`` tuples.

        Args:
            pairs: Iterable of triplets containing the hypothesis path, ground-truth path,
                and similarity score.
            hypothesis_ids: Optional list of hypothesis IDs aligned with the pairs.
            scenario_ids: Optional list of scenario IDs aligned with the pairs.
            ground_truth_ids: Optional list of ground-truth IDs aligned with the pairs.

        Returns:
            A dataset populated with the constructed pairs.
        """

        pair_list = list(pairs)
        if hypothesis_ids is not None and len(hypothesis_ids) != len(pair_list):
            raise ValueError("hypothesis_ids length must match the number of pairs.")
        if scenario_ids is not None and len(scenario_ids) != len(pair_list):
            raise ValueError("scenario_ids length must match the number of pairs.")
        if ground_truth_ids is not None and len(ground_truth_ids) != len(pair_list):
            raise ValueError("ground_truth_ids length must match the number of pairs.")

        for idx, (hypothesis_path, ground_truth_path, similarity) in enumerate(pair_list):
            metadata = PairMetadata(
                hypothesis_id=(hypothesis_ids[idx] if hypothesis_ids is not None else None),
                scenario_id=(scenario_ids[idx] if scenario_ids is not None else None),
                ground_truth_id=(ground_truth_ids[idx] if ground_truth_ids is not None else None),
            )
            self.add_from_paths(
                hypothesis_path=hypothesis_path,
                ground_truth_path=ground_truth_path,
                similarity=similarity,
                hypothesis_id=metadata.hypothesis_id,
                scenario_id=metadata.scenario_id,
                ground_truth_id=metadata.ground_truth_id,
            )

        return self

    def from_five_hypothesis_sample(
        self,
        ground_truth_path: str | Path,
        hypothesis_paths: Iterable[str | Path],
        similarities: Iterable[float | torch.Tensor],
        *,
        scenario_id: Optional[str] = None,
        ground_truth_id: Optional[str] = None,
    ) -> "SLAMGraphPairDataset":
        """Generate the five paired examples for one sample.

        For a ground-truth graph ``G`` and five hypotheses ``H1...H5``, this function
        generates:

        - (H1, G, y1)
        - (H2, G, y2)
        - (H3, G, y3)
        - (H4, G, y4)
        - (H5, G, y5)

        The order is deterministic and follows the provided iterable order.

        Args:
            ground_truth_path: Path to the ground-truth graph file.
            hypothesis_paths: Iterable of five candidate hypothesis graph paths.
            similarities: Iterable of target similarity values aligned with the
                hypothesis list.
            scenario_id: Optional scenario identifier for the sample.
            ground_truth_id: Optional ground-truth identifier.

        Returns:
            The dataset populated with the generated pairs.
        """

        hypothesis_list = list(hypothesis_paths)
        similarity_list = list(similarities)

        if len(hypothesis_list) != 5:
            raise ValueError(
                "Expected exactly five hypothesis graphs to generate a five-hypothesis sample; "
                f"got {len(hypothesis_list)}."
            )
        if len(similarity_list) != 5:
            raise ValueError(
                "Expected five similarity targets for the five-hypothesis sample; "
                f"got {len(similarity_list)}."
            )

        for index, (hypothesis_path, similarity) in enumerate(zip(hypothesis_list, similarity_list)):
            hypothesis_id = f"hypothesis_{index + 1}"
            self.add_from_paths(
                hypothesis_path=hypothesis_path,
                ground_truth_path=ground_truth_path,
                similarity=similarity,
                hypothesis_id=hypothesis_id,
                scenario_id=scenario_id,
                ground_truth_id=ground_truth_id,
            )

        return self


def _validate_similarity_target(similarity: float | torch.Tensor) -> torch.Tensor:
    """Validate and convert a similarity target to a float tensor."""

    if isinstance(similarity, torch.Tensor):
        if similarity.numel() != 1:
            raise ValueError(f"Similarity target must be a scalar tensor, got shape {tuple(similarity.shape)}.")
        value = float(similarity.detach().cpu().item())
    else:
        value = float(similarity)

    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Similarity target must be in the range [0, 1], got {value}.")

    return torch.tensor(value, dtype=torch.float32)


__all__ = [
    "PairMetadata",
    "PairExample",
    "SLAMGraphPairDataset",
    "_validate_similarity_target",
]
