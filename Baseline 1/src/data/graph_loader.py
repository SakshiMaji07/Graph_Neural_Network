"""Graph loading utilities for the Baseline 1 SLAM graph similarity pipeline.

Dataset format
--------------
This loader expects each serialized sample to be a single ``.npz`` file containing
at least the following arrays:

- ``node_features``: shape ``[N, F]``
- ``edge_index``: shape ``[2, E]``

The loader also accepts an optional array:

- ``edge_attr``: shape ``[E, D]``

The expected node feature matrix is defined by ``src.data.graph_schema``. In the
initial baseline, each node feature vector should contain the following minimum values:

1. point type
2. bearing
3. nearest neighbour distance
4. neighbour count within radius ``r``
5. local covariance eigenvalues
6. eigenvalue ratio

The edge tensor is expected to be stored in PyTorch Geometric COO format:
``edge_index[0, k]`` is the source node index and ``edge_index[1, k]`` is the target
node index for the ``k``-th edge. When edge features are provided, they are stored as
``[num_edges, num_edge_features]`` and each row describes one edge.

Metadata handling
-----------------
Metadata such as ``graph_id``, ``hypothesis_id``, ``timestamp`` and
``is_ground_truth`` must be preserved separately from the node and edge features.
These values are stored in ``graph_metadata`` and are not passed into the model as
feature tensors.

Normalization
-------------
Normalization is intentionally disabled by default. The loader does not modify the
feature statistics unless ``normalize=True`` is explicitly configured. This keeps the
loader deterministic and makes it easy to reproduce research experiments.

Important notes
---------------
- The loader returns a ``torch_geometric.data.Data`` instance compatible with the
  project's schema.
- Graphs are validated before use to catch invalid indices, NaN/Inf values, and
  empty or malformed arrays.
- The implementation is deterministic: files are processed in sorted path order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from .graph_schema import GraphData, GraphMetadata, check_no_nan_or_inf, validate_graph_data


def _as_bool(value: Any) -> bool:
    """Convert common metadata representations to a Python bool."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Could not interpret value {value!r} as a boolean metadata flag.")


def _safe_metadata_value(value: Any) -> Any:
    """Normalize metadata values from NumPy arrays or object arrays to Python scalars."""

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    return value


class GraphLoader:
    """Load serialized SLAM graph samples into ``torch_geometric.data.Data`` objects.

    The loader supports dataset files in ``.npz`` format and validates the graph
    structure before returning a graph instance. It also preserves metadata in a
    dedicated ``graph_metadata`` field.

    Args:
        normalize: If ``True``, apply optional normalization to node features using
            ``normalize_graph``. By default this is ``False`` to preserve the raw
            dataset statistics.
        strict: If ``True``, raise an error on invalid or malformed arrays. If
            ``False``, partial data may be tolerated but this is not recommended for
            research experiments.
    """

    def __init__(self, normalize: bool = False, strict: bool = True) -> None:
        self.normalize = normalize
        self.strict = strict

    def load(self, path: str | Path) -> GraphData:
        """Load a single graph from a ``.npz`` file.

        Args:
            path: Path to a serialized graph file.

        Returns:
            A validated ``GraphData`` object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file does not contain the required arrays or if the
                graph is malformed.
        """

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")
        if file_path.suffix.lower() != ".npz":
            raise ValueError(f"Unsupported graph file format: {file_path.suffix}. Expected '.npz'.")

        with np.load(file_path, allow_pickle=True) as archive:
            graph = self._from_archive(archive, source=file_path)
        return graph

    def load_directory(
        self,
        directory: str | Path,
        pattern: str = "*.npz",
        recursive: bool = False,
    ) -> list[GraphData]:
        """Load all graph files from a directory.

        Files are processed in sorted order to keep the result deterministic and
        reproducible across runs.

        Args:
            directory: Directory containing the graph files.
            pattern: Glob pattern for matching graph files.
            recursive: If ``True``, search recursively.

        Returns:
            A list of validated ``GraphData`` objects.
        """

        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Graph directory not found: {directory_path}")
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Expected a directory, got: {directory_path}")

        if recursive:
            file_paths = sorted(directory_path.rglob(pattern))
        else:
            file_paths = sorted(directory_path.glob(pattern))

        if not file_paths:
            raise FileNotFoundError(
                f"No graph files matching pattern '{pattern}' were found in '{directory_path}'."
            )

        return [self.load(path) for path in file_paths]

    def _from_archive(self, archive: np.lib.npyio.NpzFile, source: Path | str) -> GraphData:
        """Convert a NumPy archive into a ``GraphData`` instance."""

        missing_fields = [name for name in ("node_features", "edge_index") if name not in archive]
        if missing_fields:
            raise ValueError(
                f"Graph file '{source}' is missing required fields: {', '.join(missing_fields)}. "
                "Expected 'node_features' and 'edge_index'."
            )

        node_features = self._to_node_features(archive["node_features"])
        edge_index = self._to_edge_index(archive["edge_index"], num_nodes=node_features.shape[0])

        edge_attr = None
        if "edge_attr" in archive:
            edge_attr = self._to_edge_attr(archive["edge_attr"], num_edges=edge_index.shape[1])

        metadata = self._extract_metadata(archive)

        data = GraphData(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_metadata=metadata,
        )

        if self.normalize:
            data = normalize_graph(data)

        validate_graph_data(data)
        data.source_path = str(source)
        return data

    def _to_node_features(self, array: np.ndarray) -> torch.Tensor:
        """Convert node feature data into a float32 tensor."""

        node_features = np.asarray(array, dtype=np.float32)
        if node_features.ndim != 2:
            raise ValueError(
                "'node_features' must have shape [num_nodes, num_features]; "
                f"got shape {node_features.shape}."
            )
        if node_features.shape[0] < 1:
            raise ValueError("Graph must contain at least one node. 'node_features' is empty.")
        if node_features.shape[1] < 1:
            raise ValueError("Each node must have at least one feature value.")
        if not np.isfinite(node_features).all():
            raise ValueError("'node_features' contains NaN or Inf values.")
        return torch.from_numpy(node_features).to(torch.float32)

    def _to_edge_index(self, array: np.ndarray, num_nodes: int) -> torch.Tensor:
        """Convert edge connectivity to the required COO format."""

        edge_index = np.asarray(array)
        if edge_index.ndim != 2:
            raise ValueError(
                "'edge_index' must have shape [2, num_edges] or [num_edges, 2]; "
                f"got shape {edge_index.shape}."
            )

        if edge_index.shape[0] == 2 and edge_index.shape[1] >= 0:
            edge_index_2d = edge_index
        elif edge_index.shape[1] == 2:
            edge_index_2d = edge_index.T
        else:
            raise ValueError(
                "'edge_index' shape is invalid. Expected [2, num_edges] or [num_edges, 2]; "
                f"got {edge_index.shape}."
            )

        edge_index_tensor = torch.from_numpy(edge_index_2d.astype(np.int64, copy=False)).to(torch.long)
        if edge_index_tensor.shape[0] != 2:
            raise ValueError(
                "'edge_index' must be arranged with two rows for source/destination node ids; "
                f"got shape {tuple(edge_index_tensor.shape)}."
            )
        if edge_index_tensor.numel() > 0:
            if edge_index_tensor.min().item() < 0:
                raise ValueError("'edge_index' contains negative node indices.")
            if edge_index_tensor.max().item() >= num_nodes:
                raise ValueError(
                    "'edge_index' contains node indices outside the valid range for the graph: "
                    f"expected [0, {num_nodes - 1}] but found max index {edge_index_tensor.max().item()}."
                )
        check_no_nan_or_inf(edge_index_tensor, names=["edge_index"])
        return edge_index_tensor

    def _to_edge_attr(self, array: np.ndarray, num_edges: int) -> torch.Tensor:
        """Convert optional edge feature data into a float32 tensor."""

        edge_attr = np.asarray(array, dtype=np.float32)
        if edge_attr.ndim != 2:
            raise ValueError(
                "'edge_attr' must have shape [num_edges, num_edge_features]; "
                f"got shape {edge_attr.shape}."
            )
        if edge_attr.shape[0] not in (0, num_edges):
            raise ValueError(
                "'edge_attr' must describe every edge in the graph; "
                f"expected {num_edges} rows but got {edge_attr.shape[0]}."
            )
        if edge_attr.shape[1] < 1:
            raise ValueError("'edge_attr' must contain at least one feature per edge.")
        if not np.isfinite(edge_attr).all():
            raise ValueError("'edge_attr' contains NaN or Inf values.")
        return torch.from_numpy(edge_attr).to(torch.float32)

    def _extract_metadata(self, archive: np.lib.npyio.NpzFile) -> Optional[GraphMetadata]:
        """Read graph metadata from the archive and keep it separate from model inputs."""

        metadata_keys = ("graph_id", "hypothesis_id", "timestamp", "is_ground_truth")
        present_keys = [key for key in metadata_keys if key in archive]
        if not present_keys:
            return None

        graph_id = archive["graph_id"] if "graph_id" in archive else "unknown_graph"
        hypothesis_id = archive["hypothesis_id"] if "hypothesis_id" in archive else None
        timestamp = archive["timestamp"] if "timestamp" in archive else None
        is_ground_truth = archive["is_ground_truth"] if "is_ground_truth" in archive else False

        graph_id_value = _safe_metadata_value(graph_id)
        if graph_id_value is None:
            graph_id_value = "unknown_graph"
        graph_id_value = str(graph_id_value)

        hypothesis_value = _safe_metadata_value(hypothesis_id) if hypothesis_id is not None else None
        if hypothesis_value is not None:
            hypothesis_value = str(hypothesis_value)

        if timestamp is not None:
            timestamp_value = _safe_metadata_value(timestamp)
            if isinstance(timestamp_value, (np.generic,)):
                timestamp_value = timestamp_value.item()
        else:
            timestamp_value = None

        if is_ground_truth is not None:
            is_ground_truth_value = _safe_metadata_value(is_ground_truth)
            try:
                is_ground_truth_value = _as_bool(is_ground_truth_value)
            except ValueError:
                is_ground_truth_value = bool(str(is_ground_truth_value).lower() in {"true", "1", "yes", "y"})
        else:
            is_ground_truth_value = False

        return GraphMetadata(
            graph_id=graph_id_value,
            hypothesis_id=hypothesis_value,
            timestamp=timestamp_value,
            is_ground_truth=is_ground_truth_value,
        )

    def normalize_graph(self, data: Data) -> Data:
        """Apply optional feature normalization.

        This method is intentionally opt-in. Normalization is not applied during the
        default loader path unless ``normalize=True`` is set in the loader.
        """

        if getattr(data, "x", None) is None:
            raise ValueError("Cannot normalize a graph without an 'x' tensor.")

        x = data.x
        if x.numel() == 0:
            return data

        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))
        data.x = (x - mean) / std
        return data


def normalize_graph(data: Data) -> Data:
    """Normalize node features for a graph instance.

    This is a convenience function used by ``GraphLoader`` when normalization is
    enabled. It is intentionally not invoked by default to preserve raw dataset
    statistics.

    Args:
        data: A graph object with a valid ``x`` tensor.

    Returns:
        The same graph instance with normalized node features.
    """

    loader = GraphLoader(normalize=False)
    return loader.normalize_graph(data)


def load_graph(path: str | Path) -> GraphData:
    """Load a single graph from a ``.npz`` file and return a validated PyG graph.

    Args:
        path: Path to the serialized graph file.

    Returns:
        A validated ``GraphData`` instance.
    """

    return GraphLoader().load(path)


def load_graphs_from_directory(
    directory: str | Path,
    pattern: str = "*.npz",
    recursive: bool = False,
    normalize: bool = False,
) -> list[GraphData]:
    """Load all graph files from a directory in deterministic order.

    Args:
        directory: Directory containing graph files.
        pattern: Glob pattern used to identify files.
        recursive: Whether to search recursively.
        normalize: If ``True``, apply optional feature normalization after loading.

    Returns:
        A list of validated graph objects.
    """

    loader = GraphLoader(normalize=normalize)
    return loader.load_directory(directory, pattern=pattern, recursive=recursive)


__all__ = [
    "GraphLoader",
    "normalize_graph",
    "load_graph",
    "load_graphs_from_directory",
]
