"""Inference utility for selecting the best SLAM hypothesis from five candidates.

This module performs pure PyTorch inference: it evaluates each candidate hypothesis
against a reference graph using a trained Siamese GNN and returns the highest-scoring
selection. It is intentionally free of ROS2-specific code so it can be integrated into
an SGMH-SLAM node later without coupling the model logic to any middleware.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

__all__ = ["select_best_hypothesis"]


def _resolve_device(model: torch.nn.Module) -> torch.device:
    """Return the device on which the model currently resides."""
    if hasattr(model, "device") and model.device is not None:
        return torch.device(model.device)

    parameters = list(model.parameters())
    if parameters:
        return parameters[0].device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _graph_id(graph: Any, fallback: Any) -> Any:
    """Best-effort extraction of a graph identifier for reporting."""
    for attr_name in ("hypothesis_id", "graph_id", "id", "name"):
        value = getattr(graph, attr_name, None)
        if value is not None:
            return value
    return fallback


def select_best_hypothesis(
    model: torch.nn.Module,
    hypotheses: Sequence[Any],
    ground_truth: Any,
) -> dict[str, Any]:
    """Select the best hypothesis graph from exactly five candidates.

    Args:
        model: A trained Siamese GNN that accepts ``(hypothesis_graph, ground_truth_graph)``.
        hypotheses: A sequence containing exactly five candidate hypothesis graphs.
        ground_truth: The reference graph against which each hypothesis is scored.

    Returns:
        A dictionary with the selected index, selected hypothesis identifier, descending
        scores, and ranking information:

        {
            "selected_index": int,
            "selected_hypothesis_id": Any,
            "scores": list[float],
            "ranking": list[int],
        }

    Raises:
        ValueError: If the hypothesis list does not contain exactly five graphs.
    """
    if model is None:
        raise ValueError("model must be provided.")
    if ground_truth is None:
        raise ValueError("ground_truth graph must be provided.")
    if hypotheses is None:
        raise ValueError("hypotheses must be provided.")

    hypothesis_list = list(hypotheses)
    if len(hypothesis_list) != 5:
        raise ValueError(f"Expected exactly 5 hypothesis graphs, got {len(hypothesis_list)}.")

    device = _resolve_device(model)
    model = model.to(device)
    model.eval()

    ground_truth_graph = ground_truth.to(device) if hasattr(ground_truth, "to") else ground_truth

    scores: list[float] = []

    with torch.no_grad():
        for idx, hypothesis in enumerate(hypothesis_list):
            if hypothesis is None:
                raise ValueError(f"Hypothesis at index {idx} is None.")

            hypothesis_graph = hypothesis.to(device) if hasattr(hypothesis, "to") else hypothesis
            score = model(hypothesis_graph, ground_truth_graph)
            score_tensor = torch.as_tensor(score, device=device)
            if score_tensor.numel() == 0:
                raise ValueError(f"Model produced an empty score for hypothesis {idx}.")

            score_tensor = score_tensor.reshape(-1)
            if score_tensor.numel() != 1:
                raise ValueError(
                    f"Model output for hypothesis {idx} must be a single scalar score, "
                    f"got shape {tuple(score_tensor.shape)}."
                )

            scores.append(float(score_tensor.detach().cpu().item()))

    selected_index = int(max(range(len(scores)), key=lambda idx: scores[idx]))
    ranking = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    sorted_scores = [float(scores[idx]) for idx in ranking]
    selected_hypothesis_id = _graph_id(hypothesis_list[selected_index], selected_index)

    return {
        "selected_index": selected_index,
        "selected_hypothesis_id": selected_hypothesis_id,
        "scores": sorted_scores,
        "ranking": ranking,
    }
