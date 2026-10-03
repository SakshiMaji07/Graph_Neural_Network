import torch
from torch_geometric.data import Batch, Data

from src.models.siamese_gnn import SiameseGNN


def make_graph(num_nodes: int, num_edges: int, feature_dim: int = 4, seed: int = 0) -> Data:
    """Create a small synthetic graph with a deterministic feature matrix."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, feature_dim, generator=generator)

    edges = []
    for i in range(num_nodes - 1):
        edges.extend([[i, i + 1], [i + 1, i]])

    if num_nodes >= 3:
        edges.extend([[0, 2], [2, 0]])

    if num_edges > len(edges) // 2:
        extra_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if len(extra_edges) >= num_edges - len(edges) // 2:
                    break
                extra_edges.append([i, j])
                extra_edges.append([j, i])
            if len(extra_edges) >= num_edges - len(edges) // 2:
                break
        edges.extend(extra_edges)

    if len(edges) < 2 * num_edges:
        raise ValueError(f"Could not build {num_edges} edges for {num_nodes} nodes.")

    edge_index = torch.tensor(edges[: 2 * num_edges], dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def test_siamese_forward_works_with_different_node_counts():
    model = SiameseGNN(
        input_dim=4,
        hidden_dim=8,
        embedding_dim=6,
        num_layers=2,
        dropout=0.0,
        activation="relu",
        conv_type="gcn",
    )

    graph_h = make_graph(num_nodes=4, num_edges=5, feature_dim=4, seed=1)
    graph_gt = make_graph(num_nodes=7, num_edges=8, feature_dim=4, seed=2)

    output = model(graph_h, graph_gt)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1,)


def test_siamese_similarity_output_is_in_01_range():
    model = SiameseGNN(
        input_dim=4,
        hidden_dim=8,
        embedding_dim=6,
        num_layers=2,
        dropout=0.0,
        activation="relu",
        conv_type="gcn",
    )

    graph_h = make_graph(num_nodes=3, num_edges=4, feature_dim=4, seed=3)
    graph_gt = make_graph(num_nodes=5, num_edges=6, feature_dim=4, seed=4)

    output = model(graph_h, graph_gt)

    assert torch.all(output >= 0)
    assert torch.all(output <= 1)


def test_siamese_output_has_correct_batch_dimension():
    model = SiameseGNN(
        input_dim=4,
        hidden_dim=8,
        embedding_dim=6,
        num_layers=2,
        dropout=0.0,
        activation="relu",
        conv_type="gcn",
    )

    hypothesis_batch = Batch.from_data_list(
        [make_graph(3, 3, feature_dim=4, seed=5), make_graph(4, 5, feature_dim=4, seed=6)]
    )
    ground_truth_batch = Batch.from_data_list(
        [make_graph(4, 4, feature_dim=4, seed=7), make_graph(5, 6, feature_dim=4, seed=8)]
    )

    output = model(hypothesis_batch, ground_truth_batch)

    assert output.shape == (2,)


def test_siamese_encoder_parameters_are_shared_between_branches():
    model = SiameseGNN(
        input_dim=4,
        hidden_dim=8,
        embedding_dim=6,
        num_layers=2,
        dropout=0.0,
        activation="relu",
        conv_type="gcn",
    )

    state_names = set(model.state_dict().keys())
    assert any(name.startswith("encoder.") for name in state_names)
    assert not any(name.startswith("encoder_h") or name.startswith("encoder_gt") for name in state_names)

    graph_h = make_graph(num_nodes=4, num_edges=5, feature_dim=4, seed=9)
    graph_gt = make_graph(num_nodes=6, num_edges=7, feature_dim=4, seed=10)

    score = model(graph_h, graph_gt)

    assert score.shape == (1,)
    assert len(list(model.parameters())) > 0


def test_siamese_gradients_flow_through_both_graphs():
    model = SiameseGNN(
        input_dim=4,
        hidden_dim=8,
        embedding_dim=6,
        num_layers=2,
        dropout=0.0,
        activation="relu",
        conv_type="gcn",
    )

    graph_h = make_graph(num_nodes=4, num_edges=5, feature_dim=4, seed=11)
    graph_gt = make_graph(num_nodes=6, num_edges=7, feature_dim=4, seed=12)
    graph_h.x.requires_grad_(True)
    graph_gt.x.requires_grad_(True)

    loss = model(graph_h, graph_gt).sum()
    loss.backward()

    assert graph_h.x.grad is not None
    assert graph_gt.x.grad is not None
    assert torch.any(graph_h.x.grad != 0)
    assert torch.any(graph_gt.x.grad != 0)


def test_siamese_model_accepts_pyg_batch():
    model = SiameseGNN(
        input_dim=4,
        hidden_dim=8,
        embedding_dim=6,
        num_layers=2,
        dropout=0.0,
        activation="relu",
        conv_type="gcn",
    )

    batch_h = Batch.from_data_list(
        [make_graph(2, 2, feature_dim=4, seed=13), make_graph(5, 6, feature_dim=4, seed=14)]
    )
    batch_gt = Batch.from_data_list(
        [make_graph(3, 3, feature_dim=4, seed=15), make_graph(4, 5, feature_dim=4, seed=16)]
    )

    output = model(batch_h, batch_gt)

    assert output.shape == (2,)
    assert torch.all((output >= 0) & (output <= 1))


def test_siamese_backprop_produces_non_zero_gradients():
    model = SiameseGNN(
        input_dim=4,
        hidden_dim=8,
        embedding_dim=6,
        num_layers=2,
        dropout=0.0,
        activation="relu",
        conv_type="gcn",
    )

    graph_h = make_graph(num_nodes=5, num_edges=6, feature_dim=4, seed=17)
    graph_gt = make_graph(num_nodes=6, num_edges=7, feature_dim=4, seed=18)

    output = model(graph_h, graph_gt)
    loss = output.sum()
    loss.backward()

    encoder_grad = model.encoder.convs[0].lin.weight.grad
    mlp_grad = model.similarity_mlp.network[0].weight.grad

    assert encoder_grad is not None
    assert mlp_grad is not None
    assert torch.abs(encoder_grad).sum() > 0
    assert torch.abs(mlp_grad).sum() > 0


def test_siamese_model_handles_different_node_and_edge_counts_without_assuming_equality():
    model = SiameseGNN(
        input_dim=4,
        hidden_dim=8,
        embedding_dim=6,
        num_layers=2,
        dropout=0.0,
        activation="relu",
        conv_type="gcn",
    )

    graph_h = make_graph(num_nodes=2, num_edges=1, feature_dim=4, seed=19)
    graph_gt = make_graph(num_nodes=8, num_edges=12, feature_dim=4, seed=20)

    assert graph_h.x.shape[0] != graph_gt.x.shape[0]
    assert graph_h.edge_index.shape[1] != graph_gt.edge_index.shape[1]

    output = model(graph_h, graph_gt)

    assert output.shape == (1,)
    assert output.dtype.is_floating_point
    assert torch.all((output >= 0) & (output <= 1))
