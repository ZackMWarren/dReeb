import numpy as np

from dreeb import dreeb, extract_cellular_trajectory


def sample_line(n=41):
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, np.zeros_like(x)])


def test_simplified_node_and_edge_assignments_cover_dataset():
    X = sample_line()
    result = dreeb(
        X,
        k=8,
        simplify=True,
        return_point_assignment=True,
        return_edge_assignment=True,
        return_cellular_decomposition=True,
    )

    simplified = result["simplified"]
    num_points = X.shape[0]

    assert simplified["point_assignment"].shape == (num_points,)
    assert simplified["point_edge_assignment"].shape == (num_points,)
    assert len(simplified["node_points"]) == len(simplified["nodes"])
    assert len(simplified["node_support_points"]) == len(simplified["nodes"])
    assert len(simplified["edge_points"]) == len(simplified["edges"])
    assert len(simplified["edge_support_points"]) == len(simplified["edges"])
    assert len(simplified["point_edges"]) == num_points
    assert simplified["cell_assignment_kind"].shape == (num_points,)
    assert simplified["cell_assignment_id"].shape == (num_points,)
    assert len(simplified["node_cells"]) == len(simplified["nodes"])
    assert len(simplified["edge_cells"]) == len(simplified["edges"])

    node_owned = np.concatenate(simplified["node_points"])
    edge_owned = np.concatenate(simplified["edge_points"])

    assert np.array_equal(np.sort(node_owned), np.arange(num_points))
    assert np.array_equal(np.sort(edge_owned), np.arange(num_points))
    cell_owned = np.concatenate(simplified["node_cells"] + simplified["edge_cells"])
    assert np.array_equal(np.sort(cell_owned), np.arange(num_points))

    extracted = extract_cellular_trajectory(simplified, [0, len(simplified["nodes"])])
    expected = np.unique(
        np.concatenate([simplified["node_cells"][0], simplified["edge_cells"][0]])
    )
    assert np.array_equal(extracted, expected)
    assert np.array_equal(
        np.sort(simplified["point_assignment"]),
        np.repeat(np.arange(len(simplified["nodes"])), [len(x) for x in simplified["node_points"]]),
    )
    assert np.array_equal(
        np.sort(simplified["point_edge_assignment"]),
        np.repeat(np.arange(len(simplified["edges"])), [len(x) for x in simplified["edge_points"]]),
    )
