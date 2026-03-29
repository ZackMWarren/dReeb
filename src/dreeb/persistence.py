import numpy as np


def compute_edge_lengths(nodes, edges, t_vals):
    """
    Compute filter-space edge lengths for a graph whose nodes carry `step`.

    Parameters
    ----------
    nodes : list of dict
        Graph nodes. Each node dict must have key `step`.
    edges : list of tuple
        Edge list as (u, v) pairs in the indexing of `nodes`.
    t_vals : np.ndarray
        Midpoint filter values from prepare_reeb.

    Returns
    -------
    list of float
        One filter-space length per edge.
    """
    edge_lengths = []
    if not edges:
        return edge_lengths

    node_steps = np.array([node["step"] for node in nodes], dtype=int)
    for u, v in edges:
        su = node_steps[int(u)]
        sv = node_steps[int(v)]
        edge_lengths.append(abs(float(t_vals[su]) - float(t_vals[sv])))

    return edge_lengths


def compute_graph_persistence(num_nodes, edges, edge_lengths):
    """
    Compute a graph filtration persistence summary from edge lengths.

    The filtration adds all vertices at time 0 and then adds edges in
    nondecreasing order of `edge_lengths`. In this filtration:

    - H0 classes are born at 0 and die when an edge merges components.
    - H1 classes are born when an edge closes a cycle and are essential
      (death = inf) because the filtration contains no 2-cells.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list of tuple
        Edge list as (u, v) pairs.
    edge_lengths : sequence of float
        One edge length per edge.

    Returns
    -------
    dict
        Dictionary with:
        - h0 : np.ndarray, shape (m, 2), finite H0 pairs [birth, death]
        - h0_essential : np.ndarray, shape (c, 2), essential H0 pairs
        - h1 : np.ndarray, shape (k, 2), essential H1 pairs [birth, inf]
    """
    if len(edges) != len(edge_lengths):
        raise ValueError("edges and edge_lengths must have the same length.")

    parent = np.arange(num_nodes, dtype=int)
    size = np.ones(num_nodes, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    h0 = []
    h1 = []

    order = np.argsort(np.asarray(edge_lengths, dtype=float), kind="stable")

    for idx in order:
        u, v = edges[int(idx)]
        w = float(edge_lengths[int(idx)])
        u = int(u)
        v = int(v)

        if u == v:
            h1.append((w, np.inf))
            continue

        ru = find(u)
        rv = find(v)

        if ru != rv:
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]
            h0.append((0.0, w))
        else:
            h1.append((w, np.inf))

    roots = {find(i) for i in range(num_nodes)}
    h0_essential = np.array([(0.0, np.inf) for _ in roots], dtype=float)

    return {
        "h0": np.array(h0, dtype=float).reshape(-1, 2),
        "h0_essential": h0_essential.reshape(-1, 2),
        "h1": np.array(h1, dtype=float).reshape(-1, 2),
    }
