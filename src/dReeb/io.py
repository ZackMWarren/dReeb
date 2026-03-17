import numpy as np
import warnings

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


def save_reeb(
    filepath,
    simp_edges,
    node_pos,
):
    """
    Save simplified Reeb graph to disk as a .npz file.

    Parameters
    ----------
    filepath : str
        Output file path. Should end in .npz
    simp_edges : list of tuple
        Simplified edge list from simplify_reeb_graph
    node_pos : np.ndarray, shape (num_nodes, D)
        Node positions in embedding space from plot_reeb

    Returns
    -------
    summary : dict
        Summary statistics of the saved graph:
        num_nodes, num_edges, num_components, beta1, embedding_dim
    """
    node_pos = np.asarray(node_pos, dtype=float)
    num_nodes = node_pos.shape[0]
    edges_array = np.asarray(simp_edges, dtype=int) if len(simp_edges) > 0 \
                  else np.empty((0, 2), dtype=int)
    num_edges = len(simp_edges)

    # compute components and beta1
    if _NX_AVAILABLE:
        G_simple = nx.Graph()
        G_simple.add_nodes_from(range(num_nodes))
        G_simple.add_edges_from(simp_edges)
        num_components = nx.number_connected_components(G_simple)
    else:
        warnings.warn(
            "networkx not installed — computing components via scipy instead.",
            RuntimeWarning
        )
        # fallback using scipy
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        if num_edges > 0:
            rows = edges_array[:, 0]
            cols = edges_array[:, 1]
            A = sp.csr_matrix(
                (np.ones(num_edges), (rows, cols)),
                shape=(num_nodes, num_nodes)
            )
            A = (A + A.T) > 0
            num_components, _ = connected_components(A, directed=False)
        else:
            num_components = num_nodes

    beta1 = num_edges - num_nodes + num_components

    out = {
        "node_positions" : node_pos,
        "edges"          : edges_array,
        "num_nodes"      : num_nodes,
        "num_edges"      : num_edges,
        "num_components" : num_components,
        "beta1"          : beta1,
        "embedding_dim"  : node_pos.shape[1],
    }

    np.savez(filepath, **out)

    summary = {
        "num_nodes"     : num_nodes,
        "num_edges"     : num_edges,
        "num_components": num_components,
        "beta1"         : beta1,
        "embedding_dim" : node_pos.shape[1],
    }

    return summary


def load_reeb(filepath):
    """
    Load a saved Reeb graph from a .npz file.

    Parameters
    ----------
    filepath : str
        Path to .npz file saved by save_reeb

    Returns
    -------
    data : dict with keys:
        node_positions : np.ndarray, shape (num_nodes, D)
        edges          : np.ndarray, shape (num_edges, 2)
        num_nodes      : int
        num_edges      : int
        num_components : int
        beta1          : int
        embedding_dim  : int
    """
    raw = np.load(filepath, allow_pickle=False)
    return {
        "node_positions" : raw["node_positions"],
        "edges"          : raw["edges"],
        "num_nodes"      : int(raw["num_nodes"]),
        "num_edges"      : int(raw["num_edges"]),
        "num_components" : int(raw["num_components"]),
        "beta1"          : int(raw["beta1"]),
        "embedding_dim"  : int(raw["embedding_dim"]),
    }