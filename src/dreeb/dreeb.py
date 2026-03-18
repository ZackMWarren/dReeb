from .graph import build_affinity_matrix, build_diffusion_operator
from .filter import find_endpoints, compute_filter
from .reeb import prepare_reeb, build_reeb_graph
from .simplify import simplify_reeb_graph


def dreeb(X, k=80, precision=1.0):
    """
    Run the full dReeb pipeline.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input point cloud data
    k : int, optional (default: 80)
        Number of nearest neighbors
    precision : float, optional (default: 1.0)
        Controls filter quantization. 1.0 = no quantization,
        0.1 = 10% of unique levels

    Returns
    -------
    dict with keys:
        W              : affinity matrix
        P              : diffusion operator
        components     : list of node index arrays per connected component
        roots_A        : first endpoint per component
        roots_B        : second endpoint per component
        diameters      : geodesic diameter per component
        filter_values  : normalized filter function f in [0, 1]
        distA          : geodesic distances to endpoint A
        distB          : geodesic distances to endpoint B
        prep_state     : prepared state for Reeb construction
        reeb_nodes     : raw Reeb graph nodes
        reeb_edges     : raw Reeb graph edges
        step_vertices  : per-slice vertex indices
        step_comp_ids  : per-slice component IDs
        simp_edges     : simplified edge list
        keep_ids       : kept node indices after simplification
        beta1          : first Betti number (number of cycles)
        comp_count     : number of connected components
    """
    W = build_affinity_matrix(X, k=k)
    P = build_diffusion_operator(W)
    components, roots_A, roots_B, cost, diameters = find_endpoints(P)
    filter_values, distA, distB = compute_filter(
        P, cost, components, roots_A, roots_B, precision=precision
    )
    prep_state = prepare_reeb(W, filter_values)
    reeb_nodes, reeb_edges, step_vertices, step_comp_ids = build_reeb_graph(prep_state)
    simp_edges, keep_ids, beta1, comp_count = simplify_reeb_graph(reeb_nodes, reeb_edges)

    return {
        "W"             : W,
        "P"             : P,
        "components"    : components,
        "roots_A"       : roots_A,
        "roots_B"       : roots_B,
        "diameters"     : diameters,
        "filter_values" : filter_values,
        "distA"         : distA,
        "distB"         : distB,
        "prep_state"    : prep_state,
        "reeb_nodes"    : reeb_nodes,
        "reeb_edges"    : reeb_edges,
        "step_vertices" : step_vertices,
        "step_comp_ids" : step_comp_ids,
        "simp_edges"    : simp_edges,
        "keep_ids"      : keep_ids,
        "beta1"         : beta1,
        "comp_count"    : comp_count,
    }