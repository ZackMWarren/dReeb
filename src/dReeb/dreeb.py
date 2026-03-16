from process_data import build_diffusion_operator
from process_data import build_affinity_matrix
from filter import find_endpoints
from filter import compute_filter

def dreeb(X, k=80):
    """
    Does something
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input point cloud data
    k : int, optional (default: 80)
        Number of nearest neighbors for affinity matrix
    """

    W = build_affinity_matrix(X, k=k)
    P = build_diffusion_operator(W)
    # diameters is not used in filter but could be useful for analysis
    components, roots_A, roots_B, cost, diameters = find_endpoints(P)
    filter_values, distA_global, distB_global = compute_filter(P, cost, components, roots_A, roots_B)
    
    
    