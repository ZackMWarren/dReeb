import numpy as np
from scipy.sparse.csgraph import dijkstra, connected_components
import scipy.sparse as sp

def find_endpoints(P):
    """
    Find endpoints of each connected component using double-sweep
    Dijkstra in diffusion geometry.

    For each connected component, performs two Dijkstra sweeps to
    approximate the geodesic diameter and identify the two endpoints
    (roots_A and roots_B) that are farthest apart.

    Parameters
    ----------
    P : scipy.sparse matrix, shape (n_samples, n_samples)
        Diffusion operator (row-stochastic matrix from build_diffusion_operator)

    Returns
    -------
    components : list of np.ndarray
        List of arrays of node indices, one per connected component
    roots_A : np.ndarray, shape (n_components,)
        First endpoint (global index) of each component
    roots_B : np.ndarray, shape (n_components,)
        Second endpoint (global index) of each component
    diameters : np.ndarray, shape (n_components,)
        Approximate geodesic diameter of each component
    """
    if not sp.isspmatrix(P):
        P = sp.csr_matrix(P)
    P = P.astype(float)

    # --- 1) symmetrize P to get undirected affinity ---
    A = (P + P.T) / 2
    A = A.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    n = A.shape[0]

    # --- 2) cost graph: higher affinity = shorter edge ---
    eps = 1e-12
    cost = A.copy()
    cost.data = -np.log(cost.data + eps)

    # --- 3) connected components ---
    ncomp, comp_labels = connected_components(
        (A > 0).astype(int), directed=False
    )
    components = [np.where(comp_labels == c)[0] for c in range(ncomp)]

    roots_A   = np.full(ncomp, -1, dtype=int)
    roots_B   = np.full(ncomp, -1, dtype=int)
    diameters = np.zeros(ncomp, dtype=float)

    # --- 4) double-sweep dijkstra per component ---
    for ci, nodes in enumerate(components):
        if len(nodes) == 0:
            continue

        # restrict to subgraph
        cost_sub = cost[nodes, :][:, nodes]
        
        # pick an arbitrary seed in this component (index 0 in subspace)
        seed_sub = 0

        # 1st sweep: seed -> farthest point (A)
        dist_from_seed = dijkstra(
            csgraph=cost_sub,
            directed=False,
            indices=seed_sub,
            unweighted=False
        )

        if not np.any(np.isfinite(dist_from_seed)):
            roots_A[ci] = int(nodes[seed_sub])
            roots_B[ci] = int(nodes[seed_sub])
            diameters[ci] = 0.0
            continue

        A_sub = int(np.nanargmax(
            np.where(np.isfinite(dist_from_seed), dist_from_seed, np.nan)
        ))

        # 2nd sweep: A -> farthest point (B)
        dist_from_A = dijkstra(
            csgraph=cost_sub,
            directed=False,
            indices=A_sub,
            unweighted=False
        )

        if not np.any(np.isfinite(dist_from_A)):
            roots_A[ci] = int(nodes[A_sub])
            roots_B[ci] = int(nodes[A_sub])
            diameters[ci] = 0.0
            continue

        B_sub = int(np.nanargmax(
            np.where(np.isfinite(dist_from_A), dist_from_A, np.nan)
        ))

        roots_A[ci]   = int(nodes[A_sub])
        roots_B[ci]   = int(nodes[B_sub])
        diameters[ci] = float(dist_from_A[B_sub])

    return components, roots_A, roots_B, cost, diameters


def compute_filter(P, cost, components, roots_A, roots_B, precision=1.0):
    """
    Compute normalized diffusion geodesic filter function for each point.
    Step 
    
    Parameters
    ----------
    P : scipy.sparse matrix
        Diffusion operator
    cost : scipy.sparse matrix
        Cost graph (-log(P_sym))
    components : list of np.ndarray
        Connected components from find_endpoints
    roots_A : np.ndarray
        First endpoint per component
    roots_B : np.ndarray
        Second endpoint per component
    precision : float, optional (default: 1.0)
        Controls quantization. 1.0 = no quantization, 0.1 = 10% of unique levels
    
    Returns
    -------
    filter_values : np.ndarray, shape (N,)
        Normalized filter value in [0, 1] for each point
    distA_global : np.ndarray, shape (N,)
        Raw geodesic distance to endpoint A for each point
    distB_global : np.ndarray, shape (N,)
        Raw geodesic distance to endpoint B for each point
    """
    N = P.shape[0]
    eps = 1e-12

    distA_global = np.full(N, np.inf, dtype=float)
    distB_global = np.full(N, np.inf, dtype=float)
    filter_values = np.full(N, np.nan, dtype=float)

    # --- per component ---
    for ci, nodes in enumerate(components):
        if len(nodes) == 0:
            continue

        C_sub = cost[nodes, :][:, nodes]

        # map global root indices to local subgraph indices
        try:
            A_sub = int(np.where(nodes == roots_A[ci])[0][0])
        except IndexError:
            A_sub = 0
        try:
            B_sub = int(np.where(nodes == roots_B[ci])[0][0])
        except IndexError:
            B_sub = A_sub

        # dijkstra from both endpoints
        dA = np.asarray(dijkstra(C_sub, directed=False, indices=[A_sub],
                                 return_predecessors=False))[0]
        dB = np.asarray(dijkstra(C_sub, directed=False, indices=[B_sub],
                                 return_predecessors=False))[0]

        # replace any infinities within component with max finite value
        for arr in (dA, dB):
            finite = np.isfinite(arr)
            if not np.all(finite):
                max_finite = arr[finite].max() if finite.any() else 0.0
                arr[~finite] = max_finite

        distA_global[nodes] = dA
        distB_global[nodes] = dB

        # normalize distances to [0, 1]
        # uses dB as described in paper, but dA similarly works
        dmin, dmax = float(np.min(dB)), float(np.max(dB))
        filter_values[nodes] = (dB - dmin) / ((dmax - dmin) + eps)

    # --- optional quantization ---
    precision = float(np.clip(precision, 0.0, 1.0))
    if precision < 0.999:
        finite_mask = np.isfinite(filter_values)
        f = filter_values.copy()
        if not np.all(finite_mask):
            f[~finite_mask] = float(np.min(f[finite_mask]))
        alphas, inv = np.unique(f, return_inverse=True)
        K_raw = alphas.size
        if K_raw > 2:
            K_eff = max(2, int(np.ceil(precision * K_raw)))
            coarse_of_level = np.floor(np.arange(K_raw) * (K_eff / K_raw)).astype(np.int32)
            coarse_of_level = np.clip(coarse_of_level, 0, K_eff - 1)
            filter_values = coarse_of_level[inv].astype(float)

    return filter_values, distA_global, distB_global