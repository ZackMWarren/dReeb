import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import dijkstra
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import warnings

from .graph import build_diffusion_operator


def _quantize_filter_values(filter_values, precision):
    precision = float(np.clip(precision, 0.0, 1.0))
    if precision >= 0.999:
        return filter_values

    finite_mask = np.isfinite(filter_values)
    f = filter_values.copy()
    if not np.all(finite_mask):
        f[~finite_mask] = float(np.min(f[finite_mask]))
    alphas, inv = np.unique(f, return_inverse=True)
    K_raw = alphas.size
    if K_raw <= 2:
        return filter_values

    K_eff = max(2, int(np.ceil(precision * K_raw)))
    coarse_of_level = np.floor(np.arange(K_raw) * (K_eff / K_raw)).astype(np.int32)
    coarse_of_level = np.clip(coarse_of_level, 0, K_eff - 1)
    return coarse_of_level[inv].astype(float)


def _connected_components_from_affinity(W):
    if not sp.isspmatrix(W):
        W = sp.csr_matrix(W)
    A = ((W + W.T) > 0).astype(np.int8).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    ncomp, comp_labels = connected_components(A, directed=False)
    components = [np.where(comp_labels == c)[0] for c in range(ncomp)]
    return components


def _compute_von_neumann_entropy(data, t_max=100):
    singular_values = np.linalg.svd(np.asarray(data, dtype=float), compute_uv=False)
    entropy = []
    singular_values_t = np.copy(singular_values)

    for _ in range(t_max):
        prob = singular_values_t / np.sum(singular_values_t)
        prob = prob + np.finfo(float).eps
        entropy.append(-np.sum(prob * np.log(prob)))
        singular_values_t = singular_values_t * singular_values

    return np.asarray(entropy, dtype=float)


def _find_knee_point(y, x=None):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")
    if y.shape[0] < 3:
        raise ValueError("Cannot find knee point on vector of length < 3")

    if x is None:
        x = np.arange(y.shape[0], dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    n = np.arange(2, y.shape[0] + 1).astype(np.float32)

    sigma_xy = np.cumsum(x * y)[1:]
    sigma_x = np.cumsum(x)[1:]
    sigma_y = np.cumsum(y)[1:]
    sigma_xx = np.cumsum(x * x)[1:]
    det = n * sigma_xx - sigma_x * sigma_x
    mfwd = (n * sigma_xy - sigma_x * sigma_y) / det
    bfwd = -(sigma_x * sigma_xy - sigma_xx * sigma_y) / det

    sigma_xy = np.cumsum(x[::-1] * y[::-1])[1:]
    sigma_x = np.cumsum(x[::-1])[1:]
    sigma_y = np.cumsum(y[::-1])[1:]
    sigma_xx = np.cumsum(x[::-1] * x[::-1])[1:]
    det = n * sigma_xx - sigma_x * sigma_x
    mbck = ((n * sigma_xy - sigma_x * sigma_y) / det)[::-1]
    bbck = (-(sigma_x * sigma_xy - sigma_xx * sigma_y) / det)[::-1]

    error_curve = np.full_like(y, np.nan, dtype=float)
    for breakpt in np.arange(1, y.shape[0] - 1):
        delsfwd = (mfwd[breakpt - 1] * x[: breakpt + 1] + bfwd[breakpt - 1]) - y[: breakpt + 1]
        delsbck = (mbck[breakpt - 1] * x[breakpt:] + bbck[breakpt - 1]) - y[breakpt:]
        error_curve[breakpt] = np.sum(np.abs(delsfwd)) + np.sum(np.abs(delsbck))

    loc = int(np.argmin(error_curve[1:-1]) + 1)
    return int(x[loc])


def _select_diffusion_time(diff_op, diffusion_time, diffusion_time_max):
    if diffusion_time == "auto":
        ts = np.arange(int(diffusion_time_max))
        entropy = _compute_von_neumann_entropy(diff_op, t_max=int(diffusion_time_max))
        return _find_knee_point(y=entropy, x=ts)

    diffusion_time = int(diffusion_time)
    if diffusion_time < 0:
        raise ValueError("diffusion_time must be >= 0 or 'auto'.")
    return diffusion_time


def _normalize_zero_one(values):
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _build_diffused_cost_graph(W_sub, diff_op_t, eps=1e-7):
    A = ((W_sub + W_sub.T) > 0).astype(np.int8).tocsr()
    rows, cols = A.nonzero()
    probs = 0.5 * (diff_op_t[rows, cols] + diff_op_t[cols, rows])
    costs = -np.log(np.clip(probs, eps, None))
    return sp.csr_matrix((costs, (rows, cols)), shape=A.shape).tocsr()


def _choose_two_sweep_root(cost_graph, seed=0):
    dist0 = dijkstra(cost_graph, directed=False, indices=int(seed))
    finite0 = np.flatnonzero(np.isfinite(dist0))
    if finite0.size == 0:
        return int(seed), (int(seed), int(seed))
    a = int(finite0[np.argmax(dist0[finite0])])

    dista = dijkstra(cost_graph, directed=False, indices=a)
    finitea = np.flatnonzero(np.isfinite(dista))
    if finitea.size == 0:
        return a, (a, a)
    b = int(finitea[np.argmax(dista[finitea])])
    return a, (a, b)


def compute_diffusion_filter(
    W,
    diffusion_eigen_index=1,
    precision=1.0,
    filter_method="diffusion_eigenfunction",
    diffusion_time="auto",
    diffusion_time_max=40,
    return_metadata=False,
):
    """
    Compute a per-component diffusion-based scalar filter.

    Parameters
    ----------
    W : scipy.sparse matrix, shape (n_samples, n_samples)
        Affinity matrix.
    diffusion_eigen_index : int, optional (default: 1)
        One-based index of the nontrivial diffusion eigenfunction to use
        within each connected component. `1` means the first nontrivial
        eigenfunction, `2` means the second, and so on.
    precision : float, optional (default: 1.0)
        Controls quantization. 1.0 = no quantization, 0.1 = 10% of unique levels.
    filter_method : {"diffusion_eigenfunction", "rooted_potential_distance"}, optional
        Scalar filter construction method.
    diffusion_time : {"auto"} or int, optional (default: "auto")
        Diffusion time used by `rooted_potential_distance`. Ignored for
        `diffusion_eigenfunction`.
    diffusion_time_max : int, optional (default: 40)
        Maximum time tested by the PHATE-style entropy knee heuristic when
        `diffusion_time="auto"`.
    return_metadata : bool, optional (default: False)
        If True, also return a metadata dictionary describing the selected
        per-component parameters.

    Returns
    -------
    filter_values : np.ndarray, shape (N,)
        Normalized filter values in [0, 1].
    components : list of np.ndarray
        Connected components of the affinity graph.
    aux_values : np.ndarray, shape (n_components,)
        Per-component auxiliary values. For `diffusion_eigenfunction`, these
        are the selected diffusion eigenvalues. For
        `rooted_potential_distance`, these are the selected diffusion times.
    metadata : dict, optional
        Returned only when `return_metadata=True`.
    """
    if not sp.isspmatrix(W):
        W = sp.csr_matrix(W)
    W = W.tocsr().astype(float)
    W = W.maximum(W.T)
    W.setdiag(0)
    W.eliminate_zeros()

    diffusion_eigen_index = int(diffusion_eigen_index)
    if diffusion_eigen_index < 1:
        raise ValueError("diffusion_eigen_index must be >= 1.")
    if filter_method not in ("diffusion_eigenfunction", "rooted_potential_distance"):
        raise ValueError(
            "filter_method must be 'diffusion_eigenfunction' or "
            "'rooted_potential_distance'."
        )

    N = W.shape[0]
    filter_values = np.zeros(N, dtype=float)
    components = _connected_components_from_affinity(W)
    aux_values = np.full(len(components), np.nan, dtype=float)
    metadata = {
        "filter_method": filter_method,
        "component_root_indices": np.full(len(components), -1, dtype=int),
        "component_root_pairs": np.full((len(components), 2), -1, dtype=int),
    }
    if filter_method == "diffusion_eigenfunction":
        metadata["component_eigenvalues"] = aux_values
    else:
        metadata["component_diffusion_times"] = aux_values

    for ci, nodes in enumerate(components):
        nodes = np.asarray(nodes, dtype=int)
        if nodes.size == 0:
            continue
        if nodes.size == 1:
            filter_values[nodes] = 0.0
            if filter_method == "diffusion_eigenfunction":
                aux_values[ci] = 1.0
            else:
                aux_values[ci] = 0.0
                metadata["component_root_indices"][ci] = int(nodes[0])
                metadata["component_root_pairs"][ci] = (int(nodes[0]), int(nodes[0]))
            continue

        W_sub = W[nodes, :][:, nodes].tocsr()

        if filter_method == "rooted_potential_distance":
            diff_op = np.asarray(build_diffusion_operator(W_sub).toarray(), dtype=float)
            t = _select_diffusion_time(
                diff_op,
                diffusion_time=diffusion_time,
                diffusion_time_max=diffusion_time_max,
            )
            diff_op_t = np.linalg.matrix_power(diff_op, int(t))
            potential = -np.log(diff_op_t + 1e-7)
            cost_graph = _build_diffused_cost_graph(W_sub, diff_op_t)
            root_local, root_pair_local = _choose_two_sweep_root(cost_graph)

            rooted = np.linalg.norm(potential - potential[int(root_local)], axis=1)
            filter_values[nodes] = _normalize_zero_one(rooted)
            aux_values[ci] = float(t)
            metadata["component_root_indices"][ci] = int(nodes[int(root_local)])
            metadata["component_root_pairs"][ci] = (
                int(nodes[int(root_pair_local[0])]),
                int(nodes[int(root_pair_local[1])]),
            )
            continue

        if diffusion_eigen_index > nodes.size - 1:
            raise ValueError(
                "diffusion_eigen_index={} is too large for connected component {} "
                "with {} points; maximum nontrivial index is {}.".format(
                    diffusion_eigen_index, ci, nodes.size, nodes.size - 1
                )
            )

        degrees = np.asarray(W_sub.sum(axis=1)).ravel()
        safe_degrees = np.maximum(degrees, np.finfo(float).eps)
        inv_sqrt_deg = 1.0 / np.sqrt(safe_degrees)
        D_inv_sqrt = sp.diags(inv_sqrt_deg)
        S = (D_inv_sqrt @ W_sub @ D_inv_sqrt).astype(float).tocsr()

        target_rank = min(nodes.size, diffusion_eigen_index + 1)
        if nodes.size <= 8:
            eigvals, eigvecs = np.linalg.eigh(S.toarray())
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
        else:
            k = min(nodes.size - 1, max(2, target_rank))
            eigvals, eigvecs = eigsh(S, k=k, which="LA")
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

        tol = 1e-8
        nontrivial = np.where(np.abs(eigvals - 1.0) > tol)[0]
        if nontrivial.size == 0:
            warnings.warn(
                "Connected component {} has no nontrivial diffusion eigenfunction; "
                "using a constant filter on that component.".format(ci),
                RuntimeWarning,
            )
            filter_values[nodes] = 0.0
            aux_values[ci] = 1.0
            continue

        if diffusion_eigen_index > nontrivial.size:
            raise ValueError(
                "diffusion_eigen_index={} exceeds the number of nontrivial diffusion "
                "eigenfunctions ({}) available in connected component {}.".format(
                    diffusion_eigen_index, nontrivial.size, ci
                )
            )

        eig_idx = int(nontrivial[diffusion_eigen_index - 1])
        phi = np.asarray(eigvecs[:, eig_idx], dtype=float)
        psi = inv_sqrt_deg * phi
        psi_min = float(np.min(psi))
        psi_max = float(np.max(psi))

        if np.isclose(psi_min, psi_max):
            filter_values[nodes] = 0.0
        else:
            filter_values[nodes] = (psi - psi_min) / (psi_max - psi_min)
        aux_values[ci] = float(eigvals[eig_idx])

    filter_values = _quantize_filter_values(filter_values, precision)
    if return_metadata:
        return filter_values, components, aux_values, metadata
    return filter_values, components, aux_values
