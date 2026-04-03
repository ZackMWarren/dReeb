import numpy as np
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import warnings


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


def compute_diffusion_filter(W, diffusion_eigen_index=1, precision=1.0):
    """
    Compute a per-component diffusion eigenfunction filter.

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

    Returns
    -------
    filter_values : np.ndarray, shape (N,)
        Normalized filter values in [0, 1].
    components : list of np.ndarray
        Connected components of the affinity graph.
    eigenvalues : np.ndarray, shape (n_components,)
        Selected nontrivial diffusion eigenvalue per component, or 1.0 for
        singleton components.
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

    N = W.shape[0]
    filter_values = np.zeros(N, dtype=float)
    components = _connected_components_from_affinity(W)
    eigenvalues = np.full(len(components), np.nan, dtype=float)

    for ci, nodes in enumerate(components):
        nodes = np.asarray(nodes, dtype=int)
        if nodes.size == 0:
            continue
        if nodes.size == 1:
            filter_values[nodes] = 0.0
            eigenvalues[ci] = 1.0
            continue

        if diffusion_eigen_index > nodes.size - 1:
            raise ValueError(
                "diffusion_eigen_index={} is too large for connected component {} "
                "with {} points; maximum nontrivial index is {}.".format(
                    diffusion_eigen_index, ci, nodes.size, nodes.size - 1
                )
            )

        W_sub = W[nodes, :][:, nodes].tocsr()
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
            eigenvalues[ci] = 1.0
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
        eigenvalues[ci] = float(eigvals[eig_idx])

    filter_values = _quantize_filter_values(filter_values, precision)
    return filter_values, components, eigenvalues
