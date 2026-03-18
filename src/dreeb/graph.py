import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra, connected_components
import warnings

#taken from graphtools from Krishnaswamy Lab
def _check_duplicates(distances, indices):
        if np.any(distances[:, 1] == 0):
            has_duplicates = distances[:, 1] == 0
            if np.sum(distances[:, 1:] == 0) < 20:
                idx = np.argwhere((distances == 0) & has_duplicates[:, None])
                duplicate_ids = np.array(
                    [
                        [indices[i[0], i[1]], i[0]]
                        for i in idx
                        if indices[i[0], i[1]] < i[0]
                    ]
                )
                duplicate_ids = duplicate_ids[np.argsort(duplicate_ids[:, 0])]
                duplicate_names = ", ".join(
                    ["{} and {}".format(i[0], i[1]) for i in duplicate_ids]
                )
                warnings.warn(
                    "Detected zero distance between samples {}. "
                    "Consider removing duplicates to avoid errors in "
                    "downstream processing.".format(duplicate_names),
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    "Detected zero distance between {} pairs of samples. "
                    "Consider removing duplicates to avoid errors in "
                    "downstream processing.".format(
                        np.sum(np.sum(distances[:, 1:] == 0)) // 2
                    ),
                    RuntimeWarning,
                )
            
#Taken from graphtools from Krishnaswamy lab
def _validate_inputs(X, k):
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    
    if k >= X.shape[0] - 1:
        warnings.warn(
            f"Cannot set k ({k}) >= n_samples - 1 ({X.shape[0] - 1}). "
            f"Setting k={X.shape[0] - 2}",
            UserWarning
        )
        k = X.shape[0] - 2
    
    '''
    Maybe here convert X to a numpy array if it is not already
    '''
    return X, k
 

def build_affinity_matrix(X, k, n_jobs=-1):
    X, k = _validate_inputs(X, k)
    n = X.shape[0]
    
    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n), 
        algorithm="auto",
        metric="euclidean",
        n_jobs=n_jobs
    ).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    _check_duplicates(distances, indices)
    
    sigma = distances[:, -1].copy()
    sigma = np.maximum(sigma, np.finfo(float).eps)
    
    # build W using exp(-d^2 / (sigma_i * sigma_j))
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].ravel()
    dist_sq = distances[:, 1:].ravel() ** 2
    
    sigma_i = sigma[rows]
    sigma_j = sigma[cols]
    denom = sigma_i * sigma_j
    denom = np.maximum(denom, np.finfo(float).eps)
    
    weights = np.exp(-dist_sq / denom)
    weights = np.where(np.isnan(weights), 1.0, weights)
    
    # symmetrize
    W = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    W.setdiag(0)
    W.eliminate_zeros()
    
    return W

def build_diffusion_operator(W):
    W = W.tocsr().astype(float)
    degrees = np.array(W.sum(axis=1)).ravel()
    
    if np.any(degrees == 0):
        warnings.warn(
            "Found isolated nodes (degree=0). "
            "Consider increasing k.",
            RuntimeWarning
        )
    
    Q_inv = sp.diags(1.0 / (degrees + np.finfo(float).eps))
    P = Q_inv @ W
    return P

