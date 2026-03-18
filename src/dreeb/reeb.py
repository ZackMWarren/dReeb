import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra, connected_components
import warnings
from numba import njit
from numba.typed import List

def prepare_reeb(W, filter_values):
    """
    Prepare data structures for midpoint slabbing Reeb graph construction.
    
    Parameters
    ----------
    W : scipy.sparse matrix, shape (N, N)
        Affinity matrix
    filter_values : np.ndarray, shape (N,)
        Filter function f, normalized to [0, 1]
    
    Returns
    -------
    prep_state : dict containing:
        f        : cleaned filter values
        alphas   : sorted unique filter levels
        t_vals   : midpoints between consecutive levels
        level_of : per-point index into alphas
        ei_c     : compacted source vertex of each active edge
        ej_c     : compacted destination vertex of each active edge
        lmin     : lower level index of each active edge
        lmax     : upper level index of each active edge
        M        : number of active vertices
        S        : number of midpoints
        old2new  : mapping from original to compacted vertex indices
        uniq_v   : original vertex indices of active vertices
    """
    if not sp.isspmatrix(W):
        W = sp.csr_matrix(W)
    W = W.tocsr().astype(float)
    W.setdiag(0)
    W.eliminate_zeros()

    N = W.shape[0]
    f = np.asarray(filter_values, dtype=float)

    if f.shape[0] != N:
        raise ValueError(f"Filter length {f.shape[0]} != N={N} from W.")

    # --- clean up f ---
    finite_mask = np.isfinite(f)
    if not np.all(finite_mask):
        min_f = float(np.min(f[finite_mask]))
        f = f.copy()
        f[~finite_mask] = min_f
        warnings.warn(
            f"Replaced {np.sum(~finite_mask)} non-finite filter entries with {min_f:.4g}.",
            RuntimeWarning
        )

    # --- compress f to unique levels ---
    alphas, level_of = np.unique(f, return_inverse=True)
    alphas = alphas.astype(float)
    level_of = level_of.astype(np.int32)
    K = alphas.size

    if K < 2:
        raise ValueError("Filter has <2 distinct values; cannot build nontrivial Reeb structure.")

    # midpoints between consecutive levels
    t_vals = 0.5 * (alphas[:-1] + alphas[1:])
    S = t_vals.size

    # --- build undirected edge list ---
    # Use union symmetrization to be robust to minor asymmetries.
    A = ((W + W.T) > 0).astype(np.int8).tocsr()
    A_coo = A.tocoo()

    # keep only upper triangle to avoid duplicate edges
    mask_upper = A_coo.row < A_coo.col
    ei = A_coo.row[mask_upper].astype(np.int32)
    ej = A_coo.col[mask_upper].astype(np.int32)
    E = ei.size

    if E == 0:
        raise ValueError("No edges in W; cannot build Reeb graph.")

    # --- For each edge, compute its active midpoint interval [lmin, lmax) ---
    li = level_of[ei]
    lj = level_of[ej]
    lmin = np.minimum(li, lj)
    lmax = np.maximum(li, lj)

    active = (lmin < lmax)
    ei = ei[active]
    ej = ej[active]
    lmin = lmin[active]
    lmax = lmax[active]
    E_active = ei.size

    if E_active == 0:
        raise ValueError(
            "No edges span multiple f-levels. "
            "Cannot build meaningful level-set Reeb graph."
        )

    # --- compact to vertices that appear in at least one active edge ---
    uniq_v = np.unique(np.concatenate([ei, ej]))
    M = uniq_v.size
    old2new = -np.ones(N, dtype=np.int32)
    old2new[uniq_v] = np.arange(M, dtype=np.int32)

    ei_c = old2new[ei]
    ej_c = old2new[ej]

    return {
        "f": f,
        "alphas": alphas,
        "t_vals": t_vals,
        "level_of": level_of,
        "ei_c": ei_c,
        "ej_c": ej_c,
        "lmin": lmin,
        "lmax": lmax,
        "M": M,
        "S": S,
        "old2new": old2new,
        "uniq_v": uniq_v,
    }

def _build_segment_tree(S, lmin, lmax):
    """
    Build a segment tree over [0, S) storing edge intervals.
    
    Parameters
    ----------
    S : int
        Number of midpoints
    lmin : np.ndarray
        Lower level index of each active edge
    lmax : np.ndarray
        Upper level index of each active edge
    
    Returns
    -------
    size : int
        Size of segment tree (next power of 2 >= S)
    tree_starts : np.ndarray
        Start indices into tree_edges for each node
    tree_edges : np.ndarray
        Flattened edge indices stored in segment tree
    """
    E_active = lmin.size

    # find next power of 2 >= S
    size = 1
    while size < S:
        size <<= 1

    # build python list-of-lists segment tree
    tree_py = [[] for _ in range(2 * size)]

    def _add_interval(node, nl, nr, ql, qr, eidx):
        if qr <= nl or nr <= ql:
            return
        if ql <= nl and nr <= qr:
            tree_py[node].append(eidx)
            return
        mid = (nl + nr) // 2
        _add_interval(node * 2,     nl,  mid, ql, qr, eidx)
        _add_interval(node * 2 + 1, mid, nr,  ql, qr, eidx)

    for eidx in range(E_active):
        ql = int(lmin[eidx])
        qr = int(lmax[eidx])
        if ql < qr:
            _add_interval(1, 0, S, ql, qr, eidx)

    # flatten into numpy arrays for numba
    node_count = 2 * size
    counts = np.array([len(tree_py[i]) for i in range(node_count)], dtype=np.int32)

    tree_starts = np.empty(node_count + 1, dtype=np.int32)
    tree_starts[0] = 0
    np.cumsum(counts, out=tree_starts[1:])

    total_assign = int(tree_starts[-1])
    tree_edges = np.empty(total_assign, dtype=np.int32)

    pos = 0
    for i in range(node_count):
        for e in tree_py[i]:
            tree_edges[pos] = int(e)
            pos += 1

    return size, tree_starts, tree_edges


@njit
def _reeb_levelsets_numba(M, S, size, tree_starts, tree_edges, ei_c, ej_c,
                           step_vertices, step_comp_ids):
    """
    Segment tree DFS + rollback union-find to compute level-set
    connected components at each midpoint slice.
    
    Mutates step_vertices[s] and step_comp_ids[s] in place for each slice s.
    """
    parent = np.empty(M, dtype=np.int32)
    sizeUF = np.empty(M, dtype=np.int32)
    deg    = np.zeros(M, dtype=np.int32)
    for i in range(M):
        parent[i] = i
        sizeUF[i] = 1

    comps_active = 0

    op_max  = 8 * tree_edges.shape[0] + 1
    op_type = np.empty(op_max, dtype=np.int8)
    op_v    = np.empty(op_max, dtype=np.int32)
    op_rv   = np.empty(op_max, dtype=np.int32)
    op_ru   = np.empty(op_max, dtype=np.int32)
    op_sz   = np.empty(op_max, dtype=np.int32)
    op_len  = 0

    cid_for_root = -np.ones(M, dtype=np.int32)
    used_roots   = np.empty(M, dtype=np.int32)

    max_stack   = 4 * (2 * size) + 4
    node_stack  = np.empty(max_stack, dtype=np.int32)
    nl_stack    = np.empty(max_stack, dtype=np.int32)
    nr_stack    = np.empty(max_stack, dtype=np.int32)
    state_stack = np.empty(max_stack, dtype=np.int32)
    mark_stack  = np.empty(max_stack, dtype=np.int32)
    top = 0

    node_stack[top]  = 1
    nl_stack[top]    = 0
    nr_stack[top]    = S
    state_stack[top] = 0
    mark_stack[top]  = 0
    top += 1

    def uf_find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    while top > 0:
        top -= 1
        node = node_stack[top]
        nl   = nl_stack[top]
        nr   = nr_stack[top]
        st   = state_stack[top]
        mark = mark_stack[top]

        if st == 0:
            mark = op_len

            start = tree_starts[node]
            end   = tree_starts[node + 1]
            for idx in range(start, end):
                eidx = tree_edges[idx]
                u = ei_c[eidx]
                v = ej_c[eidx]

                if deg[u] == 0:
                    deg[u] = 1
                    comps_active += 1
                    op_type[op_len] = 1
                    op_v[op_len]    = u
                    op_len += 1
                else:
                    deg[u] += 1
                    op_type[op_len] = 2
                    op_v[op_len]    = u
                    op_len += 1

                if deg[v] == 0:
                    deg[v] = 1
                    comps_active += 1
                    op_type[op_len] = 1
                    op_v[op_len]    = v
                    op_len += 1
                else:
                    deg[v] += 1
                    op_type[op_len] = 2
                    op_v[op_len]    = v
                    op_len += 1

                ru = uf_find(u)
                rv = uf_find(v)
                if ru == rv:
                    op_type[op_len] = 3
                    op_len += 1
                else:
                    if sizeUF[ru] < sizeUF[rv]:
                        tmp = ru; ru = rv; rv = tmp
                    sz_child    = sizeUF[rv]
                    parent[rv]  = ru
                    sizeUF[ru] += sz_child
                    comps_active -= 1
                    op_type[op_len] = 0
                    op_rv[op_len]   = rv
                    op_ru[op_len]   = ru
                    op_sz[op_len]   = sz_child
                    op_len += 1

            if nr - nl == 1:
                s = nl
                if comps_active > 0:
                    cnt = 0
                    for v in range(M):
                        if deg[v] > 0:
                            cnt += 1
                    if cnt > 0:
                        active_vs = np.empty(cnt, dtype=np.int32)
                        roots     = np.empty(cnt, dtype=np.int32)
                        idx = 0
                        for v in range(M):
                            if deg[v] > 0:
                                active_vs[idx] = v
                                roots[idx]     = uf_find(v)
                                idx += 1
                        comp_ids = np.empty(cnt, dtype=np.int32)
                        cid = 0
                        for i in range(cnt):
                            r = roots[i]
                            c = cid_for_root[r]
                            if c == -1:
                                c = cid
                                cid_for_root[r] = c
                                used_roots[cid] = r
                                cid += 1
                            comp_ids[i] = c
                        for i in range(cid):
                            cid_for_root[used_roots[i]] = -1
                        step_vertices[s] = active_vs
                        step_comp_ids[s] = comp_ids
                    else:
                        step_vertices[s] = np.empty(0, dtype=np.int32)
                        step_comp_ids[s] = np.empty(0, dtype=np.int32)
                else:
                    step_vertices[s] = np.empty(0, dtype=np.int32)
                    step_comp_ids[s] = np.empty(0, dtype=np.int32)

                # rollback leaf
                while op_len > mark:
                    op_len -= 1
                    t = op_type[op_len]
                    if t == 0:
                        sizeUF[op_ru[op_len]] -= op_sz[op_len]
                        parent[op_rv[op_len]]  = op_rv[op_len]
                        comps_active += 1
                    elif t == 1:
                        deg[op_v[op_len]] = 0
                        comps_active -= 1
                    elif t == 2:
                        deg[op_v[op_len]] -= 1

            else:
                mid = (nl + nr) // 2

                # exit frame
                node_stack[top]  = node
                nl_stack[top]    = nl
                nr_stack[top]    = nr
                state_stack[top] = 1
                mark_stack[top]  = mark
                top += 1

                # right child
                node_stack[top]  = node * 2 + 1
                nl_stack[top]    = mid
                nr_stack[top]    = nr
                state_stack[top] = 0
                mark_stack[top]  = 0
                top += 1

                # left child
                node_stack[top]  = node * 2
                nl_stack[top]    = nl
                nr_stack[top]    = mid
                state_stack[top] = 0
                mark_stack[top]  = 0
                top += 1

        else:
            # rollback internal node
            while op_len > mark:
                op_len -= 1
                t = op_type[op_len]
                if t == 0:
                    sizeUF[op_ru[op_len]] -= op_sz[op_len]
                    parent[op_rv[op_len]]  = op_rv[op_len]
                    comps_active += 1
                elif t == 1:
                    deg[op_v[op_len]] = 0
                    comps_active -= 1
                elif t == 2:
                    deg[op_v[op_len]] -= 1


def _build_reeb_graph(S, M, step_vertices, step_comp_ids):
    """
    Build discrete Reeb multigraph from consecutive level-set components.
    
    Parameters
    ----------
    S : int
        Number of midpoints
    M : int
        Number of active (compacted) vertices
    step_vertices : list of np.ndarray
        Per-slice active vertex indices
    step_comp_ids : list of np.ndarray
        Per-slice component IDs for each active vertex
    
    Returns
    -------
    reeb_nodes : list of dict
        Each dict has keys: id, step, comp
    reeb_edges : list of tuple
        Each tuple is (u, v) node id pair
    """
    reeb_nodes = []
    reeb_edges = []
    _node_id   = {}

    def _get_node_id(s, c):
        key = (s, c)
        nid = _node_id.get(key)
        if nid is None:
            nid = len(reeb_nodes)
            _node_id[key] = nid
            reeb_nodes.append({"id": nid, "step": s, "comp": c})
        return nid

    if S >= 2:
        scratch = np.full(M, -1, dtype=np.int32)
        for s in range(S - 1):
            vs  = step_vertices[s]
            cs  = step_comp_ids[s]
            vsn = step_vertices[s + 1]
            csn = step_comp_ids[s + 1]

            if vs.size == 0 or vsn.size == 0:
                continue

            scratch[:] = -1
            for v, c in zip(vs, cs):
                scratch[v] = c

            pairs = set()
            for v, c_next in zip(vsn, csn):
                c_prev = scratch[v]
                if c_prev != -1:
                    pairs.add((int(c_prev), int(c_next)))

            for c_prev, c_next in pairs:
                u = _get_node_id(s,     c_prev)
                v = _get_node_id(s + 1, c_next)
                reeb_edges.append((u, v))

    return reeb_nodes, reeb_edges


def build_reeb_graph(prep_state):
    """
    Build the Reeb graph using midpoint slabbing with a segment tree
    and rollback union-find (Numba-accelerated).

    Parameters
    ----------
    prep_state : dict
        Output of prepare_reeb()

    Returns
    -------
    reeb_nodes : list of dict
        Nodes of the Reeb graph, each with keys: id, step, comp
    reeb_edges : list of tuple
        Edges of the Reeb graph as (u, v) node id pairs
    step_vertices : list of np.ndarray
        Per-slice active vertex indices (compact)
    step_comp_ids : list of np.ndarray
        Per-slice component IDs per active vertex
    """
    ei_c = prep_state["ei_c"].astype(np.int32)
    ej_c = prep_state["ej_c"].astype(np.int32)
    lmin = prep_state["lmin"]
    lmax = prep_state["lmax"]
    M    = int(prep_state["M"])
    S    = int(prep_state["S"])

    if S <= 0 or ei_c.size == 0:
        raise ValueError("Nothing to do: no midpoints or no active edges.")

    # 1) build segment tree
    size, tree_starts, tree_edges = _build_segment_tree(S, lmin, lmax)

    # 2) allocate typed lists for numba
    step_vertices_nb = List()
    step_comp_ids_nb = List()
    for _ in range(S):
        step_vertices_nb.append(np.empty(0, dtype=np.int32))
        step_comp_ids_nb.append(np.empty(0, dtype=np.int32))

    # 3) run numba core
    _reeb_levelsets_numba(
        M, S, size,
        tree_starts, tree_edges,
        ei_c, ej_c,
        step_vertices_nb, step_comp_ids_nb
    )

    # 4) convert back to regular python lists
    step_vertices = [np.array(a, copy=False) for a in step_vertices_nb]
    step_comp_ids = [np.array(a, copy=False) for a in step_comp_ids_nb]

    # 5) assemble reeb graph
    reeb_nodes, reeb_edges = _build_reeb_graph(S, M, step_vertices, step_comp_ids)

    return reeb_nodes, reeb_edges, step_vertices, step_comp_ids