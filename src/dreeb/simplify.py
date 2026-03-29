import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def simplify_reeb_graph(reeb_nodes, reeb_edges, return_paths=False):
    """
    Simplify Reeb graph by contracting degree-2 chains and compute β1.
    
    Parameters
    ----------
    reeb_nodes : list of dict
        Raw Reeb graph nodes from build_reeb_graph
    reeb_edges : list of tuple
        Raw Reeb graph edges from build_reeb_graph
    
    Parameters
    ----------
    return_paths : bool, optional (default: False)
        If True, also return the list of raw edge ids contracted into
        each simplified edge.

    Returns
    -------
    simp_edges : list of tuple
        Simplified edge list as (u, v) pairs in new node indices
    keep_ids : np.ndarray
        Original node indices that were kept
    beta1 : int
        First Betti number (number of independent cycles)
    comp_count : int
        Number of connected components in simplified graph
    simp_edge_paths : list of list of int, optional
        Raw edge ids underlying each simplified edge. Returned only when
        `return_paths=True`.
    """
    num_nodes = len(reeb_nodes)
    if num_nodes == 0:
        if return_paths:
            return [], np.array([]), 0, 0, []
        return [], np.array([]), 0, 0

    # build adjacency with raw edge ids preserved
    adj = [[] for _ in range(num_nodes)]
    for eid, (u, v) in enumerate(reeb_edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    deg = np.array([len(nei) for nei in adj], dtype=int)

    # mark nodes to keep
    keep = (deg != 2)
    if not np.any(keep):
        keep[0] = True

    keep_ids = np.flatnonzero(keep)
    old2new = -np.ones(num_nodes, dtype=int)
    for i, old in enumerate(keep_ids):
        old2new[old] = i

    # contract degree-2 chains
    visited_mid = np.zeros(num_nodes, dtype=bool)
    simp_edges = []
    simp_edge_paths = []

    for u_old in keep_ids:
        u_new = old2new[u_old]
        for v_old, eid0 in adj[u_old]:
            if keep[v_old]:
                v_new = old2new[v_old]
                if u_old < v_old:
                    simp_edges.append((u_new, v_new))
                    simp_edge_paths.append([int(eid0)])
                continue
            if (not keep[v_old]) and deg[v_old] == 2 and not visited_mid[v_old]:
                prev = u_old
                curr = v_old
                path = [int(eid0)]
                visited_mid[curr] = True
                while True:
                    nbrs = adj[curr]
                    (n0, e0), (n1, e1) = nbrs
                    if n1 == prev:
                        nxt, next_eid = n0, e0
                    else:
                        nxt, next_eid = n1, e1
                    path.append(int(next_eid))
                    prev, curr = curr, nxt
                    if keep[curr]:
                        v_new = old2new[curr]
                        if u_new != v_new:
                            simp_edges.append((u_new, v_new))
                        else:
                            simp_edges.append((u_new, u_new))
                        simp_edge_paths.append(path.copy())
                        break
                    if deg[curr] != 2 or visited_mid[curr]:
                        break
                    visited_mid[curr] = True

    simp_edges = [(int(u), int(v)) for u, v in simp_edges]
    simp_num_nodes = len(keep_ids)
    simp_num_edges = len(simp_edges)

    # compute connected components and β1
    simp_adj = [[] for _ in range(simp_num_nodes)]
    for u, v in simp_edges:
        simp_adj[u].append(v)
        if u != v:
            simp_adj[v].append(u)

    visited = np.zeros(simp_num_nodes, dtype=bool)
    comp_count = 0
    for s in range(simp_num_nodes):
        if not visited[s]:
            comp_count += 1
            stack = [s]
            visited[s] = True
            while stack:
                x = stack.pop()
                for y in simp_adj[x]:
                    if not visited[y]:
                        visited[y] = True
                        stack.append(y)

    beta1 = simp_num_edges - simp_num_nodes + comp_count

    if return_paths:
        return simp_edges, keep_ids, beta1, comp_count, simp_edge_paths

    return simp_edges, keep_ids, beta1, comp_count


def assign_points_to_raw_edges(
    W,
    reeb_nodes,
    reeb_edges,
    step_vertices,
    step_comp_ids,
    uniq_v,
    level_of,
):
    """
    Assign points to raw Reeb edges using adjacent-slice bridging regions.

    For a raw edge connecting nodes (step s, comp c) and (step s+1, comp c_next),
    a point is assigned to that edge if:
    1) its filter level lies between the two adjacent slice values, i.e.
       `level_of == s + 1`, and
    2) it lies in a connected component of the induced graph on levels
       `s, s+1, s+2` that intersects both endpoint node supports.

    Returns
    -------
    point_edges : list of np.ndarray
        For each point, the raw edge ids it belongs to.
    edge_points : list of np.ndarray
        For each raw edge, the point indices assigned to that edge.
    """
    N = W.shape[0]
    edge_points = [np.empty(0, dtype=int) for _ in range(len(reeb_edges))]
    point_edges_accum = [[] for _ in range(N)]

    if len(reeb_edges) == 0:
        return [np.empty(0, dtype=int) for _ in range(N)], edge_points

    if not sp.isspmatrix(W):
        W = sp.csr_matrix(W)
    A = ((W + W.T) > 0).astype(np.int8).tocsr()

    uniq_v_arr = np.asarray(uniq_v, dtype=int)
    level_of = np.asarray(level_of, dtype=int)

    node_points = []
    for info in reeb_nodes:
        s = int(info["step"])
        c = int(info["comp"])
        vs = step_vertices[s]
        cs = step_comp_ids[s]
        if vs.size == 0:
            node_points.append(np.empty(0, dtype=int))
            continue
        comp_vs = vs[cs == c]
        if comp_vs.size == 0:
            comp_vs = vs
        node_points.append(np.unique(uniq_v_arr[comp_vs]))

    edges_by_step = {}
    for eid, (u, v) in enumerate(reeb_edges):
        su = int(reeb_nodes[int(u)]["step"])
        sv = int(reeb_nodes[int(v)]["step"])
        if abs(su - sv) != 1:
            continue
        s = min(su, sv)
        edges_by_step.setdefault(s, []).append((eid, int(u), int(v)))

    local_index = -np.ones(N, dtype=int)

    for s, edge_infos in edges_by_step.items():
        slab_idx = np.flatnonzero((level_of >= s) & (level_of <= s + 2))
        if slab_idx.size == 0:
            continue

        A_sub = A[slab_idx, :][:, slab_idx]
        _, comp_labels = connected_components(A_sub, directed=False)
        comp_labels = np.asarray(comp_labels, dtype=int)
        middle_mask = (level_of[slab_idx] == (s + 1))
        if not np.any(middle_mask):
            continue

        local_index[slab_idx] = np.arange(slab_idx.size, dtype=int)

        for eid, u, v in edge_infos:
            if int(reeb_nodes[u]["step"]) == s:
                left_id, right_id = u, v
            else:
                left_id, right_id = v, u

            left_pts = node_points[left_id]
            right_pts = node_points[right_id]
            if left_pts.size == 0 or right_pts.size == 0:
                continue

            left_local = local_index[left_pts]
            right_local = local_index[right_pts]
            left_local = left_local[left_local >= 0]
            right_local = right_local[right_local >= 0]
            if left_local.size == 0 or right_local.size == 0:
                continue

            left_labels = np.unique(comp_labels[left_local])
            right_labels = np.unique(comp_labels[right_local])
            bridge_labels = np.intersect1d(left_labels, right_labels, assume_unique=False)
            if bridge_labels.size == 0:
                continue

            bridge_mask = middle_mask & np.isin(comp_labels, bridge_labels)
            pts = slab_idx[bridge_mask]
            edge_points[eid] = pts.astype(int, copy=False)
            for p in pts:
                point_edges_accum[int(p)].append(int(eid))

        local_index[slab_idx] = -1

    point_edges = [np.array(ids, dtype=int) for ids in point_edges_accum]
    return point_edges, edge_points


def assign_points_to_simplified_edges(
    simp_edge_paths,
    raw_edge_points,
    num_points,
):
    """
    Aggregate raw edge point assignments onto simplified edges.

    Parameters
    ----------
    simp_edge_paths : list of list of int
        Raw edge ids underlying each simplified edge.
    raw_edge_points : list of np.ndarray
        Point indices per raw edge.
    num_points : int
        Number of points in the original dataset.

    Returns
    -------
    point_edges : list of np.ndarray
        For each point, the simplified edge ids it belongs to.
    edge_points : list of np.ndarray
        For each simplified edge, the point indices assigned to that edge.
    """
    edge_points = []
    point_edges_accum = [[] for _ in range(int(num_points))]

    for simp_eid, raw_path in enumerate(simp_edge_paths):
        if len(raw_path) == 0:
            pts = np.empty(0, dtype=int)
        else:
            pts = np.unique(
                np.concatenate([raw_edge_points[int(raw_eid)] for raw_eid in raw_path])
            )
        edge_points.append(pts)
        for p in pts:
            point_edges_accum[int(p)].append(int(simp_eid))

    point_edges = [np.array(ids, dtype=int) for ids in point_edges_accum]
    return point_edges, edge_points


def assign_points_to_raw_nodes(
    reeb_nodes,
    step_vertices,
    step_comp_ids,
    uniq_v,
    filter_values,
    t_vals,
    chunk_size=100000,
):
    """
    Assign points to raw Reeb nodes by filter-distance.

    Strategy:
    1) Points belonging to a raw node (step, comp) are assigned to the
       closest such node in filter space |f(x) - t_vals[step]|.
    2) Any remaining points are assigned to the closest raw node in
       filter space.

    Parameters
    ----------
    reeb_nodes : list of dict
        Raw Reeb nodes from build_reeb_graph
    step_vertices : list of np.ndarray
        Per-slice active vertex indices (compact)
    step_comp_ids : list of np.ndarray
        Per-slice component IDs per active vertex
    uniq_v : np.ndarray
        Mapping from compacted vertex indices to original point indices
    filter_values : np.ndarray, shape (N,)
        Filter values for all points
    t_vals : np.ndarray, shape (S,)
        Midpoint filter values per slice from prepare_reeb
    chunk_size : int, optional
        Chunk size for assigning remaining points

    Returns
    -------
    point_assignment : np.ndarray, shape (N,)
        Raw node index for each point (-1 if no nodes)
    node_points : list of np.ndarray
        List of point index arrays per raw node
    """
    N = int(filter_values.shape[0])
    point_assignment = -np.ones(N, dtype=int)
    best_dist = np.full(N, np.inf, dtype=float)

    if len(reeb_nodes) == 0:
        return point_assignment, []

    uniq_v_arr = np.asarray(uniq_v, dtype=int)
    f = np.asarray(filter_values, dtype=float)

    # Precompute filter value per raw node
    node_f = np.empty(len(reeb_nodes), dtype=float)
    for nid, info in enumerate(reeb_nodes):
        node_f[nid] = float(t_vals[int(info["step"])])

    # 1) Assign points from raw nodes
    for nid, info in enumerate(reeb_nodes):
        s = int(info["step"])
        c = int(info["comp"])

        vs = step_vertices[s]
        cs = step_comp_ids[s]
        if vs.size == 0:
            continue

        mask = (cs == c)
        comp_vs = vs[mask]
        if comp_vs.size == 0:
            comp_vs = vs

        orig_idx = uniq_v_arr[comp_vs]
        d = np.abs(f[orig_idx] - node_f[nid])
        better = d < best_dist[orig_idx]
        if np.any(better):
            sel = orig_idx[better]
            point_assignment[sel] = nid
            best_dist[sel] = d[better]

    # 2) Assign remaining points by nearest raw node in filter space
    unassigned = np.flatnonzero(point_assignment == -1)
    if unassigned.size > 0:
        K = node_f.shape[0]
        for i in range(0, unassigned.size, chunk_size):
            idx = unassigned[i:i + chunk_size]
            dist = np.abs(f[idx][:, None] - node_f[None, :])
            best = np.argmin(dist, axis=1)
            point_assignment[idx] = best

    node_points = []
    for k in range(len(reeb_nodes)):
        node_points.append(np.flatnonzero(point_assignment == k))

    return point_assignment, node_points


def assign_points_to_simplified_nodes(
    reeb_nodes,
    keep_ids,
    step_vertices,
    step_comp_ids,
    uniq_v,
    filter_values,
    t_vals,
    chunk_size=100000,
):
    """
    Assign points to simplified nodes by filter-distance.

    Strategy:
    1) Points belonging to kept raw Reeb nodes are assigned to that
       simplified node (ties broken by closer filter distance).
    2) Any remaining points are assigned to the closest kept node in
       filter space |f(x) - t_vals[step]|.

    Parameters
    ----------
    reeb_nodes : list of dict
        Raw Reeb nodes from build_reeb_graph
    keep_ids : np.ndarray
        Raw node indices kept after simplification
    step_vertices : list of np.ndarray
        Per-slice active vertex indices (compact)
    step_comp_ids : list of np.ndarray
        Per-slice component IDs per active vertex
    uniq_v : np.ndarray
        Mapping from compacted vertex indices to original point indices
    filter_values : np.ndarray, shape (N,)
        Filter values for all points
    t_vals : np.ndarray, shape (S,)
        Midpoint filter values per slice from prepare_reeb
    chunk_size : int, optional
        Chunk size for assigning remaining points

    Returns
    -------
    point_assignment : np.ndarray, shape (N,)
        Simplified node index for each point (-1 if no kept nodes)
    node_points : list of np.ndarray
        List of point index arrays per simplified node
    """
    N = int(filter_values.shape[0])
    point_assignment = -np.ones(N, dtype=int)
    best_dist = np.full(N, np.inf, dtype=float)

    if keep_ids.size == 0:
        return point_assignment, []

    raw2simp = -np.ones(len(reeb_nodes), dtype=int)
    for new_id, old_id in enumerate(keep_ids):
        raw2simp[int(old_id)] = new_id

    # Precompute filter value per kept node
    keep_node_f = np.empty(len(keep_ids), dtype=float)
    for new_id, old_id in enumerate(keep_ids):
        s = reeb_nodes[int(old_id)]["step"]
        keep_node_f[new_id] = float(t_vals[int(s)])

    # 1) Assign points from kept raw nodes
    uniq_v_arr = np.asarray(uniq_v, dtype=int)
    f = np.asarray(filter_values, dtype=float)
    for old_id in keep_ids:
        old_id = int(old_id)
        new_id = raw2simp[old_id]
        info = reeb_nodes[old_id]
        s = int(info["step"])
        c = int(info["comp"])

        vs = step_vertices[s]
        cs = step_comp_ids[s]
        if vs.size == 0:
            continue

        mask = (cs == c)
        comp_vs = vs[mask]
        if comp_vs.size == 0:
            comp_vs = vs

        orig_idx = uniq_v_arr[comp_vs]
        d = np.abs(f[orig_idx] - keep_node_f[new_id])
        better = d < best_dist[orig_idx]
        if np.any(better):
            sel = orig_idx[better]
            point_assignment[sel] = new_id
            best_dist[sel] = d[better]

    # 2) Assign remaining points by nearest kept node in filter space
    unassigned = np.flatnonzero(point_assignment == -1)
    if unassigned.size > 0:
        K = keep_node_f.shape[0]
        for i in range(0, unassigned.size, chunk_size):
            idx = unassigned[i:i + chunk_size]
            # compute argmin |f(x) - keep_node_f|
            dist = np.abs(f[idx][:, None] - keep_node_f[None, :])
            best = np.argmin(dist, axis=1)
            point_assignment[idx] = best

    # build per-node point lists
    node_points = []
    for k in range(len(keep_ids)):
        node_points.append(np.flatnonzero(point_assignment == k))

    return point_assignment, node_points
