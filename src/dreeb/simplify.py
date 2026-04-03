import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

from .graph import build_diffusion_operator


def _compute_raw_node_supports(
    reeb_nodes,
    step_vertices,
    step_comp_ids,
    uniq_v,
):
    """
    Recover the original point indices supporting each raw Reeb node.
    """
    uniq_v_arr = np.asarray(uniq_v, dtype=int)
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
    return node_points


def _build_symmetrized_diffusion_cost_graph(W, eps=1e-12):
    """
    Build an undirected shortest-path metric from the symmetrized
    row-stochastic diffusion operator.
    """
    if not sp.isspmatrix(W):
        W = sp.csr_matrix(W)
    W = W.tocsr().astype(float)
    W = W.maximum(W.T)
    W.setdiag(0)
    W.eliminate_zeros()

    P = build_diffusion_operator(W).tocsr().astype(float)
    P_sym = (0.5 * (P + P.T)).tocsr()
    P_sym.setdiag(0)
    P_sym.eliminate_zeros()

    cost_graph = P_sym.copy()
    cost_graph.data = -np.log(np.clip(cost_graph.data, eps, None))
    return cost_graph.tocsr()


def _multi_source_distances(cost_graph, support_points):
    """
    Compute distance from every point to each support set.
    """
    num_points = cost_graph.shape[0]
    distances = np.full((len(support_points), num_points), np.inf, dtype=float)
    for obj_id, pts in enumerate(support_points):
        pts = np.unique(np.asarray(pts, dtype=int))
        if pts.size == 0:
            continue
        dist = dijkstra(cost_graph, directed=False, indices=pts)
        if dist.ndim == 1:
            distances[obj_id] = dist
        else:
            distances[obj_id] = np.min(dist, axis=0)
    return distances


def _partition_from_distances(distances, candidate_ids=None):
    """
    Assign each point to the nearest object with deterministic tie-breaks
    from object order.
    """
    num_objects, num_points = distances.shape
    assignment = -np.ones(num_points, dtype=int)
    if num_objects == 0:
        return assignment

    if candidate_ids is None:
        finite_cols = np.any(np.isfinite(distances), axis=0)
        if np.any(finite_cols):
            assignment[finite_cols] = np.argmin(distances[:, finite_cols], axis=0)
        return assignment

    for point_id in range(num_points):
        candidates = np.asarray(candidate_ids[point_id], dtype=int)
        if candidates.size == 0:
            continue
        point_dist = distances[candidates, point_id]
        finite = np.isfinite(point_dist)
        if not np.any(finite):
            continue
        assignment[point_id] = int(candidates[np.argmin(point_dist)])
    return assignment


def _assignment_to_point_lists(assignment, num_objects):
    point_lists = []
    for obj_id in range(int(num_objects)):
        point_lists.append(np.flatnonzero(assignment == obj_id))
    return point_lists


def _build_point_memberships(support_points, num_points):
    memberships = [[] for _ in range(int(num_points))]
    for obj_id, pts in enumerate(support_points):
        for point_id in np.asarray(pts, dtype=int):
            memberships[int(point_id)].append(int(obj_id))
    return [np.array(ids, dtype=int) for ids in memberships]


def build_simplified_cellular_decomposition(
    W,
    node_support_points,
    edge_support_points,
):
    """
    Build a mixed node/edge cellular decomposition on the simplified graph.

    Each point is assigned to exactly one simplified node cell or one
    simplified edge cell using graph Voronoi distance on the symmetrized
    diffusion metric.
    """
    num_points = W.shape[0]
    cost_graph = _build_symmetrized_diffusion_cost_graph(W)
    node_distances = _multi_source_distances(cost_graph, node_support_points)
    edge_distances = _multi_source_distances(cost_graph, edge_support_points)

    cell_assignment_kind = np.empty(num_points, dtype="<U4")
    cell_assignment_id = -np.ones(num_points, dtype=int)

    node_cells_accum = [[] for _ in range(len(node_support_points))]
    edge_cells_accum = [[] for _ in range(len(edge_support_points))]

    for point_id in range(num_points):
        best_node = np.inf
        best_node_id = -1
        if node_distances.shape[0] > 0:
            d = node_distances[:, point_id]
            finite = np.isfinite(d)
            if np.any(finite):
                best_node_id = int(np.argmin(d))
                best_node = float(d[best_node_id])

        best_edge = np.inf
        best_edge_id = -1
        if edge_distances.shape[0] > 0:
            d = edge_distances[:, point_id]
            finite = np.isfinite(d)
            if np.any(finite):
                best_edge_id = int(np.argmin(d))
                best_edge = float(d[best_edge_id])

        if best_node <= best_edge:
            cell_assignment_kind[point_id] = "node"
            cell_assignment_id[point_id] = best_node_id
            if best_node_id >= 0:
                node_cells_accum[best_node_id].append(point_id)
        else:
            cell_assignment_kind[point_id] = "edge"
            cell_assignment_id[point_id] = best_edge_id
            if best_edge_id >= 0:
                edge_cells_accum[best_edge_id].append(point_id)

    node_cells = [np.array(ids, dtype=int) for ids in node_cells_accum]
    edge_cells = [np.array(ids, dtype=int) for ids in edge_cells_accum]
    return cell_assignment_kind, cell_assignment_id, node_cells, edge_cells


def extract_cellular_trajectory(simplified_graph, cell_indices):
    """
    Extract the unique point indices belonging to selected simplified cells.

    The mixed cell index space is:
    - `0 .. num_nodes - 1` for simplified node cells
    - `num_nodes .. num_nodes + num_edges - 1` for simplified edge cells

    Parameters
    ----------
    simplified_graph : dict
        The `result["simplified"]` section returned by `dreeb(...)` with
        `return_cellular_decomposition=True`.
    cell_indices : sequence of int
        Mixed cell indices in the combined node-then-edge indexing.

    Returns
    -------
    np.ndarray
        Sorted unique point indices belonging to the selected cells.
    """
    if "node_cells" not in simplified_graph or "edge_cells" not in simplified_graph:
        raise ValueError(
            "simplified_graph must contain `node_cells` and `edge_cells`; "
            "run dreeb(..., return_cellular_decomposition=True)."
        )

    node_cells = simplified_graph["node_cells"]
    edge_cells = simplified_graph["edge_cells"]
    num_nodes = len(node_cells)
    num_edges = len(edge_cells)
    total_cells = num_nodes + num_edges

    pieces = []
    for cell_idx in np.asarray(cell_indices, dtype=int):
        if cell_idx < 0 or cell_idx >= total_cells:
            raise IndexError(
                f"cell index {cell_idx} out of range for {total_cells} total cells."
            )
        if cell_idx < num_nodes:
            pts = np.asarray(node_cells[int(cell_idx)], dtype=int)
        else:
            pts = np.asarray(edge_cells[int(cell_idx - num_nodes)], dtype=int)
        if pts.size > 0:
            pieces.append(pts)

    if not pieces:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate(pieces))


def enumerate_terminal_cellular_trajectories(simplified_graph):
    """
    Enumerate edge-simple terminal trajectories on the simplified graph.

    A terminal node is any simplified node with multigraph degree not equal
    to 2. Trajectories are simple paths between terminal nodes, plus simple
    loops that start and end at the same terminal node.

    Returns
    -------
    list of dict
        Each trajectory dictionary contains:
        - node_path
        - edge_path
        - cell_indices
        - graph_length
        - num_points (when cellular decomposition is available)
    """
    nodes = simplified_graph["nodes"]
    edges = simplified_graph["edges"]
    num_nodes = len(nodes)

    adj = [[] for _ in range(num_nodes)]
    degree = np.zeros(num_nodes, dtype=int)
    for edge_id, (u, v) in enumerate(edges):
        u = int(u)
        v = int(v)
        adj[u].append((v, edge_id))
        adj[v].append((u, edge_id))
        if u == v:
            degree[u] += 2
        else:
            degree[u] += 1
            degree[v] += 1

    terminal_nodes = [int(i) for i in range(num_nodes) if degree[i] != 2]
    if not terminal_nodes and num_nodes > 0:
        terminal_nodes = list(range(num_nodes))

    edge_lengths = simplified_graph.get("edge_lengths")

    def graph_length(edge_path):
        if edge_lengths is None:
            return float(len(edge_path))
        return float(sum(float(edge_lengths[int(eid)]) for eid in edge_path))

    def to_cell_indices(node_path, edge_path):
        cell_indices = []
        edge_offset = num_nodes
        for idx, node_id in enumerate(node_path):
            cell_indices.append(int(node_id))
            if idx < len(edge_path):
                cell_indices.append(edge_offset + int(edge_path[idx]))
        return cell_indices

    trajectories = []
    seen = set()

    def add_trajectory(node_path, edge_path):
        edge_key = tuple(int(e) for e in edge_path)
        rev_key = tuple(reversed(edge_key))
        key = min(edge_key, rev_key)
        if key in seen:
            return
        seen.add(key)

        record = {
            "node_path": [int(x) for x in node_path],
            "edge_path": [int(x) for x in edge_path],
            "cell_indices": to_cell_indices(node_path, edge_path),
            "graph_length": graph_length(edge_path),
        }
        if "node_cells" in simplified_graph and "edge_cells" in simplified_graph:
            points = extract_cellular_trajectory(simplified_graph, record["cell_indices"])
            record["num_points"] = int(points.size)
        trajectories.append(record)

    def dfs(start, current, node_path, edge_path, visited_edges, visited_nodes):
        for nxt, edge_id in adj[current]:
            edge_id = int(edge_id)
            nxt = int(nxt)
            if edge_id in visited_edges:
                continue

            if nxt == start and len(edge_path) >= 1:
                add_trajectory(node_path + [start], edge_path + [edge_id])
                continue

            if nxt in visited_nodes:
                continue

            next_node_path = node_path + [nxt]
            next_edge_path = edge_path + [edge_id]
            next_visited_edges = visited_edges | {edge_id}
            next_visited_nodes = visited_nodes | {nxt}

            if nxt in terminal_nodes and nxt != start:
                add_trajectory(next_node_path, next_edge_path)

            dfs(
                start,
                nxt,
                next_node_path,
                next_edge_path,
                next_visited_edges,
                next_visited_nodes,
            )

    for start in terminal_nodes:
        dfs(
            start=int(start),
            current=int(start),
            node_path=[int(start)],
            edge_path=[],
            visited_edges=set(),
            visited_nodes={int(start)},
        )

    trajectories.sort(
        key=lambda item: (
            -float(item["graph_length"]),
            -int(item.get("num_points", 0)),
            tuple(item["cell_indices"]),
        )
    )
    return trajectories


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
    simp_node_paths : list of list of int, optional
        Raw node ids encountered along each simplified edge path, including
        endpoints. Returned only when `return_paths=True`.
    """
    num_nodes = len(reeb_nodes)
    if num_nodes == 0:
        if return_paths:
            return [], np.array([]), 0, 0, [], []
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
    simp_node_paths = []

    for u_old in keep_ids:
        u_new = old2new[u_old]
        for v_old, eid0 in adj[u_old]:
            if keep[v_old]:
                v_new = old2new[v_old]
                if u_old < v_old:
                    simp_edges.append((u_new, v_new))
                    simp_edge_paths.append([int(eid0)])
                    simp_node_paths.append([int(u_old), int(v_old)])
                continue
            if (not keep[v_old]) and deg[v_old] == 2 and not visited_mid[v_old]:
                prev = u_old
                curr = v_old
                path = [int(eid0)]
                node_path = [int(u_old), int(v_old)]
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
                    node_path.append(int(curr))
                    if keep[curr]:
                        v_new = old2new[curr]
                        if u_new != v_new:
                            simp_edges.append((u_new, v_new))
                        else:
                            simp_edges.append((u_new, u_new))
                        simp_edge_paths.append(path.copy())
                        simp_node_paths.append(node_path.copy())
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
        return simp_edges, keep_ids, beta1, comp_count, simp_edge_paths, simp_node_paths

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

    level_of = np.asarray(level_of, dtype=int)
    node_points = _compute_raw_node_supports(
        reeb_nodes=reeb_nodes,
        step_vertices=step_vertices,
        step_comp_ids=step_comp_ids,
        uniq_v=uniq_v,
    )

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
    W,
    reeb_nodes,
    keep_ids,
    simp_edges,
    simp_edge_paths,
    simp_node_paths,
    raw_edge_points,
    step_vertices,
    step_comp_ids,
    uniq_v,
    num_points,
):
    """
    Compute simplified-edge support and ownership.

    Parameters
    ----------
    W : scipy.sparse matrix
        Affinity matrix on the original points.
    reeb_nodes : list of dict
        Raw Reeb nodes from build_reeb_graph.
    keep_ids : np.ndarray
        Raw node ids kept after simplification.
    simp_edges : list of tuple
        Simplified edge list.
    simp_edge_paths : list of list of int
        Raw edge ids underlying each simplified edge.
    simp_node_paths : list of list of int
        Raw node ids encountered along each simplified edge path.
    raw_edge_points : list of np.ndarray
        Point indices per raw edge.
    step_vertices : list of np.ndarray
        Per-slice active vertex indices (compact).
    step_comp_ids : list of np.ndarray
        Per-slice component IDs per active vertex.
    uniq_v : np.ndarray
        Mapping from compacted vertex indices to original point indices.
    num_points : int
        Number of points in the original dataset.

    Returns
    -------
    point_edge_assignment : np.ndarray
        One simplified edge id per point.
    edge_points : list of np.ndarray
        Ownership partition of points per simplified edge.
    point_edges : list of np.ndarray
        Overlapping simplified-edge support memberships per point.
    edge_support_points : list of np.ndarray
        Inherited support points per simplified edge.
    """
    raw_node_points = _compute_raw_node_supports(
        reeb_nodes=reeb_nodes,
        step_vertices=step_vertices,
        step_comp_ids=step_comp_ids,
        uniq_v=uniq_v,
    )

    edge_support_points = []
    for raw_path, node_path in zip(simp_edge_paths, simp_node_paths):
        pieces = []
        for raw_eid in raw_path:
            pts = np.asarray(raw_edge_points[int(raw_eid)], dtype=int)
            if pts.size > 0:
                pieces.append(pts)

        for raw_nid in node_path[1:-1]:
            pts = np.asarray(raw_node_points[int(raw_nid)], dtype=int)
            if pts.size > 0:
                pieces.append(pts)

        if pieces:
            pts = np.unique(np.concatenate(pieces))
        else:
            endpoint_pieces = []
            for raw_nid in node_path:
                pts = np.asarray(raw_node_points[int(raw_nid)], dtype=int)
                if pts.size > 0:
                    endpoint_pieces.append(pts)
            if endpoint_pieces:
                pts = np.unique(np.concatenate(endpoint_pieces))
            else:
                pts = np.empty(0, dtype=int)
        edge_support_points.append(pts)

    point_edges = _build_point_memberships(edge_support_points, num_points)
    cost_graph = _build_symmetrized_diffusion_cost_graph(W)
    edge_distances = _multi_source_distances(cost_graph, edge_support_points)

    raw2simp = -np.ones(len(reeb_nodes), dtype=int)
    for simp_nid, raw_nid in enumerate(np.asarray(keep_ids, dtype=int)):
        raw2simp[int(raw_nid)] = int(simp_nid)

    incident_edges = [[] for _ in range(len(keep_ids))]
    for edge_id, (u, v) in enumerate(simp_edges):
        incident_edges[int(u)].append(int(edge_id))
        if int(v) != int(u):
            incident_edges[int(v)].append(int(edge_id))

    point_candidates = [[] for _ in range(int(num_points))]
    for point_id, support_ids in enumerate(point_edges):
        if support_ids.size > 0:
            point_candidates[point_id] = support_ids.tolist()

    for raw_nid, support_pts in enumerate(raw_node_points):
        simp_nid = raw2simp[int(raw_nid)]
        if simp_nid < 0:
            continue
        candidates = incident_edges[int(simp_nid)]
        if not candidates:
            continue
        for point_id in np.asarray(support_pts, dtype=int):
            if not point_candidates[int(point_id)]:
                point_candidates[int(point_id)] = list(candidates)

    point_edge_assignment = _partition_from_distances(
        edge_distances,
        candidate_ids=point_candidates,
    )
    unresolved = np.flatnonzero(point_edge_assignment < 0)
    if unresolved.size > 0:
        fallback_assignment = _partition_from_distances(edge_distances)
        point_edge_assignment[unresolved] = fallback_assignment[unresolved]

    edge_points = _assignment_to_point_lists(point_edge_assignment, len(simp_edge_paths))
    return point_edge_assignment, edge_points, point_edges, edge_support_points


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
        Ownership partition of points per raw node.
    node_support_points : list of np.ndarray
        Intrinsic support points per raw node recovered from the raw
        Reeb slices.
    """
    N = int(filter_values.shape[0])
    point_assignment = -np.ones(N, dtype=int)
    best_dist = np.full(N, np.inf, dtype=float)

    if len(reeb_nodes) == 0:
        return point_assignment, [], []

    uniq_v_arr = np.asarray(uniq_v, dtype=int)
    f = np.asarray(filter_values, dtype=float)
    node_support_points = _compute_raw_node_supports(
        reeb_nodes=reeb_nodes,
        step_vertices=step_vertices,
        step_comp_ids=step_comp_ids,
        uniq_v=uniq_v,
    )

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

    return point_assignment, node_points, node_support_points


def assign_points_to_simplified_nodes(
    W,
    reeb_nodes,
    keep_ids,
    simp_node_paths,
    step_vertices,
    step_comp_ids,
    uniq_v,
):
    """
    Assign points to simplified nodes via graph Voronoi on the
    symmetrized diffusion metric.

    Parameters
    ----------
    reeb_nodes : list of dict
        Raw Reeb nodes from build_reeb_graph
    keep_ids : np.ndarray
        Raw node indices kept after simplification
    simp_node_paths : list of list of int
        Raw node ids encountered along each simplified edge path,
        including endpoints.
    step_vertices : list of np.ndarray
        Per-slice active vertex indices (compact)
    step_comp_ids : list of np.ndarray
        Per-slice component IDs per active vertex
    uniq_v : np.ndarray
        Mapping from compacted vertex indices to original point indices
    Returns
    -------
    point_assignment : np.ndarray, shape (N,)
        Simplified node index for each point (-1 if no kept nodes)
    node_points : list of np.ndarray
        Ownership partition of points per simplified node.
    node_support_points : list of np.ndarray
        Intrinsic support points for the kept simplified nodes only.
    """
    if keep_ids.size == 0:
        return -np.ones(W.shape[0], dtype=int), [], []

    raw_node_points = _compute_raw_node_supports(
        reeb_nodes=reeb_nodes,
        step_vertices=step_vertices,
        step_comp_ids=step_comp_ids,
        uniq_v=uniq_v,
    )
    node_support_points = [raw_node_points[int(old_id)] for old_id in keep_ids]
    cost_graph = _build_symmetrized_diffusion_cost_graph(W)
    node_distances = _multi_source_distances(cost_graph, node_support_points)
    point_assignment = _partition_from_distances(node_distances)
    node_points = _assignment_to_point_lists(point_assignment, len(keep_ids))
    return point_assignment, node_points, node_support_points
