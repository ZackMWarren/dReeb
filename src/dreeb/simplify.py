import numpy as np

def simplify_reeb_graph(reeb_nodes, reeb_edges):
    """
    Simplify Reeb graph by contracting degree-2 chains and compute β1.
    
    Parameters
    ----------
    reeb_nodes : list of dict
        Raw Reeb graph nodes from build_reeb_graph
    reeb_edges : list of tuple
        Raw Reeb graph edges from build_reeb_graph
    
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
    """
    num_nodes = len(reeb_nodes)
    if num_nodes == 0:
        return [], np.array([]), 0, 0

    # build adjacency
    adj = [[] for _ in range(num_nodes)]
    for u, v in reeb_edges:
        adj[u].append(v)
        adj[v].append(u)
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

    for u_old in keep_ids:
        u_new = old2new[u_old]
        for v_old in adj[u_old]:
            if keep[v_old]:
                v_new = old2new[v_old]
                if u_old < v_old:
                    simp_edges.append((u_new, v_new))
                continue
            if (not keep[v_old]) and deg[v_old] == 2 and not visited_mid[v_old]:
                prev = u_old
                curr = v_old
                visited_mid[curr] = True
                while True:
                    nbrs = adj[curr]
                    nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                    prev, curr = curr, nxt
                    if keep[curr]:
                        v_new = old2new[curr]
                        if u_new != v_new:
                            simp_edges.append((u_new, v_new))
                        else:
                            simp_edges.append((u_new, u_new))
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

    return simp_edges, keep_ids, beta1, comp_count