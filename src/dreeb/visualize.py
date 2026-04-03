try:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    raise ImportError(
        "Visualization requires matplotlib and networkx."
    )

import numpy as np
from collections import defaultdict, Counter


def _parallel_edge_radii(count, base=0.18):
    """
    Symmetric arc radii for drawing parallel undirected edges.

    Examples:
    count=1 -> [0.0]
    count=2 -> [-0.18, 0.18]
    count=3 -> [-0.18, 0.0, 0.18]
    count=4 -> [-0.36, -0.12, 0.12, 0.36]
    """
    if count <= 1:
        return [0.0]

    if count % 2 == 1:
        offsets = np.arange(-(count // 2), count // 2 + 1, dtype=float)
    else:
        offsets = np.arange(-count + 1, count, 2, dtype=float) / 2.0

    return (base * offsets).tolist()


def _orthonormal_basis_3d(direction):
    """
    Build two unit vectors orthogonal to the given 3D direction.
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    tangent = direction / norm
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangent, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])

    normal_1 = np.cross(tangent, trial)
    normal_1 /= (np.linalg.norm(normal_1) + 1e-12)
    normal_2 = np.cross(tangent, normal_1)
    normal_2 /= (np.linalg.norm(normal_2) + 1e-12)
    return normal_1, normal_2


def _arc_points_3d(p0, p1, rad, span_scale, num=80):
    """
    Quadratic Bezier-style arc between two 3D points.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    chord = p1 - p0
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-12:
        return np.repeat(p0[None, :], num, axis=0)

    n1, n2 = _orthonormal_basis_3d(chord)
    midpoint = 0.5 * (p0 + p1)
    control = midpoint + (rad * span_scale) * n1 + (0.35 * abs(rad) * span_scale) * n2

    t = np.linspace(0.0, 1.0, num)[:, None]
    return ((1.0 - t) ** 2) * p0 + 2.0 * (1.0 - t) * t * control + (t ** 2) * p1


def _loop_points_3d(center, radius, basis_u, basis_v, num=100):
    """
    Circle embedded in 3D using a local 2D basis.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, num)
    center = np.asarray(center, dtype=float)
    return (
        center[None, :]
        + radius * np.cos(theta)[:, None] * basis_u[None, :]
        + radius * np.sin(theta)[:, None] * basis_v[None, :]
    )


def plot_dreeb(
    pts,
    simp_edges,
    keep_ids,
    reeb_nodes,
    step_vertices,
    step_comp_ids,
    uniq_v,
    color_cycles=True,
    label_degree=True,
    figsize=(7, 6),
):
    """
    Visualize the simplified Reeb graph overlaid on embedding coordinates.

    Parameters
    ----------
    pts : np.ndarray, shape (N, D)
        Embedding coordinates to plot on (2D or 3D). This is NOT computed
        by the dreeb pipeline — pass in PHATE, UMAP, PCA, or any 2D/3D
        embedding of your data.
    simp_edges : list of tuple
        Simplified edge list from simplify_reeb_graph
    keep_ids : np.ndarray
        Original node indices kept after simplification, from simplify_reeb_graph
    reeb_nodes : list of dict
        Raw Reeb nodes from build_reeb_graph
    step_vertices : list of np.ndarray
        Per-slice vertex indices from build_reeb_graph
    step_comp_ids : list of np.ndarray
        Per-slice component IDs from build_reeb_graph
    uniq_v : np.ndarray
        Active vertex indices from prepare_reeb (prep_state["uniq_v"])
    color_cycles : bool, optional (default: True)
        If True, color cycle edges crimson and tree edges black
    label_degree : bool, optional (default: True)
        If True, label each node with its multigraph degree
    figsize : tuple, optional (default: (7, 6))
        Figure size passed to matplotlib

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object so the user can save or further modify it
    node_pos : np.ndarray, shape (simp_num_nodes, D)
        The centroid positions used for each Reeb node in embedding space
    """
    pts = np.asarray(pts)
    N, D = pts.shape
    if D not in (2, 3):
        raise ValueError(f"pts must have 2 or 3 columns, got shape {pts.shape}")

    simp_num_nodes = len(keep_ids)
    if simp_num_nodes == 0:
        raise ValueError("No simplified nodes to plot — Reeb graph is empty.")

    # --- 1. compute node positions as centroids ---
    uniq_v_arr = np.asarray(uniq_v, dtype=int)
    node_pos = np.zeros((simp_num_nodes, D), dtype=float)

    for new_id, old_id in enumerate(keep_ids):
        info = reeb_nodes[old_id]
        s = info["step"]
        c = info["comp"]

        vs = step_vertices[s]
        cs = step_comp_ids[s]

        if vs.size == 0:
            node_pos[new_id] = pts.mean(axis=0)
            continue

        mask = (cs == c)
        comp_vs = vs[mask]
        if comp_vs.size == 0:
            comp_vs = vs

        orig_idx = uniq_v_arr[comp_vs]
        node_pos[new_id] = pts[orig_idx].mean(axis=0)

    # --- 2. build networkx multigraph for degree and cycle detection ---
    Gm = nx.MultiGraph()
    Gm.add_nodes_from(range(simp_num_nodes))
    for u, v in simp_edges:
        Gm.add_edge(int(u), int(v))

    deg_map = dict(Gm.degree())

    # --- 3. group edges ---
    pair_to_edges = defaultdict(list)
    loop_counts = Counter()
    for u, v in simp_edges:
        u, v = int(u), int(v)
        if u == v:
            loop_counts[u] += 1
        else:
            a, b = (u, v) if u < v else (v, u)
            pair_to_edges[(a, b)].append((u, v))

    # --- 4. identify cycle edges ---
    G_simple = nx.Graph()
    G_simple.add_nodes_from(range(simp_num_nodes))
    for (a, b) in pair_to_edges.keys():
        G_simple.add_edge(a, b)

    bridge_set = set()
    for a, b in nx.bridges(G_simple):
        bridge_set.add((a, b) if a < b else (b, a))

    pair_is_cycle = {}
    for (a, b), elist in pair_to_edges.items():
        if len(elist) >= 2:
            pair_is_cycle[(a, b)] = True
        else:
            pair_is_cycle[(a, b)] = ((a, b) not in bridge_set)

    cycle_color    = "crimson"
    noncycle_color = "black"
    span = np.ptp(pts, axis=0).max()

    # --- 5. draw ---
    if D == 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.15, color="gray")

        arc_base = 0.22
        node_size = 95
        node_linewidth = 0.9

        # parallel edges
        for (a, b), elist in pair_to_edges.items():
            p0, p1 = node_pos[a], node_pos[b]
            m = len(elist)
            is_cyc = pair_is_cycle[(a, b)]
            col = (cycle_color if (color_cycles and is_cyc) else noncycle_color)
            radii = _parallel_edge_radii(m, base=arc_base)
            line_width = 1.4 if m == 1 else 1.2

            for rad in radii:
                patch = FancyArrowPatch(
                    posA=(p0[0], p0[1]),
                    posB=(p1[0], p1[1]),
                    arrowstyle="-",
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=1.0,
                    linewidth=line_width,
                    color=col,
                    alpha=0.9,
                    zorder=2,
                    shrinkA=0,
                    shrinkB=0,
                )
                ax.add_patch(patch)

        # self-loops
        for u, count in loop_counts.items():
            cx, cy = node_pos[u]
            base_r = 0.012 * span
            for i in range(count):
                r = base_r * (1 + 0.4 * i)
                col = cycle_color if color_cycles else noncycle_color
                circ = plt.Circle((cx, cy), r, edgecolor=col,
                                  facecolor="none", linewidth=1.0, alpha=0.9)
                ax.add_patch(circ)

        # nodes
        ax.scatter(node_pos[:, 0], node_pos[:, 1],
                   s=node_size, c="gold", edgecolors="black",
                   linewidths=node_linewidth, zorder=3)

        # degree labels
        if label_degree:
            for i in range(simp_num_nodes):
                x, y = node_pos[i]
                ax.text(x, y, str(deg_map.get(i, 0)),
                        fontsize=9.5, ha="center", va="center",
                        color="black", zorder=4)

        ax.set_title("Simplified Reeb Graph")
        ax.set_xlabel("Embedding dim 1")
        ax.set_ylabel("Embedding dim 2")

    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=3, alpha=0.15, color="gray")

        arc_base = 0.28 * span

        for (a, b), elist in pair_to_edges.items():
            p0, p1 = node_pos[a], node_pos[b]
            m = len(elist)
            is_cyc = pair_is_cycle[(a, b)]
            col = (cycle_color if (color_cycles and is_cyc) else noncycle_color)
            radii = _parallel_edge_radii(m, base=0.9)
            line_width = 1.4 if m == 1 else 1.2
            for rad in radii:
                curve = _arc_points_3d(p0, p1, rad=rad, span_scale=arc_base, num=90)
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                        color=col, linewidth=line_width, alpha=0.95)

        # self-loops
        for u, count in loop_counts.items():
            center = node_pos[u]
            local_dir = center - pts.mean(axis=0)
            basis_u, basis_v = _orthonormal_basis_3d(local_dir)
            base_r = 0.02 * span
            for i in range(count):
                r = base_r * (1 + 0.4 * i)
                col = cycle_color if color_cycles else noncycle_color
                loop = _loop_points_3d(center, r, basis_u, basis_v, num=120)
                ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                        color=col, linewidth=1.0, alpha=0.9)

        ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2],
                   s=55, c="gold", edgecolors="black",
                   linewidths=0.9, depthshade=False)

        if label_degree:
            for i in range(simp_num_nodes):
                x, y, z = node_pos[i]
                ax.text(x, y, z, str(deg_map.get(i, 0)),
                        fontsize=8, ha="center", va="center", color="black")

        ax.set_title("Simplified Reeb Graph")
        ax.set_xlabel("Embedding dim 1")
        ax.set_ylabel("Embedding dim 2")
        ax.set_zlabel("Embedding dim 3")
        ax.set_box_aspect(np.ptp(pts, axis=0) + 1e-12)
        ax.view_init(elev=20, azim=-58)

    plt.tight_layout()
    if "agg" not in plt.get_backend().lower():
        plt.show()

    return fig, node_pos


def plot_reeb(*args, **kwargs):
    """
    Backward-compatible alias for plot_dreeb().
    """
    return plot_dreeb(*args, **kwargs)
