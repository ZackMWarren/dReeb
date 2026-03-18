try:
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError:
    raise ImportError(
        "Visualization requires matplotlib and networkx. "
        "Install with: pip install dreeb[visualize]"
    )

import numpy as np
from collections import defaultdict, Counter


def plot_reeb(
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

        base_delta = 0.015 * span

        # parallel edges
        for (a, b), elist in pair_to_edges.items():
            p0, p1 = node_pos[a], node_pos[b]
            d = p1 - p0
            if np.allclose(d, 0):
                continue
            perp = np.array([-d[1], d[0]])
            perp /= (np.linalg.norm(perp) + 1e-12)

            m = len(elist)
            is_cyc = pair_is_cycle[(a, b)]
            col = (cycle_color if (color_cycles and is_cyc) else noncycle_color)

            if m == 1:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color=col, linewidth=1.4, alpha=0.9, zorder=2)
            else:
                for k in range(m):
                    off = (k - (m - 1) / 2.0) * base_delta * perp
                    q0, q1 = p0 + off, p1 + off
                    ax.plot([q0[0], q1[0]], [q0[1], q1[1]],
                            color=col, linewidth=1.1, alpha=0.9, zorder=2)

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
                   s=40, c="gold", edgecolors="black",
                   linewidths=0.8, zorder=3)

        # degree labels
        if label_degree:
            for i in range(simp_num_nodes):
                x, y = node_pos[i]
                ax.text(x, y, str(deg_map.get(i, 0)),
                        fontsize=9, ha="center", va="center",
                        color="black", zorder=4)

        ax.set_title("Simplified Reeb Graph")
        ax.set_xlabel("Embedding dim 1")
        ax.set_ylabel("Embedding dim 2")

    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=2, alpha=0.03, color="gray")

        base_scale = 0.03 * span

        def _offset_3d(p0, p1, k, m):
            d = p1 - p0
            if np.allclose(d, 0):
                return p0, p1
            perp = np.array([-d[1], d[0], 0.0]) if (abs(d[0]) > 1e-9 or abs(d[1]) > 1e-9) \
                   else np.array([1.0, 0.0, 0.0])
            perp /= (np.linalg.norm(perp) + 1e-12)
            off = ((k - (m - 1) / 2.0) * base_scale) * perp
            return p0 + off, p1 + off

        for (a, b), elist in pair_to_edges.items():
            p0, p1 = node_pos[a], node_pos[b]
            m = len(elist)
            is_cyc = pair_is_cycle[(a, b)]
            col = (cycle_color if (color_cycles and is_cyc) else noncycle_color)
            for k in range(m):
                q0, q1 = _offset_3d(p0, p1, k, m)
                ax.plot([q0[0], q1[0]], [q0[1], q1[1]], [q0[2], q1[2]],
                        color=col, linewidth=1.2, alpha=0.95)

        # self-loops
        theta = np.linspace(0, 2 * np.pi, 60)
        for u, count in loop_counts.items():
            cx, cy, cz = node_pos[u]
            base_r = 0.015 * span
            for i in range(count):
                r = base_r * (1 + 0.4 * i)
                col = cycle_color if color_cycles else noncycle_color
                ax.plot(cx + r * np.cos(theta),
                        cy + r * np.sin(theta),
                        cz + 0 * theta,
                        color=col, linewidth=1.0, alpha=0.9)

        ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2],
                   s=35, c="gold", edgecolors="black",
                   linewidths=0.8, depthshade=False)

        if label_degree:
            for i in range(simp_num_nodes):
                x, y, z = node_pos[i]
                ax.text(x, y, z, str(deg_map.get(i, 0)),
                        fontsize=8, ha="center", va="center", color="black")

        ax.set_title("Simplified Reeb Graph")
        ax.set_xlabel("Embedding dim 1")
        ax.set_ylabel("Embedding dim 2")
        ax.set_zlabel("Embedding dim 3")

    plt.tight_layout()
    plt.show()

    return fig, node_pos