import numpy as np

from .graph import build_affinity_matrix, build_diffusion_operator
from .filter import find_endpoints, compute_filter
from .reeb import prepare_reeb, build_reeb_graph
from .persistence import compute_edge_lengths, compute_graph_persistence
from .simplify import (
    simplify_reeb_graph,
    assign_points_to_simplified_nodes,
    assign_points_to_raw_nodes,
    assign_points_to_raw_edges,
    assign_points_to_simplified_edges,
)


def dreeb(
    X,
    k=80,
    precision=1.0,
    simplify=True,
    return_raw=False,
    return_edge_lengths=False,
    return_simp_persistence=False,
    return_raw_persistence=False,
    return_point_assignment=False,
    return_edge_assignment=False,
    return_intermediates=False,
):
    """
    Run the full dReeb pipeline.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input point cloud data
    k : int, optional (default: 80)
        Number of nearest neighbors
    precision : float, optional (default: 1.0)
        Controls filter quantization. 1.0 = no quantization,
        0.1 = 10% of unique levels
    simplify : bool, optional (default: True)
        If True, simplify the Reeb graph by contracting degree-2 chains.
    return_raw : bool, optional (default: False)
        If True, include the raw Reeb graph in the returned result.
    return_edge_lengths : bool, optional (default: False)
        If True, compute and return edge lengths in filter space for
        whichever graph outputs are returned.
    return_simp_persistence : bool, optional (default: False)
        If True, compute and return an edge-length persistence summary
        for the simplified graph.
    return_raw_persistence : bool, optional (default: False)
        If True, compute and return an edge-length persistence summary
        for the raw Reeb graph.
    return_point_assignment : bool, optional (default: False)
        If True, assign points to nodes by filter-distance. When
        simplify=True, assigns to simplified nodes; otherwise assigns
        to raw nodes.
    return_edge_assignment : bool, optional (default: False)
        If True, assign points to raw and/or simplified edges using
        adjacent-slice bridging regions.
    return_intermediates : bool, optional (default: False)
        If True, include intermediate matrices and arrays used during
        construction.

    Returns
    -------
    dict
        Structured result with keys:

        - primary_graph : "simplified" or "raw"
        - simplified    : simplified graph section, when requested/primary
        - raw           : raw graph section, when requested/primary
        - intermediates : intermediate construction outputs, when requested
    """
    W = build_affinity_matrix(X, k=k)
    P = build_diffusion_operator(W)
    components, roots_A, roots_B, cost, diameters = find_endpoints(P)
    filter_values, distA, distB = compute_filter(
        P, cost, components, roots_A, roots_B, precision=precision
    )
    prep_state = prepare_reeb(W, filter_values)
    reeb_nodes, reeb_edges, step_vertices, step_comp_ids = build_reeb_graph(prep_state)
    t_vals = prep_state["t_vals"]

    result = {
        "primary_graph": "simplified" if simplify else "raw",
    }

    raw_section = None
    simp_section = None
    raw_edge_lengths = None
    raw_point_edges = None
    raw_edge_points = None

    if simplify:
        if return_edge_assignment:
            simp_edges, keep_ids, beta1, comp_count, simp_edge_paths = simplify_reeb_graph(
                reeb_nodes, reeb_edges, return_paths=True
            )
        else:
            simp_edges, keep_ids, beta1, comp_count = simplify_reeb_graph(reeb_nodes, reeb_edges)
        simp_nodes = [reeb_nodes[int(i)] for i in keep_ids]

        simp_section = {
            "nodes": simp_nodes,
            "edges": simp_edges,
            "keep_ids": keep_ids,
            "beta1": beta1,
            "comp_count": comp_count,
        }

        if return_edge_lengths or return_simp_persistence:
            simp_edge_lengths = compute_edge_lengths(simp_nodes, simp_edges, t_vals)

        if return_edge_lengths:
            simp_section["edge_lengths"] = simp_edge_lengths

        if return_simp_persistence:
            simp_section["persistence"] = compute_graph_persistence(
                num_nodes=len(simp_nodes),
                edges=simp_edges,
                edge_lengths=simp_edge_lengths,
            )

        if return_point_assignment:
            point_assignment, node_points = assign_points_to_simplified_nodes(
                reeb_nodes=reeb_nodes,
                keep_ids=keep_ids,
                step_vertices=step_vertices,
                step_comp_ids=step_comp_ids,
                uniq_v=prep_state["uniq_v"],
                filter_values=filter_values,
                t_vals=prep_state["t_vals"],
            )
            simp_section["point_assignment"] = point_assignment
            simp_section["node_points"] = node_points

        if return_edge_assignment:
            raw_point_edges, raw_edge_points = assign_points_to_raw_edges(
                W=W,
                reeb_nodes=reeb_nodes,
                reeb_edges=reeb_edges,
                step_vertices=step_vertices,
                step_comp_ids=step_comp_ids,
                uniq_v=prep_state["uniq_v"],
                level_of=prep_state["level_of"],
            )
            simp_point_edges, simp_edge_points = assign_points_to_simplified_edges(
                simp_edge_paths=simp_edge_paths,
                raw_edge_points=raw_edge_points,
                num_points=X.shape[0],
            )
            simp_section["point_edges"] = simp_point_edges
            simp_section["edge_points"] = simp_edge_points
            if raw_section is None:
                raw_section = {
                    "nodes": reeb_nodes,
                    "edges": reeb_edges,
                }
            else:
                raw_section.setdefault("nodes", reeb_nodes)
                raw_section.setdefault("edges", reeb_edges)
            raw_section["point_edges"] = raw_point_edges
            raw_section["edge_points"] = raw_edge_points

        if return_raw_persistence:
            raw_edge_lengths = compute_edge_lengths(reeb_nodes, reeb_edges, t_vals)
            if raw_section is None:
                raw_section = {}
            raw_section["persistence"] = compute_graph_persistence(
                num_nodes=len(reeb_nodes),
                edges=reeb_edges,
                edge_lengths=raw_edge_lengths,
            )
    else:
        raw_section = {
            "nodes": reeb_nodes,
            "edges": reeb_edges,
        }

        if return_edge_lengths or return_raw_persistence:
            raw_edge_lengths = compute_edge_lengths(reeb_nodes, reeb_edges, t_vals)

        if return_edge_lengths:
            raw_section["edge_lengths"] = raw_edge_lengths

        if return_raw_persistence:
            raw_section["persistence"] = compute_graph_persistence(
                num_nodes=len(reeb_nodes),
                edges=reeb_edges,
                edge_lengths=raw_edge_lengths,
            )

        if return_point_assignment:
            point_assignment, node_points = assign_points_to_raw_nodes(
                reeb_nodes=reeb_nodes,
                step_vertices=step_vertices,
                step_comp_ids=step_comp_ids,
                uniq_v=prep_state["uniq_v"],
                filter_values=filter_values,
                t_vals=prep_state["t_vals"],
            )
            raw_section["point_assignment"] = point_assignment
            raw_section["node_points"] = node_points

        if return_edge_assignment:
            raw_point_edges, raw_edge_points = assign_points_to_raw_edges(
                W=W,
                reeb_nodes=reeb_nodes,
                reeb_edges=reeb_edges,
                step_vertices=step_vertices,
                step_comp_ids=step_comp_ids,
                uniq_v=prep_state["uniq_v"],
                level_of=prep_state["level_of"],
            )
            raw_section["point_edges"] = raw_point_edges
            raw_section["edge_points"] = raw_edge_points

    if simplify and return_raw:
        if raw_section is None:
            raw_section = {
                "nodes": reeb_nodes,
                "edges": reeb_edges,
            }
        else:
            raw_section.setdefault("nodes", reeb_nodes)
            raw_section.setdefault("edges", reeb_edges)

        if return_edge_lengths:
            if "edge_lengths" not in raw_section:
                if raw_edge_lengths is None:
                    raw_edge_lengths = compute_edge_lengths(reeb_nodes, reeb_edges, t_vals)
                raw_section["edge_lengths"] = raw_edge_lengths

        if return_edge_assignment:
            if raw_point_edges is None or raw_edge_points is None:
                raw_point_edges, raw_edge_points = assign_points_to_raw_edges(
                    W=W,
                    reeb_nodes=reeb_nodes,
                    reeb_edges=reeb_edges,
                    step_vertices=step_vertices,
                    step_comp_ids=step_comp_ids,
                    uniq_v=prep_state["uniq_v"],
                    level_of=prep_state["level_of"],
                )
            raw_section["point_edges"] = raw_point_edges
            raw_section["edge_points"] = raw_edge_points

    if simp_section is not None:
        result["simplified"] = simp_section
    if raw_section is not None:
        result["raw"] = raw_section

    if return_intermediates:
        result["intermediates"] = {
            "W": W,
            "P": P,
            "components": components,
            "roots_A": roots_A,
            "roots_B": roots_B,
            "diameters": diameters,
            "filter_values": filter_values,
            "distA": distA,
            "distB": distB,
            "prep_state": prep_state,
            "step_vertices": step_vertices,
            "step_comp_ids": step_comp_ids,
        }

    return result
