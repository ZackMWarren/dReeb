import numpy as np

from .graph import build_affinity_matrix
from .filter import compute_diffusion_filter
from .reeb import prepare_reeb, build_reeb_graph
from .persistence import compute_edge_lengths, compute_graph_persistence
from .simplify import (
    simplify_reeb_graph,
    assign_points_to_simplified_nodes,
    assign_points_to_raw_nodes,
    assign_points_to_raw_edges,
    assign_points_to_simplified_edges,
    build_simplified_cellular_decomposition,
)


def dreeb(
    X,
    k=80,
    precision=1.0,
    diffusion_eigen_index=1,
    filter_method="diffusion_eigenfunction",
    diffusion_time="auto",
    diffusion_time_max=40,
    simplify=True,
    return_raw=False,
    return_edge_lengths=False,
    return_simp_persistence=False,
    return_raw_persistence=False,
    return_point_assignment=False,
    return_edge_assignment=False,
    return_cellular_decomposition=False,
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
    diffusion_eigen_index : int, optional (default: 1)
        One-based index of the nontrivial diffusion eigenfunction to use
        per connected component. `1` means the first nontrivial
        diffusion eigenfunction, `2` the second, and so on.
    filter_method : {"diffusion_eigenfunction", "rooted_potential_distance"}, optional
        Scalar filter construction method. The default uses a single
        nontrivial diffusion eigenfunction. The rooted potential-distance
        option uses a PHATE-style potential distance from an automatically
        selected root in each connected component.
    diffusion_time : {"auto"} or int, optional (default: "auto")
        Diffusion time for `filter_method="rooted_potential_distance"`.
        Ignored for `diffusion_eigenfunction`.
    diffusion_time_max : int, optional (default: 40)
        Maximum time tested by the entropy-knee heuristic when
        `diffusion_time="auto"` and
        `filter_method="rooted_potential_distance"`.
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
    return_cellular_decomposition : bool, optional (default: False)
        If True and simplify=True, return a mixed disjoint partition of
        all points onto simplified node cells and simplified edge cells.
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
    if return_cellular_decomposition and not simplify:
        raise ValueError(
            "return_cellular_decomposition=True requires simplify=True."
        )

    W = build_affinity_matrix(X, k=k)
    (
        filter_values,
        components,
        filter_aux_values,
        filter_metadata,
    ) = compute_diffusion_filter(
        W,
        diffusion_eigen_index=diffusion_eigen_index,
        precision=precision,
        filter_method=filter_method,
        diffusion_time=diffusion_time,
        diffusion_time_max=diffusion_time_max,
        return_metadata=True,
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
        if return_edge_assignment or return_point_assignment or return_cellular_decomposition:
            (
                simp_edges,
                keep_ids,
                beta1,
                comp_count,
                simp_edge_paths,
                simp_node_paths,
            ) = simplify_reeb_graph(
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
            point_assignment, node_points, node_support_points = assign_points_to_simplified_nodes(
                W=W,
                reeb_nodes=reeb_nodes,
                keep_ids=keep_ids,
                simp_node_paths=simp_node_paths,
                step_vertices=step_vertices,
                step_comp_ids=step_comp_ids,
                uniq_v=prep_state["uniq_v"],
            )
            simp_section["point_assignment"] = point_assignment
            simp_section["node_points"] = node_points
            simp_section["node_support_points"] = node_support_points

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
            (
                simp_point_edge_assignment,
                simp_edge_points,
                simp_point_edges,
                simp_edge_support_points,
            ) = assign_points_to_simplified_edges(
                W=W,
                reeb_nodes=reeb_nodes,
                keep_ids=keep_ids,
                simp_edges=simp_edges,
                simp_edge_paths=simp_edge_paths,
                simp_node_paths=simp_node_paths,
                raw_edge_points=raw_edge_points,
                step_vertices=step_vertices,
                step_comp_ids=step_comp_ids,
                uniq_v=prep_state["uniq_v"],
                num_points=X.shape[0],
            )
            simp_section["point_edge_assignment"] = simp_point_edge_assignment
            simp_section["point_edges"] = simp_point_edges
            simp_section["edge_points"] = simp_edge_points
            simp_section["edge_support_points"] = simp_edge_support_points
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

        if return_cellular_decomposition:
            if "node_support_points" not in simp_section:
                point_assignment, node_points, node_support_points = assign_points_to_simplified_nodes(
                    W=W,
                    reeb_nodes=reeb_nodes,
                    keep_ids=keep_ids,
                    simp_node_paths=simp_node_paths,
                    step_vertices=step_vertices,
                    step_comp_ids=step_comp_ids,
                    uniq_v=prep_state["uniq_v"],
                )
                simp_section["point_assignment"] = point_assignment
                simp_section["node_points"] = node_points
                simp_section["node_support_points"] = node_support_points

            if "edge_support_points" not in simp_section:
                raw_point_edges, raw_edge_points = assign_points_to_raw_edges(
                    W=W,
                    reeb_nodes=reeb_nodes,
                    reeb_edges=reeb_edges,
                    step_vertices=step_vertices,
                    step_comp_ids=step_comp_ids,
                    uniq_v=prep_state["uniq_v"],
                    level_of=prep_state["level_of"],
                )
                (
                    simp_point_edge_assignment,
                    simp_edge_points,
                    simp_point_edges,
                    simp_edge_support_points,
                ) = assign_points_to_simplified_edges(
                    W=W,
                    reeb_nodes=reeb_nodes,
                    keep_ids=keep_ids,
                    simp_edges=simp_edges,
                    simp_edge_paths=simp_edge_paths,
                    simp_node_paths=simp_node_paths,
                    raw_edge_points=raw_edge_points,
                    step_vertices=step_vertices,
                    step_comp_ids=step_comp_ids,
                    uniq_v=prep_state["uniq_v"],
                    num_points=X.shape[0],
                )
                simp_section["point_edge_assignment"] = simp_point_edge_assignment
                simp_section["point_edges"] = simp_point_edges
                simp_section["edge_points"] = simp_edge_points
                simp_section["edge_support_points"] = simp_edge_support_points

            (
                cell_assignment_kind,
                cell_assignment_id,
                node_cells,
                edge_cells,
            ) = build_simplified_cellular_decomposition(
                W=W,
                node_support_points=simp_section["node_support_points"],
                edge_support_points=simp_section["edge_support_points"],
            )
            simp_section["cell_assignment_kind"] = cell_assignment_kind
            simp_section["cell_assignment_id"] = cell_assignment_id
            simp_section["node_cells"] = node_cells
            simp_section["edge_cells"] = edge_cells

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
            point_assignment, node_points, node_support_points = assign_points_to_raw_nodes(
                reeb_nodes=reeb_nodes,
                step_vertices=step_vertices,
                step_comp_ids=step_comp_ids,
                uniq_v=prep_state["uniq_v"],
                filter_values=filter_values,
                t_vals=prep_state["t_vals"],
            )
            raw_section["point_assignment"] = point_assignment
            raw_section["node_points"] = node_points
            raw_section["node_support_points"] = node_support_points

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
            "components": components,
            "filter_method": filter_method,
            "diffusion_eigen_index": int(diffusion_eigen_index),
            "filter_aux_values": filter_aux_values,
            "filter_metadata": filter_metadata,
            "filter_values": filter_values,
            "prep_state": prep_state,
            "step_vertices": step_vertices,
            "step_comp_ids": step_comp_ids,
        }
        if filter_method == "diffusion_eigenfunction":
            result["intermediates"]["diffusion_eigenvalues"] = filter_aux_values

    return result
