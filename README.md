# dReeb

`dreeb` builds a discrete Reeb graph from a point cloud.

The main entrypoint is `dreeb.dreeb()`. By default it returns the simplified
graph only and skips optional work such as persistence, point assignment, edge
assignment, and large intermediate outputs.

The scalar filter is the first nontrivial diffusion eigenfunction computed
separately on each connected component of the affinity graph. You can choose a
later nontrivial eigenfunction with `diffusion_eigen_index`.

## Installation

Install with pip:

```bash
pip install dreeb
```

After installation, both the main pipeline and plotting are available directly:

```python
from dreeb import (
    dreeb,
    plot_dreeb,
    extract_cellular_trajectory,
    enumerate_terminal_cellular_trajectories,
)
```

## Quickstart

```python
from dreeb import dreeb

result = dreeb(X)

graph = result["simplified"]
nodes = graph["nodes"]
edges = graph["edges"]
beta1 = graph["beta1"]
```

Default return:

- `result["primary_graph"] == "simplified"`
- `result["simplified"]["nodes"]`
- `result["simplified"]["edges"]`
- `result["simplified"]["keep_ids"]`
- `result["simplified"]["beta1"]`
- `result["simplified"]["comp_count"]`

If you want the raw graph instead:

```python
result = dreeb(X, simplify=False)

raw = result["raw"]
nodes = raw["nodes"]
edges = raw["edges"]
```

If you want to plot the result, request the raw graph and intermediates:

```python
from dreeb import dreeb, plot_dreeb

result = dreeb(X, return_raw=True, return_intermediates=True)

fig, node_pos = plot_dreeb(
    pts=embedding_2d,
    simp_edges=result["simplified"]["edges"],
    keep_ids=result["simplified"]["keep_ids"],
    reeb_nodes=result["raw"]["nodes"],
    step_vertices=result["intermediates"]["step_vertices"],
    step_comp_ids=result["intermediates"]["step_comp_ids"],
    uniq_v=result["intermediates"]["prep_state"]["uniq_v"],
)
```

## Result Structure

`dreeb()` returns a structured dictionary with stable top-level keys:

- `primary_graph`: `"simplified"` or `"raw"`
- `simplified`: simplified graph outputs when `simplify=True`
- `raw`: raw graph outputs when requested, or when `simplify=False`
- `intermediates`: internal construction outputs when requested

Each section only contains the fields you asked for, but the top-level layout
stays stable.

## Main Options

```python
result = dreeb(
    X,
    diffusion_eigen_index=1,
    return_raw=True,
    return_edge_lengths=True,
    return_simp_persistence=True,
    return_raw_persistence=True,
    return_point_assignment=True,
    return_edge_assignment=True,
    return_cellular_decomposition=True,
    return_intermediates=True,
)
```

- `return_raw=True`
  Include the raw graph under `result["raw"]`.
- `diffusion_eigen_index=1`
  Use the first nontrivial diffusion eigenfunction. Set `2`, `3`, ... to use
  later nontrivial diffusion eigenfunctions.
- `return_edge_lengths=True`
  Return filter-space edge lengths in the graph sections that are present.
- `return_simp_persistence=True`
  Return simplified graph persistence under `result["simplified"]["persistence"]`.
- `return_raw_persistence=True`
  Return raw graph persistence under `result["raw"]["persistence"]`.
- `return_point_assignment=True`
  Return node-level point ownership in the active graph section.
- `return_edge_assignment=True`
  Return edge-level support and ownership outputs in the graph sections.
- `return_cellular_decomposition=True`
  Return a mixed disjoint partition of all points onto simplified node cells
  and simplified edge cells. Requires `simplify=True`.
- `return_intermediates=True`
  Return matrices and arrays needed for debugging or plotting.

## Common Patterns

Simplified graph plus raw graph:

```python
result = dreeb(X, return_raw=True)

simplified = result["simplified"]
raw = result["raw"]
```

Simplified graph plus persistence:

```python
result = dreeb(X, return_simp_persistence=True)

h1 = result["simplified"]["persistence"]["h1"]
```

Use the second nontrivial diffusion eigenfunction:

```python
result = dreeb(X, diffusion_eigen_index=2)
```

Simplified graph plus node assignment:

```python
result = dreeb(X, return_point_assignment=True)

point_assignment = result["simplified"]["point_assignment"]
node_points = result["simplified"]["node_points"]
node_support_points = result["simplified"]["node_support_points"]
```

Simplified graph plus edge assignment:

```python
result = dreeb(X, return_edge_assignment=True)

edge_points = result["simplified"]["edge_points"]
edge_support_points = result["simplified"]["edge_support_points"]
point_edge_assignment = result["simplified"]["point_edge_assignment"]
point_edges = result["simplified"]["point_edges"]
```

Simplified graph plus cellular decomposition:

```python
result = dreeb(X, return_cellular_decomposition=True)

cell_kind = result["simplified"]["cell_assignment_kind"]
cell_id = result["simplified"]["cell_assignment_id"]
node_cells = result["simplified"]["node_cells"]
edge_cells = result["simplified"]["edge_cells"]
```

Raw persistence without returning raw edges explicitly:

```python
result = dreeb(X, return_raw_persistence=True)

raw_pd = result["raw"]["persistence"]
```

## Point Assignment

Point assignment is opt-in via `return_point_assignment=True`.

There are two related but different node-level outputs:

- `node_points`: an ownership partition, one simplified/raw node id per point
- `node_support_points`: the intrinsic local support of each node

If you want the points supporting a trajectory through the simplified graph,
use `edge_support_points` together with a graph path in
`result["simplified"]["edges"]` and take the union of the edge supports along
that path.

When `simplify=True`:

- `node_support_points` are inherited from the kept raw nodes
- `node_points` are a full ownership partition of the dataset onto simplified
  nodes
- ownership is computed by graph Voronoi on the original point graph using
  shortest-path distance on the symmetrized diffusion operator

When `simplify=False`:

- points are assigned to raw Reeb nodes

Returned fields:

- `point_assignment`: node index per point
- `node_points`: ownership partition of points per node
- `node_support_points`: intrinsic support points per node

## Edge Assignment

Edge assignment is opt-in via `return_edge_assignment=True`.

Returned fields:

- `edge_points`: ownership partition of points per edge
- `point_edge_assignment`: one edge id per point
- `edge_support_points`: intrinsic support points per edge
- `point_edges`: overlapping edge-support memberships per point

Raw edge assignment rule:

- for a raw edge connecting `(step s, comp c)` to `(step s+1, comp c_next)`,
  points are assigned if they lie in the connected bridging region across
  levels `s, s+1, s+2` that touches both endpoint node supports

Simplified edge assignment rule:

- simplified edge support is inherited from the contracted raw path behind that
  simplified edge
- inherited support includes raw edge supports on that path plus the support
  points of contracted interior raw nodes
- simplified edge ownership is a full partition of the dataset onto simplified
  edges
- ownership is computed by graph Voronoi on the original point graph using
  shortest-path distance on the symmetrized diffusion operator

## Cellular Decomposition

Cellular decomposition is opt-in via `return_cellular_decomposition=True` and
is only defined on the simplified graph.

It is a mixed disjoint partition of the whole dataset onto:

- simplified node cells, corresponding to 0-cells
- simplified edge cells, corresponding to 1-cells

Returned fields:

- `cell_assignment_kind`: per-point `"node"` or `"edge"`
- `cell_assignment_id`: per-point simplified node id or simplified edge id
- `node_cells`: list of point index arrays, one per simplified node
- `edge_cells`: list of point index arrays, one per simplified edge

This decomposition uses the same graph metric as the ownership assignments:
shortest-path distance on the symmetrized diffusion operator, comparing node
supports against edge supports directly.

Helper for extracting points from selected cells:

```python
from dreeb import extract_cellular_trajectory

result = dreeb(X, return_cellular_decomposition=True)
simplified = result["simplified"]

# Mixed cell indexing: nodes first, then edges.
points = extract_cellular_trajectory(simplified, cell_indices=[0, 3, 7])
```

If there are `n` simplified nodes, then:

- cell indices `0 .. n-1` are node cells
- cell indices `n .. n + m - 1` are edge cells

Helper for enumerating terminal cellular trajectories:

```python
from dreeb import enumerate_terminal_cellular_trajectories

result = dreeb(
    X,
    return_edge_lengths=True,
    return_cellular_decomposition=True,
)
trajectories = enumerate_terminal_cellular_trajectories(result["simplified"])

top = trajectories[0]
top["node_path"]
top["edge_path"]
top["cell_indices"]
top["graph_length"]
```

Here, a terminal simplified node means a node whose multigraph degree is not
equal to `2`, so trajectories run between leaf tips, branch points, isolated
nodes, or loop anchors.

## Edge Lengths

Edge lengths are opt-in via `return_edge_lengths=True`.

- `result["simplified"]["edge_lengths"]`: one filter-space length per simplified edge
- `result["raw"]["edge_lengths"]`: one filter-space length per raw edge

For an edge `(u, v)`, the length is the absolute difference between the
midpoint slice values of its endpoint nodes.

## Persistence

Persistence is opt-in via `return_simp_persistence=True` and
`return_raw_persistence=True`.

This is graph-filtration persistence on the returned multigraph:

- all vertices are born at filtration value `0`
- edges are added in nondecreasing order of edge length
- an edge that merges two connected components creates an `H0` death
- an edge that closes a cycle creates an essential `H1` class with death `inf`

Returned dictionary format:

- `h0`: finite `H0` pairs with shape `(m, 2)`
- `h0_essential`: essential `H0` pairs with shape `(c, 2)`
- `h1`: essential `H1` pairs with shape `(k, 2)`

Example:

```python
result = dreeb(X, return_simp_persistence=True)
pd = result["simplified"]["persistence"]

h0 = pd["h0"]
h1 = pd["h1"]
```

This is not higher-dimensional simplicial persistence; it is persistence of the
edge-weighted graph itself.

## Plotting

`plot_dreeb()` overlays the simplified graph on a 2D or 3D embedding that you
compute separately. It works with the standard package install; there is no
separate plotting extra or second install step.

```python
from dreeb import dreeb, plot_dreeb

result = dreeb(X, return_raw=True, return_intermediates=True)

fig, node_pos = plot_dreeb(
    pts=embedding_2d,
    simp_edges=result["simplified"]["edges"],
    keep_ids=result["simplified"]["keep_ids"],
    reeb_nodes=result["raw"]["nodes"],
    step_vertices=result["intermediates"]["step_vertices"],
    step_comp_ids=result["intermediates"]["step_comp_ids"],
    uniq_v=result["intermediates"]["prep_state"]["uniq_v"],
)
```

Practical notes:

- `pts` must be a 2D or 3D embedding in the same row order as `X`
- 2D multiedges are drawn as curved arcs so parallel edges are legible
- cycle edges are colored crimson when `color_cycles=True`
- node labels show multigraph degree when `label_degree=True`
- the function returns `(fig, node_pos)` so you can save or modify the figure

## Index Alignment

`plot_dreeb()` assumes the embedding array `pts` is aligned row-for-row with the
original data matrix `X` passed to `dreeb()`.

How node membership is recovered for plotting:

- each raw Reeb node is identified by `(step, comp)` in `reeb_nodes`
- `step_vertices[step]` gives the compacted active vertex indices in that slice
- `step_comp_ids[step]` gives the component id for each active vertex
- compacted indices map back to original point indices through
  `prep_state["uniq_v"]`
- those original indices are used to index into `pts` and compute node centroids

If you shuffle or subset your data before embedding, apply the same
transformation to both `X` and `pts`, or remap indices accordingly.

## Workflow Guide

Typical calls:

- topology only:
  `result = dreeb(X)`
- topology plus raw graph:
  `result = dreeb(X, return_raw=True)`
- topology plus persistence:
  `result = dreeb(X, return_simp_persistence=True)`
- topology plus node ownership:
  `result = dreeb(X, return_point_assignment=True)`
- topology plus edge ownership:
  `result = dreeb(X, return_edge_assignment=True)`
- topology plus plotting support:
  `result = dreeb(X, return_raw=True, return_intermediates=True)`
