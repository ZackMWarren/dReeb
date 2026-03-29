# Memory

- `dreeb()` now returns a structured result with stable top-level sections:
  `primary_graph`, `simplified`, `raw`, and optional `intermediates`.
- Default behavior is computationally cheap: simplified graph only.
- Optional expensive outputs are flag-driven:
  `return_raw`, `return_edge_lengths`, `return_simp_persistence`,
  `return_raw_persistence`, `return_point_assignment`,
  `return_edge_assignment`, `return_intermediates`.
- Point assignment and edge assignment are different:
  node assignment is total-by-default when requested;
  edge assignment is support-based and may be sparse.
- Raw edge assignment rule:
  for raw edge `(step s, comp c) -> (step s+1, comp c_next)`, assign points in
  the bridging region across levels `s, s+1, s+2` that touches both endpoint
  node supports.
- Simplified edge assignment is built by aggregating raw edge supports along the
  contracted raw-edge path behind each simplified edge.
- Persistence is graph-filtration persistence on the returned multigraph,
  using edge lengths as filtration values.
- `plot_reeb()` uses curved arcs for parallel 2D multiedges, keeps self-loops,
  colors cycle edges crimson, and returns `(fig, node_pos)`.
- Good demo setting found in-session:
  double offset tori with 1500 samples each and `k=30` gives a much cleaner
  simplified graph than `k=15`.
