# Memory

Project: dReeb package, tutorial notebook, and benchmarking pipeline.

## Current benchmark assets
- Graphs: `artifacts/benchmark_graphs/*.json` + `*.png` (Graphviz layouts). Includes added `torus` entry.
- Point clouds: `artifacts/benchmark_pointclouds/*_points_2d.png`, `*_points_2d.npy`, `*_highd.npy`, `*_phate.png`.
  - Density: quarter of earlier; `base_density=32.5`, `min_per_edge=18`, `noise_sigma=0.10` in `scripts/benchmark_pointclouds.py`.
  - Torus: generated as 3D torus, saved to `*_highd.npy` (3D), 2D view is PCA; not embedded to 20D.
- dReeb results: `artifacts/benchmark_dreeb/*_dreeb_simplified.npz` + `index.json` (comparison summary, topology_ok).
- dReeb overlays: `artifacts/benchmark_dreeb_overlays/*_dreeb_overlay.png` + `index.json`.
- PAGA graphs: `artifacts/benchmark_paga/*_paga.png` + `index.json` (generated via scanpy, Graphviz sfdp).
- PAGA persistence: `artifacts/benchmark_paga_persistence/*_paga_pd.npz` + `index.json` (comparisons, topology_ok).
- PDFs:
  - dReeb summary: `artifacts/benchmark_summary.pdf`
  - PAGA summary: `artifacts/benchmark_summary_paga.pdf`
  - Both include a Topology column (green/red) based on H1 count match; torus uses “H1 == 1” rule.

## Key scripts
- `scripts/benchmark_graphs.py`: builds graph templates; now includes `torus` as a cycle graph for “Graph” column.
- `scripts/benchmark_pointclouds.py`: generates point clouds + PHATE + saves 2D/3D/highD.
- `scripts/benchmark_run_dreeb.py`: runs dReeb on `*_highd.npy`, computes persistence comparisons; currently set to `k=35` and now uses the default diffusion-eigenfunction filter.
- `scripts/benchmark_plot_dreeb_over_2d.py`: overlays simplified dReeb on 2D points; uses `k=35` and runs on `highd_npy` but plots `points_2d_npy`.
- `scripts/benchmark_paga_graphs.py`: runs scanpy PAGA; patched to disable numba caching (Dispatcher/UFuncBuilder/FunctionCache); uses Graphviz `sfdp`.
- `scripts/benchmark_paga_persistence.py`: computes PDs on PAGA graph edges using Graphviz layout distances.
- `scripts/benchmark_make_pdf.py` and `scripts/benchmark_make_pdf_paga.py`: generate PDFs with topology bar.

## Environment notes
- PAGA required conda env at `/home/drew/miniconda3/envs/paga/bin/python` with Python 3.10, scanpy, numpy, python-igraph, leidenalg.
- numba caching errors in scanpy/pynndescent worked around inside `scripts/benchmark_paga_graphs.py`.
- Running PAGA shows harmless warnings about `/dev/shm` permissions and joblib serial mode.

## Filter files
- Live filter in `src/dreeb/filter.py` is now the per-connected-component diffusion eigenfunction filter.
- `dreeb()` defaults to the first nontrivial diffusion eigenfunction and supports overriding via `diffusion_eigen_index`.
- Older diffusion-time / endpoint-style experiments remain in `src/dreeb/filterT.py` as a saved alternate file, not the active pipeline.

## API notes
- Public plotting entrypoint is now `plot_dreeb` (with `plot_reeb` kept only as a compatibility alias).
- The repo install story in `README.md` is now simplified around `pip install dreeb`.
- Simplified node outputs distinguish:
  - `node_points`: ownership partition
  - `node_support_points`: intrinsic local node support
- Trajectory extraction is documented as unioning `edge_points` along a path in the simplified graph.

## Tutorial notebook
- User-facing notebook added at `notebooks/dreeb_tutorial.ipynb`.
- Notebook sections:
  - torus with default filter, 3D/2D overlays, persistence
  - hollow cylinder comparing `diffusion_eigen_index=1` vs `2`
  - sun point cloud with node support, edge support, and trajectory extraction
- Notebook now patches `sys.path` to import the local repo from `src/` before the package is on PyPI.

## Last requested run
- dReeb benchmark rerun with `k=35` using the new default diffusion-eigenfunction filter, regenerated overlays, and rebuilt `artifacts/benchmark_summary.pdf`.
- Tutorial notebook, diffusion-filter workflow, plotting rename, and support-point semantics were committed and pushed to GitHub.
