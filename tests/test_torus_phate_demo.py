import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import phate

from dreeb import dreeb
from dreeb.visualize import plot_reeb


def sample_torus(n=800, R=2.0, r=0.7, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    return np.column_stack([x, y, z])


def run_torus_demo(output_path="artifacts/torus_phate_reeb.png"):
    X = sample_torus()

    phate_op = phate.PHATE(n_components=2, random_state=0, knn=15, verbose=False)
    pts = phate_op.fit_transform(X)

    result = dreeb(
        X,
        k=30,
        precision=1.0,
        simplify=True,
        return_raw=True,
        return_simp_persistence=True,
        return_intermediates=True,
    )

    simplified = result["simplified"]
    raw = result["raw"]
    intermediates = result["intermediates"]

    fig, _ = plot_reeb(
        pts=pts,
        simp_edges=simplified["edges"],
        keep_ids=simplified["keep_ids"],
        reeb_nodes=raw["nodes"],
        step_vertices=intermediates["step_vertices"],
        step_comp_ids=intermediates["step_comp_ids"],
        uniq_v=intermediates["prep_state"]["uniq_v"],
        color_cycles=True,
        label_degree=True,
        figsize=(8, 6),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    persistence = simplified["persistence"]
    pd_output_path = "artifacts/torus_simplified_persistence.png"
    plot_persistence_diagram(persistence, pd_output_path)

    print("primary_graph:", result["primary_graph"])
    print("simplified nodes:", len(simplified["nodes"]))
    print("simplified edges:", len(simplified["edges"]))
    print("beta1:", simplified["beta1"])
    print("components:", simplified["comp_count"])
    print("simplified persistence h0:")
    print(persistence["h0"])
    print("simplified persistence h0_essential:")
    print(persistence["h0_essential"])
    print("simplified persistence h1:")
    print(persistence["h1"])
    print("plot saved to:", output_path)
    print("persistence diagram saved to:", pd_output_path)

    assert result["primary_graph"] == "simplified"
    assert "persistence" in simplified
    assert pts.shape == (X.shape[0], 2)
    assert len(simplified["edges"]) >= 1
    assert persistence["h0"].shape[1] == 2
    assert persistence["h0_essential"].shape[1] == 2
    assert persistence["h1"].shape[1] == 2

    return {
        "output_path": output_path,
        "persistence_output_path": pd_output_path,
        "result": result,
        "embedding": pts,
    }


def plot_persistence_diagram(persistence, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    finite_vals = []
    for key in ("h0", "h1"):
        arr = persistence[key]
        if arr.size == 0:
            continue
        finite_vals.extend(arr[np.isfinite(arr)].ravel().tolist())

    max_finite = max(finite_vals) if finite_vals else 1.0
    inf_y = max_finite * 1.1 + 1e-6

    h0 = persistence["h0"]
    if h0.size > 0:
        ax.scatter(h0[:, 0], h0[:, 1], color="tab:blue", s=36, label="H0")

    h1 = persistence["h1"]
    if h1.size > 0:
        y = np.where(np.isfinite(h1[:, 1]), h1[:, 1], inf_y)
        ax.scatter(h1[:, 0], y, color="tab:red", marker="x", s=48, label="H1")

    h0e = persistence["h0_essential"]
    if h0e.size > 0:
        y = np.where(np.isfinite(h0e[:, 1]), h0e[:, 1], inf_y)
        ax.scatter(h0e[:, 0], y, color="tab:green", marker="^", s=44, label="H0 essential")

    lo = 0.0
    hi = max(max_finite, inf_y)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.0)
    ax.axhline(inf_y, linestyle=":", color="gray", linewidth=1.0)
    ax.text(lo, inf_y, "inf", va="bottom", ha="left", fontsize=9)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, inf_y * 1.02)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Simplified Persistence Diagram")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_torus_demo()
