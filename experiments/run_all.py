"""Run every experiment of the README and regenerate its figures and numbers.

    python experiments/run_all.py

Writes the figures to ``docs/figures/`` and the numbers to ``docs/results.json``.
Takes well under a minute on a laptop; ``--quick`` runs a shortened version
(used by the CI) and writes nothing unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from admm_svm import (
    SVMProblem,
    accuracy,
    admm_centralized,
    admm_distributed,
    graph,
    load_banknote,
    reference_solution,
    subgradient_descent,
)

ROOT = Path(__file__).resolve().parents[1]

LAMBDA = 0.01
SUBGRADIENT_STEP = 3.0
COLORS = {"subgradient": "#8c8c8c", "centralized": "#1f6fb4", "distributed": "#d1495b"}
NODE_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
TOPOLOGIES = {
    "path": graph.path(5),
    "project tree": graph.PROJECT_ADJACENCY,
    "star": graph.star(5),
    "ring": graph.ring(5),
    "complete": graph.complete(5),
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)


def iterations_to(gap: np.ndarray, tol: float) -> int | None:
    """First iteration at which the suboptimality drops to ``tol`` (None if never)."""
    hits = np.nonzero(gap <= tol)[0]
    return int(hits[0]) if len(hits) else None


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


# Hand-placed layout of the project network:  0 - 3 - 4 - 1, with 2 hanging from 4.
NODE_POSITIONS = {0: (-1.2, 0.3), 3: (-0.4, 0.3), 4: (0.4, 0.3), 1: (1.2, 0.3), 2: (0.4, -0.6)}


def draw_network(ax, adj, blocks, y, title):
    """Draw the project network; each node is a pie of the two classes it holds."""
    for i, j in graph.edges(adj):
        (x0, y0), (x1, y1) = NODE_POSITIONS[i], NODE_POSITIONS[j]
        ax.plot([x0, x1], [y0, y1], color="#555555", lw=2.2, zorder=1)
    for n, (px, py) in NODE_POSITIONS.items():
        forged = float(np.mean(y[blocks[n]] > 0))
        shares = [(forged, "#c44e52"), (1 - forged, "#55a868")]
        ax.pie(
            [share for share, _ in shares if share > 0],
            colors=[color for share, color in shares if share > 0],
            center=(px, py),
            radius=0.2,
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        )
        below = n == 2
        ax.text(
            px,
            py + (-0.27 if below else 0.27),
            f"node {n}\n{len(blocks[n])} notes",
            ha="center",
            va="top" if below else "bottom",
            fontsize=8.5,
        )
    ax.set_title(title)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.45, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#55a868", label="genuine"),
        plt.Line2D([], [], marker="o", ls="", color="#c44e52", label="forged"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: docs/)")
    args = parser.parse_args()

    n_iter = 300 if args.quick else 2000
    out = args.out if args.out else (None if args.quick else ROOT / "docs")
    figures = out / "figures" if out else None
    if figures:
        figures.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        if figures:
            fig.savefig(figures / name)
        plt.close(fig)

    data = load_banknote()
    problem = data.problem(LAMBDA)
    N = problem.n_samples
    adj = graph.PROJECT_ADJACENCY
    iid_blocks = graph.split_samples(N, 5)
    # Label-skewed split: sort by class before dealing, so most nodes see one class only.
    skewed_blocks = np.array_split(np.argsort(data.y_train, kind="stable"), 5)

    # ------------------------------------------------------------------ optimum
    x_star, f_sklearn = reference_solution(problem)
    polish = admm_centralized(problem, n_iter=200 if args.quick else 20000, rho=0.5)
    f_star = min(f_sklearn, polish.objective.min())
    print(f"f* = {f_star:.10f}   (scikit-learn: {f_sklearn:.10f})")

    # ------------------------------------------------------ the three algorithms
    sub, t_sub = timed(subgradient_descent, problem, n_iter, SUBGRADIENT_STEP, "sqrt")
    cen, t_cen = timed(admm_centralized, problem, n_iter)
    dis, t_dis = timed(admm_distributed, problem, adj, n_iter, keep_history=True)
    skew = admm_distributed(problem, adj, n_iter, blocks=skewed_blocks, keep_history=True)
    runs = {"subgradient": (sub, t_sub), "centralized": (cen, t_cen), "distributed": (dis, t_dis)}
    labels = {
        "subgradient": "Subgradient descent",
        "centralized": "ADMM (centralized)",
        "distributed": "ADMM (distributed, 5 nodes)",
    }

    summary = {}
    for key, (run, seconds) in runs.items():
        gap = run.objective - f_star
        summary[key] = {
            "objective": run.objective[-1] if key != "subgradient" else run.best_objective[-1],
            "gap_final": float(run.best_objective[-1] - f_star),
            "iters_to_1e-3": iterations_to(gap, 1e-3),
            "iters_to_1e-5": iterations_to(gap, 1e-5),
            "iters_to_1e-7": iterations_to(gap, 1e-7),
            "train_accuracy": accuracy(run.x, data.A_train, data.y_train),
            "test_accuracy": accuracy(run.x, data.A_test, data.y_test),
            "ms_per_iteration": 1000 * seconds / n_iter,
        }
        print(key, summary[key])

    # ------------------------------------------------------------- hero figure
    fig = plt.figure(figsize=(11.5, 4.3))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.55], wspace=0.3)
    draw_network(
        fig.add_subplot(grid[0]), adj, iid_blocks, data.y_train, "Five nodes, each with its own data"
    )
    ax = fig.add_subplot(grid[1])
    k = np.arange(n_iter + 1)
    ax.semilogy(k, sub.objective - f_star, color=COLORS["subgradient"], lw=0.6, alpha=0.45)
    ax.semilogy(
        k, sub.best_objective - f_star, color=COLORS["subgradient"], lw=2, label=labels["subgradient"]
    )
    ax.semilogy(
        k, cen.objective - f_star, color=COLORS["centralized"], lw=2, label=labels["centralized"]
    )
    ax.semilogy(
        k, dis.objective - f_star, color=COLORS["distributed"], lw=2, label=labels["distributed"]
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"distance to the optimum,  $f(x_k) - f^\star$")
    ax.set_title("All three reach the same classifier")
    ax.set_ylim(1e-10, 2)
    ax.set_xlim(0, n_iter)
    ax.legend(loc="upper right")
    save(fig, "overview.png")

    # --------------------------------------------------------- consensus figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    draw_network(axes[0], adj, skewed_blocks, data.y_train, "Label-skewed split")
    shown = min(n_iter, 400)
    for n in range(5):
        node_gap = [problem.objective(x) - f_star for x in skew.node_history[: shown + 1, n]]
        axes[1].semilogy(node_gap, color=NODE_COLORS[n], lw=1.6, label=f"node {n}")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$f(x_n) - f^\star$ on the full training set")
    axes[1].set_title("Every node converges to the global model")
    axes[1].legend(ncol=2, fontsize=8.5)
    # the nodes only start to differ at the second iteration (they all start from zero)
    axes[2].semilogy(k[2:], dis.consensus_error[2:], color="#2a9d8f", lw=2, label="even split")
    axes[2].semilogy(k[2:], skew.consensus_error[2:], color="#e76f51", lw=2, label="label-skewed split")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel(r"disagreement,  $\max_n \|x_n - \bar{x}\|$")
    axes[2].set_title("Nodes agree with each other")
    axes[2].legend()
    fig.tight_layout()
    save(fig, "consensus.png")

    # What can a node do on its own?  (even split: fine but suboptimal; skewed: nothing)
    alone = []
    for idx in iid_blocks:
        x_local, _ = reference_solution(SVMProblem(data.A_train[idx], data.y_train[idx], LAMBDA))
        alone.append(accuracy(x_local, data.A_test, data.y_test))
    single_class = sum(len(np.unique(data.y_train[idx])) == 1 for idx in skewed_blocks)
    skew_gap = skew.objective - f_star

    # ----------------------------------------------------------- rho sensitivity
    rhos = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0] if not args.quick else [1.0, 10.0]
    consensus_rhos = [0.002, 0.005, 0.02, 0.05, 0.2, 1.0] if not args.quick else [0.02, 0.2]
    sweep_iter = 300 if args.quick else 5000
    tol = 1e-5
    cen_sweep = [
        iterations_to(admm_centralized(problem, sweep_iter, r).objective - f_star, tol) for r in rhos
    ]
    dis_sweep = [
        iterations_to(admm_distributed(problem, adj, sweep_iter, 1.0, rc).objective - f_star, tol)
        for rc in consensus_rhos
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, xs, ys, color, xlabel, title in [
        (axes[0], rhos, cen_sweep, COLORS["centralized"], r"data penalty $\rho$", "Centralized ADMM"),
        (
            axes[1],
            consensus_rhos,
            dis_sweep,
            COLORS["distributed"],
            r"consensus penalty $\rho_c$   ($\rho = 1$)",
            "Distributed ADMM",
        ),
    ]:
        ys_plot = [y if y is not None else sweep_iter for y in ys]
        ax.loglog(xs, ys_plot, "o-", color=color, lw=2)
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(
                str(y) if y is not None else f">{sweep_iter}",
                (x, y if y is not None else sweep_iter),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8.5,
            )
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].set_ylabel(rf"iterations to reach $f - f^\star \leq 10^{{{int(np.log10(tol))}}}$")
    fig.tight_layout()
    save(fig, "rho_sensitivity.png")

    # ------------------------------------------------- other network topologies
    topo_iter = 300 if args.quick else 3000
    topo = {}
    for name, a in TOPOLOGIES.items():
        run = admm_distributed(problem, a, topo_iter, blocks=skewed_blocks)
        topo[name] = {
            "edges": len(graph.edges(a)),
            "algebraic_connectivity": graph.algebraic_connectivity(a),
            "iters_to_1e-3": iterations_to(run.objective - f_star, 1e-3),
            "iters_to_1e-5": iterations_to(run.objective - f_star, 1e-5),
        }

    results = {
        "problem": {
            "n_train": N,
            "n_test": len(data.y_test),
            "lambda": LAMBDA,
            "f_star": f_star,
            "f_sklearn": f_sklearn,
            "test_accuracy_optimum": accuracy(x_star, data.A_test, data.y_test),
            "n_iter": n_iter,
        },
        "algorithms": summary,
        "nodes_alone": {
            "test_accuracy_even_split": alone,
            "single_class_nodes_in_skewed_split": int(single_class),
            "skewed_iters_to_1e-5": iterations_to(skew_gap, 1e-5),
            "skewed_test_accuracy": accuracy(skew.x, data.A_test, data.y_test),
            "skewed_node_spread_final": float(skew.consensus_error[-1]),
        },
        "rho_sweep": {
            "tolerance": tol,
            "centralized": dict(zip(map(str, rhos), cen_sweep, strict=True)),
            "distributed_rho_consensus": dict(zip(map(str, consensus_rhos), dis_sweep, strict=True)),
        },
        "topologies": topo,
    }
    print(json.dumps({k: v for k, v in results.items() if k != "algorithms"}, indent=2))
    if out:
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote figures and results.json to {out}")


if __name__ == "__main__":
    main()
