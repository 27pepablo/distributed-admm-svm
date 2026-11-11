"""Three solvers for the same SVM problem.

* :func:`subgradient_descent` - first-order baseline.
* :func:`admm_centralized`    - ADMM with all the data in one place.
* :func:`admm_distributed`    - consensus ADMM: the data is split across the
  nodes of a network and nodes only talk to their neighbours.

All three minimise exactly the same objective (see :mod:`admm_svm.problem`), so
their iterates can be compared against a single optimum.

The derivations are in ``docs/derivations.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import graph
from .problem import SVMProblem, prox_hinge


@dataclass
class Result:
    """Output of a solver.

    ``objective[k]`` is f at the k-th iterate (index 0 is the starting point).
    For the distributed solver the iterate is the average of the node models,
    and ``consensus_error[k]`` is the largest distance from a node to that average.
    """

    x: np.ndarray
    objective: np.ndarray
    consensus_error: np.ndarray | None = None
    node_x: np.ndarray | None = None  # (n_nodes, dim) final model of every node
    node_history: np.ndarray | None = None  # (n_iter + 1, n_nodes, dim)
    info: dict = field(default_factory=dict)

    @property
    def best_objective(self) -> np.ndarray:
        """Best objective value found up to each iteration."""
        return np.minimum.accumulate(self.objective)


def subgradient_descent(
    problem: SVMProblem,
    n_iter: int = 1000,
    step: float = 1.0,
    schedule: str = "sqrt",
) -> Result:
    """Subgradient descent, ``x <- x - mu_k * g_k``.

    ``schedule`` picks the step sizes: ``"sqrt"`` uses ``mu_k = step / sqrt(k + 1)``
    (converges, slowly), ``"constant"`` uses ``mu_k = step`` (stalls at a distance
    from the optimum that depends on the step).

    The objective is not monotone, so the best iterate seen is returned.
    """
    if schedule not in ("sqrt", "constant"):
        raise ValueError("schedule must be 'sqrt' or 'constant'")
    x = np.zeros(problem.dim)
    best_x, best_f = x.copy(), problem.objective(x)
    objective = np.empty(n_iter + 1)
    objective[0] = best_f
    for k in range(n_iter):
        mu = step / np.sqrt(k + 1) if schedule == "sqrt" else step
        x = x - mu * problem.subgradient(x)
        objective[k + 1] = f = problem.objective(x)
        if f < best_f:
            best_x, best_f = x.copy(), f
    return Result(x=best_x, objective=objective, info={"step": step, "schedule": schedule})


def admm_centralized(problem: SVMProblem, n_iter: int = 1000, rho: float = 1.0) -> Result:
    """ADMM on the splitting ``s = B x`` (one auxiliary variable per margin).

    Each iteration is closed-form:

    * ``x``: a small linear system whose matrix never changes (factorised once),
    * ``s``: the proximal operator of the hinge loss,
    * ``u``: running sum of the constraint violation ``B x - s``.

    ``rho`` is the weight of the penalty ``(rho / 2N) * ||B x - s||^2``. Any positive
    value converges; it only changes the speed.
    """
    B, N = problem.B, problem.n_samples
    reg = problem.lam * problem.reg_diag
    chol = cho_factor(np.diag(reg) + (rho / N) * B.T @ B)

    x = np.zeros(problem.dim)
    s = np.zeros(N)
    u = np.zeros(N)
    objective = np.empty(n_iter + 1)
    objective[0] = problem.objective(x)
    for k in range(n_iter):
        x = cho_solve(chol, (rho / N) * B.T @ (s - u))
        Bx = B @ x
        s = prox_hinge(Bx + u, 1.0 / rho)
        u = u + Bx - s
        objective[k + 1] = problem.objective(x)
    return Result(x=x, objective=objective, info={"rho": rho})


def admm_distributed(
    problem: SVMProblem,
    adjacency: np.ndarray = graph.PROJECT_ADJACENCY,
    n_iter: int = 1000,
    rho: float = 1.0,
    rho_consensus: float = 0.02,
    blocks: list[np.ndarray] | None = None,
    keep_history: bool = False,
) -> Result:
    """Consensus ADMM over a network.

    Node ``n`` owns the samples ``blocks[n]`` and its own copy ``x_n`` of the model.
    Neighbouring nodes are tied together through one shared variable ``z_e`` per
    edge. In every iteration a node

    1. solves a small linear system using only its own data and the latest
       ``z_e`` of its edges,
    2. sends the new ``x_n`` to its neighbours,
    3. agrees with each neighbour on ``z_e`` (the average of the two ends).

    No node ever sees another node's data, and there is no central coordinator.

    By default the samples are dealt to the nodes in equal contiguous blocks.

    ``rho`` plays the same role as in :func:`admm_centralized`; ``rho_consensus``
    is the weight of the penalty ``(rho_consensus / 2) * ||x_n - z_e||^2`` that pulls
    a node towards each of its neighbours.
    """
    adj = graph.validate(adjacency)
    n_nodes = adj.shape[0]
    edge_list = graph.edges(adj)
    incident = [[e for e, (i, j) in enumerate(edge_list) if n in (i, j)] for n in range(n_nodes)]
    if blocks is None:
        blocks = graph.split_samples(problem.n_samples, n_nodes)
    if len(blocks) != n_nodes:
        raise ValueError("need one block of samples per node")

    B, N, p = problem.B, problem.n_samples, problem.dim
    rho_c = rho_consensus
    B_loc = [B[idx] for idx in blocks]
    # The regulariser is shared equally, so that the local objectives add up to f.
    reg = problem.lam * problem.reg_diag / n_nodes
    chol = [
        cho_factor(np.diag(reg) + (rho / N) * Bn.T @ Bn + rho_c * len(incident[n]) * np.eye(p))
        for n, Bn in enumerate(B_loc)
    ]

    x = np.zeros((n_nodes, p))
    s = [np.zeros(len(idx)) for idx in blocks]  # local margins
    u = [np.zeros(len(idx)) for idx in blocks]  # dual of  B_n x_n = s_n
    z = np.zeros((len(edge_list), p))  # one shared variable per edge
    w = np.zeros((n_nodes, len(edge_list), p))  # dual of  x_n = z_e  (used where e touches n)

    objective = np.empty(n_iter + 1)
    consensus = np.empty(n_iter + 1)
    history = np.empty((n_iter + 1, n_nodes, p)) if keep_history else None

    def log(k: int) -> None:
        x_bar = x.mean(axis=0)
        objective[k] = problem.objective(x_bar)
        consensus[k] = np.linalg.norm(x - x_bar, axis=1).max()
        if history is not None:
            history[k] = x

    log(0)
    for k in range(n_iter):
        # 1) every node updates its model (in parallel in a real deployment)
        for n in range(n_nodes):
            pull_data = (rho / N) * B_loc[n].T @ (s[n] - u[n])
            pull_neighbours = rho_c * sum(z[e] - w[n, e] for e in incident[n])
            x[n] = cho_solve(chol[n], pull_data + pull_neighbours)
        # 2) local margins and their duals
        for n in range(n_nodes):
            Bx = B_loc[n] @ x[n]
            s[n] = prox_hinge(Bx + u[n], 1.0 / rho)
            u[n] = u[n] + Bx - s[n]
        # 3) neighbours agree on the edge variables, then update the consensus duals
        for e, (i, j) in enumerate(edge_list):
            z[e] = 0.5 * (x[i] + w[i, e] + x[j] + w[j, e])
            w[i, e] += x[i] - z[e]
            w[j, e] += x[j] - z[e]
        log(k + 1)

    return Result(
        x=x.mean(axis=0),
        objective=objective,
        consensus_error=consensus,
        node_x=x.copy(),
        node_history=history,
        info={
            "rho": rho,
            "rho_consensus": rho_c,
            "n_nodes": n_nodes,
            "n_edges": len(edge_list),
        },
    )
