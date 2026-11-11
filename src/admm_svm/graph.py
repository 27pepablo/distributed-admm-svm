"""Communication networks: who can talk to whom.

A network is described by a symmetric 0/1 adjacency matrix. Node ``i`` and node
``j`` may exchange messages only if ``adj[i, j] == 1``.
"""

from __future__ import annotations

import numpy as np

# The 5-node network assigned in the course project: a tree with node 4 as hub
# and node 0 two hops away from it.
#
#     0 --- 3 --- 4 --- 1
#                 |
#                 2
PROJECT_ADJACENCY = np.array(
    [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ]
)


def path(n: int) -> np.ndarray:
    """Nodes in a line: 0 - 1 - ... - (n-1)."""
    adj = np.zeros((n, n), dtype=int)
    idx = np.arange(n - 1)
    adj[idx, idx + 1] = adj[idx + 1, idx] = 1
    return adj


def ring(n: int) -> np.ndarray:
    """A path whose two ends are also connected."""
    adj = path(n)
    adj[0, n - 1] = adj[n - 1, 0] = 1
    return adj


def star(n: int) -> np.ndarray:
    """Node 0 connected to every other node."""
    adj = np.zeros((n, n), dtype=int)
    adj[0, 1:] = adj[1:, 0] = 1
    return adj


def complete(n: int) -> np.ndarray:
    """Every pair of nodes connected."""
    return np.ones((n, n), dtype=int) - np.eye(n, dtype=int)


def validate(adj: np.ndarray) -> np.ndarray:
    """Return ``adj`` as a clean 0/1 integer matrix, or raise if it is not a valid network."""
    adj = (np.asarray(adj) != 0).astype(int)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if not np.array_equal(adj, adj.T):
        raise ValueError("adjacency matrix must be symmetric")
    if np.any(np.diag(adj)):
        raise ValueError("adjacency matrix must have a zero diagonal")
    if not is_connected(adj):
        raise ValueError("the network must be connected")
    return adj


def edges(adj: np.ndarray) -> list[tuple[int, int]]:
    """Undirected edges as ``(i, j)`` pairs with ``i < j``."""
    i, j = np.nonzero(np.triu(adj, k=1))
    return list(zip(i.tolist(), j.tolist(), strict=True))


def is_connected(adj: np.ndarray) -> bool:
    """True if every node can reach every other node."""
    n = adj.shape[0]
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for nb in np.nonzero(adj[node])[0].tolist():
            if nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    return len(seen) == n


def algebraic_connectivity(adj: np.ndarray) -> float:
    """Second-smallest eigenvalue of the graph Laplacian.

    Larger means better connected: information needs fewer hops to spread, and
    consensus algorithms converge faster.
    """
    laplacian = np.diag(adj.sum(axis=1)) - adj
    return float(np.linalg.eigvalsh(laplacian)[1])


def split_samples(n_samples: int, n_nodes: int) -> list[np.ndarray]:
    """Deal ``n_samples`` row indices into ``n_nodes`` contiguous, near-equal blocks."""
    return np.array_split(np.arange(n_samples), n_nodes)
