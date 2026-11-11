import numpy as np
import pytest

from admm_svm import graph


def test_project_network():
    adj = graph.validate(graph.PROJECT_ADJACENCY)
    assert graph.edges(adj) == [(0, 3), (1, 4), (2, 4), (3, 4)]
    assert graph.is_connected(adj)


@pytest.mark.parametrize(
    "builder, n_edges", [(graph.path, 5), (graph.ring, 6), (graph.star, 5), (graph.complete, 15)]
)
def test_standard_topologies(builder, n_edges):
    adj = graph.validate(builder(6))
    assert len(graph.edges(adj)) == n_edges


def test_algebraic_connectivity_known_values():
    assert graph.algebraic_connectivity(graph.complete(5)) == pytest.approx(5.0)
    assert graph.algebraic_connectivity(graph.star(5)) == pytest.approx(1.0)
    # more links, better connected
    assert graph.algebraic_connectivity(graph.path(5)) < graph.algebraic_connectivity(graph.ring(5))


def test_invalid_networks_are_rejected():
    two_islands = np.zeros((4, 4), dtype=int)
    two_islands[0, 1] = two_islands[1, 0] = two_islands[2, 3] = two_islands[3, 2] = 1
    one_way = np.array([[0, 1], [0, 0]])
    self_loop = np.array([[1, 1], [1, 0]])
    for bad in (two_islands, one_way, self_loop, np.zeros((2, 3))):
        with pytest.raises(ValueError):
            graph.validate(bad)


def test_split_samples_covers_everything_once():
    blocks = graph.split_samples(503, 5)
    assert sorted(np.concatenate(blocks).tolist()) == list(range(503))
    assert max(map(len, blocks)) - min(map(len, blocks)) <= 1
