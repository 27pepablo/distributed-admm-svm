import numpy as np
import pytest

from admm_svm import (
    accuracy,
    admm_centralized,
    admm_distributed,
    graph,
    load_banknote,
    reference_solution,
    subgradient_descent,
)


@pytest.fixture(scope="module")
def data():
    return load_banknote()


@pytest.fixture(scope="module")
def problem(data):
    return data.problem(lam=0.01)


@pytest.fixture(scope="module")
def optimum(problem):
    return reference_solution(problem)


def test_dataset_split(data):
    assert data.A_train.shape == (500, 4) and data.A_test.shape == (872, 4)
    assert set(np.unique(data.y_train)) == {-1.0, 1.0}
    assert np.allclose(data.A_train.mean(axis=0), 0) and np.allclose(data.A_train.std(axis=0), 1)


def test_reference_is_a_minimum(problem, optimum):
    x_star, f_star = optimum
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert problem.objective(x_star + 1e-3 * rng.normal(size=5)) >= f_star - 1e-9


def test_subgradient_descent_converges(problem, optimum):
    run = subgradient_descent(problem, n_iter=1000, step=3.0)
    assert run.best_objective[-1] - optimum[1] < 1e-5
    assert problem.objective(run.x) == pytest.approx(run.best_objective[-1])


def test_constant_step_stalls_above_the_optimum(problem, optimum):
    run = subgradient_descent(problem, n_iter=1000, step=1e-3, schedule="constant")
    assert run.best_objective[-1] - optimum[1] > 1e-2


def test_centralized_admm_matches_scikit_learn(problem, optimum):
    x_star, f_star = optimum
    run = admm_centralized(problem, n_iter=3000)
    assert abs(run.objective[-1] - f_star) < 1e-8
    assert np.allclose(run.x, x_star, atol=1e-3)


@pytest.mark.parametrize("rho", [0.1, 1.0, 10.0])
def test_centralized_admm_converges_for_any_rho(problem, optimum, rho):
    run = admm_centralized(problem, n_iter=3000, rho=rho)
    assert run.objective[-1] - optimum[1] < 1e-4


def test_distributed_admm_matches_centralized(problem, optimum):
    x_star, f_star = optimum
    run = admm_distributed(problem, n_iter=3000)
    assert abs(run.objective[-1] - f_star) < 1e-7
    assert np.allclose(run.x, x_star, atol=1e-3)
    assert run.consensus_error[-1] < 1e-6
    assert run.node_x.shape == (5, 5)


@pytest.mark.parametrize("builder", [graph.path, graph.ring, graph.star, graph.complete])
def test_distributed_admm_on_other_topologies(problem, optimum, builder):
    run = admm_distributed(problem, builder(4), n_iter=1500)
    assert run.objective[-1] - optimum[1] < 1e-5


def test_single_class_nodes_still_reach_the_global_model(data, problem, optimum):
    # sort by label so that most nodes hold one class only and cannot learn alone
    blocks = np.array_split(np.argsort(data.y_train, kind="stable"), 5)
    assert sum(len(np.unique(data.y_train[b])) == 1 for b in blocks) >= 3
    run = admm_distributed(problem, n_iter=1500, blocks=blocks)
    assert run.objective[-1] - optimum[1] < 1e-5
    for x_node in run.node_x:
        assert accuracy(x_node, data.A_test, data.y_test) > 0.97


def test_history_is_recorded_on_request(problem):
    run = admm_distributed(problem, n_iter=20, keep_history=True)
    assert run.node_history.shape == (21, 5, 5)
    assert np.array_equal(run.node_history[-1], run.node_x)
    assert admm_distributed(problem, n_iter=5).node_history is None


def test_bad_arguments(problem):
    with pytest.raises(ValueError):
        subgradient_descent(problem, schedule="magic")
    with pytest.raises(ValueError):
        admm_distributed(problem, blocks=[np.arange(10)])
