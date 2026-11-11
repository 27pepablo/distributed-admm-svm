import numpy as np
import pytest

from admm_svm import SVMProblem, accuracy, predict
from admm_svm.problem import prox_hinge


@pytest.fixture
def tiny():
    A = np.array([[2.0, 0.0], [1.0, 1.0], [-1.0, 0.5], [-2.0, -1.0]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    return SVMProblem(A, y, lam=0.1)


def test_objective_by_hand(tiny):
    # at x = 0 every margin is 0, so every hinge term is 1 and the regulariser is 0
    assert tiny.objective(np.zeros(3)) == pytest.approx(1.0)
    # w = (1, 0), c = 0: margins are 2, 1, 1, 2 -> no hinge loss, only 0.5 * lam * ||w||^2
    assert tiny.objective(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.05)


def test_intercept_is_not_regularised(tiny):
    x = np.array([1.0, 0.0, 7.0])
    hinge_only = np.maximum(0, 1 - tiny.B @ x).mean()
    assert tiny.objective(x) == pytest.approx(hinge_only + 0.05)


def test_subgradient_inequality(tiny):
    # g is a subgradient at x  <=>  f(z) >= f(x) + g.(z - x) for every z
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, z = rng.normal(size=3), rng.normal(size=3)
        g = tiny.subgradient(x)
        assert tiny.objective(z) >= tiny.objective(x) + g @ (z - x) - 1e-12


def test_prox_hinge_matches_brute_force():
    grid = np.linspace(-5, 5, 200001)
    for q in (-2.0, 0.3, 0.9, 1.0, 1.7):
        for t in (0.05, 0.5, 2.0):
            brute = grid[np.argmin(t * np.maximum(0, 1 - grid) + 0.5 * (grid - q) ** 2)]
            assert prox_hinge(np.array([q]), t)[0] == pytest.approx(brute, abs=1e-4)


def test_predict_and_accuracy(tiny):
    x = np.array([1.0, 0.0, 0.0])
    assert np.array_equal(predict(x, tiny.A), tiny.y)
    assert accuracy(x, tiny.A, tiny.y) == 1.0
    assert accuracy(-x, tiny.A, tiny.y) == 0.0


@pytest.mark.parametrize(
    "A, y, lam",
    [
        (np.zeros((3, 2)), np.array([1.0, -1.0]), 0.1),  # shapes disagree
        (np.zeros((2, 2)), np.array([1.0, 0.0]), 0.1),  # label 0 is not allowed
        (np.zeros((2, 2)), np.array([1.0, -1.0]), 0.0),  # lam must be positive
    ],
)
def test_invalid_problems_are_rejected(A, y, lam):
    with pytest.raises(ValueError):
        SVMProblem(A, y, lam)
