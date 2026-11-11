"""The optimisation problem: a linear soft-margin SVM in primal form.

    minimise   f(x) = (1/N) * sum_i max(0, 1 - y_i * (a_i^T w + c))  +  (lam/2) * ||w||^2

with x = [w; c]. The intercept ``c`` is not regularised.

Every solver in this package works on the *signed design matrix* ``B``, whose i-th
row is ``y_i * [a_i, 1]``, so that the margin of sample i is simply ``(B @ x)[i]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class SVMProblem:
    """Training data and regularisation weight of one SVM problem."""

    A: np.ndarray  # (N, d) features
    y: np.ndarray  # (N,) labels in {-1, +1}
    lam: float  # regularisation weight (> 0)

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.y.shape != (self.A.shape[0],):
            raise ValueError("A must be (N, d) and y must be (N,)")
        if not np.all(np.isin(self.y, (-1.0, 1.0))):
            raise ValueError("labels must be -1 or +1")
        if self.lam <= 0:
            raise ValueError("lam must be positive")

    @property
    def n_samples(self) -> int:
        return self.A.shape[0]

    @property
    def dim(self) -> int:
        """Number of optimisation variables (features + intercept)."""
        return self.A.shape[1] + 1

    @cached_property
    def B(self) -> np.ndarray:
        """Signed design matrix: row i is ``y_i * [a_i, 1]``."""
        ones = np.ones((self.n_samples, 1))
        return self.y[:, None] * np.hstack([self.A, ones])

    @property
    def reg_diag(self) -> np.ndarray:
        """Diagonal of the regulariser: ones for the weights, zero for the intercept."""
        r = np.ones(self.dim)
        r[-1] = 0.0
        return r

    def objective(self, x: np.ndarray) -> float:
        """Value of f at ``x = [w; c]``."""
        hinge = np.maximum(0.0, 1.0 - self.B @ x).mean()
        return float(hinge + 0.5 * self.lam * np.sum(self.reg_diag * x * x))

    def subgradient(self, x: np.ndarray) -> np.ndarray:
        """One subgradient of f at ``x``."""
        B = self.B
        active = (B @ x) < 1.0
        return -B[active].sum(axis=0) / self.n_samples + self.lam * self.reg_diag * x


def predict(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Predicted labels (-1 / +1) of the classifier ``x = [w; c]`` on features ``A``."""
    return np.where(A @ x[:-1] + x[-1] >= 0.0, 1.0, -1.0)


def accuracy(x: np.ndarray, A: np.ndarray, y: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    return float(np.mean(predict(x, A) == y))


def prox_hinge(q: np.ndarray, t: float) -> np.ndarray:
    """Proximal operator of ``s -> t * max(0, 1 - s)``, applied element-wise.

    Solves ``argmin_s  t * max(0, 1 - s) + 0.5 * (s - q)^2``. Margins already
    above 1 are left alone, margins far below are pushed up by ``t``, and the
    ones in between are clamped to exactly 1.
    """
    return np.where(q >= 1.0, q, np.minimum(q + t, 1.0))
