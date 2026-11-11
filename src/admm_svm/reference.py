"""Independent reference solution, used to measure how far each solver is from the optimum."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from .problem import SVMProblem


def reference_solution(problem: SVMProblem, tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Solve the problem with scikit-learn (libsvm) and return ``(x_star, f_star)``.

    libsvm minimises ``0.5 * ||w||^2 + C * sum(hinge)``, which is our objective
    divided by ``lam`` when ``C = 1 / (lam * N)``, so both have the same minimiser.
    """
    C = 1.0 / (problem.lam * problem.n_samples)
    svc = SVC(kernel="linear", C=C, tol=tol, cache_size=500).fit(problem.A, problem.y)
    x_star = np.append(svc.coef_.ravel(), svc.intercept_[0])
    return x_star, problem.objective(x_star)
