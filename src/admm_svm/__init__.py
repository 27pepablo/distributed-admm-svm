"""Linear SVM solved three ways: subgradient descent, ADMM and distributed consensus ADMM."""

from .data import Dataset, load_banknote
from .problem import SVMProblem, accuracy, predict
from .reference import reference_solution
from .solvers import Result, admm_centralized, admm_distributed, subgradient_descent

__version__ = "1.0.0"

__all__ = [
    "Dataset",
    "Result",
    "SVMProblem",
    "accuracy",
    "admm_centralized",
    "admm_distributed",
    "load_banknote",
    "predict",
    "reference_solution",
    "subgradient_descent",
]
