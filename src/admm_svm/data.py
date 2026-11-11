"""Banknote Authentication dataset (UCI): genuine vs forged banknotes.

1372 samples, 4 features computed from a wavelet transform of a photo of the
note (variance, skewness, kurtosis, entropy). In the raw file class 0 is a
genuine note and class 1 a forgery; here they become -1 and +1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .problem import SVMProblem

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "banknote.csv"
FEATURE_NAMES = ("variance", "skewness", "kurtosis", "entropy")


@dataclass(frozen=True)
class Dataset:
    A_train: np.ndarray
    y_train: np.ndarray
    A_test: np.ndarray
    y_test: np.ndarray

    def problem(self, lam: float = 0.01) -> SVMProblem:
        """The SVM training problem on the training split."""
        return SVMProblem(self.A_train, self.y_train, lam)


def load_banknote(csv_path: str | Path = DEFAULT_CSV, n_train: int = 500, seed: int = 0) -> Dataset:
    """Shuffle, keep ``n_train`` samples for training and the rest for testing.

    Features are standardised with the mean and standard deviation of the
    training split only, so the test split stays unseen.
    """
    raw = np.loadtxt(csv_path, delimiter=",")
    if not 0 < n_train < len(raw):
        raise ValueError(f"n_train must be between 1 and {len(raw) - 1}")
    raw = raw[np.random.default_rng(seed).permutation(len(raw))]
    A, y = raw[:, :4], np.where(raw[:, 4] > 0, 1.0, -1.0)

    mean, std = A[:n_train].mean(axis=0), A[:n_train].std(axis=0)
    A = (A - mean) / std
    return Dataset(A[:n_train], y[:n_train], A[n_train:], y[n_train:])
