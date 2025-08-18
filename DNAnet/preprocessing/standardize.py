from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from .base import PreprocessingStep


@dataclass
class Standardizer(PreprocessingStep):
    """Standardize RFU values of an electropherogram.

    Parameters
    ----------
    per_dye:
        If ``True`` (default) the mean and variance are computed for each dye
        individually. If ``False`` the statistics are computed over the entire
        electropherogram after concatenating all dyes.
    plot_path:
        Optional filesystem path to which a plot of the mean and variance is
        written when :meth:`fit` is called.
    """

    def __init__(self, per_dye: bool = True, axis = 1):
        self.per_dye = per_dye
        self.axis = axis
        self.mean_ = None
        self.std_ = None


    def fit(self, data: np.ndarray, y: Iterable | None = None) -> "Standardizer":
        arr = np.asarray(data, dtype=float)
        if self.per_dye and arr.ndim > 1:
            self.mean_ = arr.mean(axis=self.axis, keepdims=True)
            self.std_ = arr.std(axis=self.axis, keepdims=True)
        else:
            self.mean_ = arr.mean()
            self.std_ = arr.std()
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardizeRFUStep must be fitted before calling transform().")

        return (data - self.mean_) / (self.std_ + 1e-8)

    def fit_transform(self, data: np.ndarray, y: Iterable | None = None) -> np.ndarray:
        return super().fit_transform(data, y)
    

    def _reset(self):
        """Reset the step to its initial state."""
        self.mean_ = None
        self.std_ = None
        return self