from typing import Any, Callable, Optional
import numpy as np

from DNAnet.preprocessing.base import PreprocessingStep

class DynamicRangeCompress(PreprocessingStep):
    """Dynamic range compression preprocessing step supporting multiple monotonic functions."""
    def __init__(self, method: str = 'tanh', factor: float = 1.0, custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Parameters
        ----------
        method : str
            Compression method: 'tanh', 'log', or 'custom'.
        factor : float
            Scaling factor for the compression function.
        custom_func : Callable, optional
            Custom monotonic function to use if method is 'custom'.
        """
        self.method = method
        self.factor = factor
        self.custom_func = custom_func

    def fit(self, data: np.ndarray, y: Any = None, **kwargs) -> 'DynamicRangeCompress':
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.method == 'tanh':
            return np.tanh(data / self.factor) * self.factor
        elif self.method == 'log':
            # Add 1 to avoid log(0); assumes data >= 0
            return np.log1p(np.abs(data) / self.factor) * np.sign(data) * self.factor
        elif self.method == 'custom' and self.custom_func is not None:
            return self.custom_func(data)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")

    def fit_transform(self, data: np.ndarray, y: Any = None, **kwargs) -> Any:
        self.fit(data, y, **kwargs)
        return self.transform(data)