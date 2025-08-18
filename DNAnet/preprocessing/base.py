from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class PreprocessingStep(ABC):
    """Base class for preprocessing steps.

    Subclasses should implement :meth:`transform` and optionally override
    :meth:`fit` when training is required.
    """

    def fit(self, data: np.ndarray, y: np.ndarray | None = None, **kwargs) -> "PreprocessingStep":
        """Learn from the provided data.

        The default implementation does nothing and simply returns ``self``.
        Subclasses that require fitting can override this method.
        """
        return self

    @abstractmethod
    def transform(self, data: np.ndarray) -> Any:
        """Transform ``data`` and return the result."""
        raise NotImplementedError

    def fit_transform(self, data: np.ndarray, y: Any | None = None, **kwargs) -> Any:
        """Fit to ``data`` and return the transformed result."""
        self.fit(data, y, **kwargs)
        return self.transform(data)