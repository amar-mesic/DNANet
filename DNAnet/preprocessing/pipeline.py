from typing import Iterable, List, Any
from .base import PreprocessingStep


class PreprocessingPipeline(PreprocessingStep):
    """Execute a sequence of :class:`PreprocessingStep` objects."""

    def __init__(self, steps: Iterable[PreprocessingStep]):
        self.steps: List[PreprocessingStep] = list(steps)

    def fit(self, data: Any, y: Any | None = None) -> "PreprocessingPipeline":
        for step in self.steps:
            data = step.fit_transform(data, y)
        return self

    def transform(self, data: Any) -> Any:
        for step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, data: Any, y: Any | None = None) -> Any:
        for step in self.steps:
            data = step.fit_transform(data, y)
        return data
    

    def to_config(self):
        return [step.to_config() for step in self.steps]

    @classmethod
    def from_config(cls, cfg):
        steps = [globals()[c["name"]].from_config(c) for c in cfg]
        return cls(steps)
