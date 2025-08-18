import numpy as np
from numpy.testing import assert_array_equal

from DNAnet.data.preprocessing import PreprocessingStep, PreprocessingPipeline
from DNAnet.data.utils import process_image


class MultiplyStep(PreprocessingStep):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, data):
        return (data * self.factor).astype(data.dtype, copy=False)


class AddStep(PreprocessingStep):
    def __init__(self, value):
        self.value = value

    def transform(self, data):
        return (data + self.value).astype(data.dtype, copy=False)


def test_preprocessing_pipeline_sequential():
    pipeline = PreprocessingPipeline([MultiplyStep(2), AddStep(1)])
    data = np.array([1, 2, 3])
    assert_array_equal(pipeline.transform(data), np.array([3, 5, 7]))


def test_process_image_with_pipeline():
    data = np.zeros((2, 2, 3), dtype=np.uint8)
    pipeline = PreprocessingPipeline([AddStep(1)])
    result = process_image(data, preprocessing_pipeline=pipeline)
    assert_array_equal(result, np.ones_like(data))