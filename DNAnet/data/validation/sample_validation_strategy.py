from typing import Protocol

from DNAnet.data.data_models.hid_image import HIDImage


class SampleValidationStrategy(Protocol):
    def __call__(self, image: HIDImage) -> bool : ...


class NFIValidationStrategy(SampleValidationStrategy):
    def __call__(self, image) -> bool:
        # check if the image is a valid sample
        return image.data is not None