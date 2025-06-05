

from typing import Optional
from DNAnet.data.data_models.base import InMemoryDataset
from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.kit_compatibility.lane_standards import InternalSizeStandard
from DNAnet.data.parsing.file_categorization_strategy import FileCategorizationStrategy, categorize_files
from DNAnet.data.validation.sample_validation_strategy import SampleValidationStrategy


class CustomHIDDataset(InMemoryDataset):
    def __init__(self, 
                 files, 
                 panel: Panel, 
                 shuffle: Optional[bool] = False,
                 limit: Optional[int] = None,
                 size_standard: str = InternalSizeStandard.WEN_ILS.value,
                 file_categorization_strategy: FileCategorizationStrategy = lambda file_name: "sample",
                 sample_validation_strategy: SampleValidationStrategy = lambda image: True
                ):
        super().__init__(shuffle)

        self.files = files
        self.panel = panel
        self.limit = limit
        self.size_standard = size_standard

        self.file_categorization_strategy = file_categorization_strategy

        categorized_files = categorize_files(self.files, self.file_categorization_strategy)
        sample_files = categorized_files["sample"]

        if limit is not None:
            sample_files = sample_files[:limit]

        unvalidated_images = [
            HIDImage(path=f, panel=self.panel, size_standard=self.size_standard)
            for f in sample_files
        ]

        validated_images = [
            image for image in unvalidated_images
            if sample_validation_strategy(image)
        ]
        
        self._data = validated_images
