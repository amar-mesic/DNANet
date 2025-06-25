from typing import List, Optional, Union
from DNAnet.data.data_models.base import InMemoryDataset
from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.kit_compatibility.lane_standards import InternalSizeStandard
from DNAnet.data.parsing.file_categorization_strategy import FileCategorizationStrategy, categorize_files
from DNAnet.data.parsing.file_parsing import find_files_by_suffix
from DNAnet.data.validation.sample_validation_strategy import SampleValidationStrategy
from DNAnet.typing import PathLike


class CustomHIDDataset(InMemoryDataset):
    def __init__(self, 
                 root_path: PathLike,
                 panel_path: PathLike, 
                 shuffle: Optional[bool] = False,
                 limit: Optional[int] = None,
                 size_standard: str = InternalSizeStandard.WEN_ILS.value,
                 file_categorization_strategy: FileCategorizationStrategy = lambda file_name: "sample",
                 sample_validation_strategy: SampleValidationStrategy = lambda image: True
                ):
        super().__init__(shuffle)

        self.root_path = root_path
        self.files = find_files_by_suffix(root_path, ".hid")

        # if isinstance(panel, PathLike):
        #     self.panel = Panel(panel)
        # else:
        #     self.panel = panel
        self.panel = Panel(panel_path)

        self.limit = limit
        self.size_standard = size_standard

        self.file_categorization_strategy = file_categorization_strategy
        self.sample_validation_strategy = sample_validation_strategy

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
            if self.sample_validation_strategy(image)
        ]
        
        self._data = validated_images


    def __str__(self):
        return (
            "CustomHIDDataset:\n"
            f"  files: \"<{len(self.files)} files>\"\n"
            f"  panel: \"{str(self.panel)}\"\n"
            f"  shuffle: \"{str(self.shuffle)}\"\n"
            f"  limit: \"{str(self.limit)}\"\n"
            f"  size_standard: \"{str(self.size_standard)}\"\n"
            f"  file_categorization_strategy: \"{getattr(self.file_categorization_strategy, '__name__', str(self.file_categorization_strategy))}\"\n"
            f"  sample_validation_strategy: \"{getattr(self.sample_validation_strategy, '__name__', str(self.sample_validation_strategy))}\""
        )
