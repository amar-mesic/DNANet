import logging
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Tuple

import numpy as np

from DNAnet.data.data_models import Annotation, Panel
from DNAnet.data.data_models.base import Image
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.kit_compatibility.lane_standards import InternalSizeStandard
from DNAnet.data.utils import find_peak_boundary, find_peak_idx_near_or_in_range, get_interpolated_basepairs, rescale_dye
from DNAnet.data.parsing import parse_called_alleles
from DNAnet.utils import load_donor_alleles, load_donor_alleles_provedit, load_donor_alleles_synthetic_data
from DNAnet.typing import PathLike

LOGGER = logging.getLogger("dnanet")

class SyntheticImage(Image):
    """
    Image representation for synthetic DNA profile data stored in .npy format.

    :param path: location of .npy file
    :param panel: the panel to be used
    :param annotation: any Annotation belonging to the image
    :param use_cache: whether retrieved data should be cached
    :param meta: meta information of the file.
    """
    def __init__(self,
                 path: PathLike,
                 panel: Panel,
                 use_cache: bool = True,
                 meta: Optional[MutableMapping[str, Any]] = None,
                 size_standard: str = InternalSizeStandard.WEN_ILS.value,
                 include_size_standard: bool = False,
                 annotations_file: Optional[str] = None):
        self.path = path if isinstance(path, Path) else Path(path)
        self._panel = panel
        self.use_cache = use_cache
        self._meta = meta or dict()
        self._data: Optional[np.ndarray] = None
        self.size_standard = size_standard
        self.include_size_standard = include_size_standard
        self.annotations_file = annotations_file
        self._scaler: Optional[np.ndarray] = None

    @property
    def data(self) -> np.ndarray:
        if self.use_cache:
            if self._data is None:
                self._data = self._read()
            return self._data
        return self._read()

    def _read(self) -> np.ndarray:
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        profile = np.load(self.path)
        # Squeeze last dimension if present (e.g., (6, N, 1) -> (6, N))
        if profile.ndim == 3 and profile.shape[-1] == 1:
            profile = np.squeeze(profile, axis=-1)

        # Pad the profile to shape (6, 9641) along axis=1 (scan points)
        target_scan_points = 9641
        current_scan_points = profile.shape[1]
        if current_scan_points == 5000:
            # The 5000 scan points are from indices 4000:9000 of the original
            pad_left = 4000
            pad_right = target_scan_points - (pad_left + current_scan_points)
            profile = np.pad(profile, ((0, 0), (pad_left, pad_right)), mode='constant')
        elif current_scan_points < target_scan_points:
            # Unexpected case: pad all at the end
            pad_right = target_scan_points - current_scan_points
            profile = np.pad(profile, ((0, 0), (0, pad_right)), mode='constant')
        elif current_scan_points > target_scan_points:
            raise ValueError(f"Profile scan points ({current_scan_points}) exceed expected ({target_scan_points})")

        interpolated_base_pairs = get_interpolated_basepairs(np.array(profile[-1]), self.size_standard)
        if interpolated_base_pairs is None:
            raise ValueError(f"Invalid size standard for file {self.path}")
        # Scale the profile using the size standard
        data = self._rescale_profile(profile,
                                     interpolated_base_pairs,
                                     self.size_standard,
                                     self.include_size_standard)
        
        # Create a scaler, which is used to map a pixel index in the profile to a base pair
        # location, i.e. the first pixel is in fact BASE_PAIR_START, the last pixel is BASE_PAIR_END
        self._scaler = interpolated_base_pairs[rescale_dye(interpolated_base_pairs, self.size_standard)]

        # called_alleles = None
        # # Determine the called alleles from the annotations file 
        # if self.annotations_file and self._panel and \
        #         (annotations_name := self.meta.get('annotations_name')):
        #     called_alleles = parse_called_alleles(self.annotations_file,
        #                                           self._panel,
        #                                           annotations_name)

        # if called_alleles and self.annotation is None:
        #     # Parse the called alleles into a segmentation
        #     segmentation = self._get_segmentation(called_alleles, data.shape)
        #     self._annotation = Annotation(image=segmentation) # where the annotation is ASSIGNED
        #     self._meta['called_alleles'] = called_alleles

        # But what if there is no annotations file, only genotype info?
        # This is ofc hardcoded for the ProvedIt dataset for now
        # if self.annotation is None and self._panel:
        try:
            true_alleles = load_donor_alleles_synthetic_data(self.path, self._panel)
            segmentation = self._get_segmentation(true_alleles, data.shape)
            self._annotation = Annotation(image=segmentation)
            self._meta["called_alleles"] = true_alleles
        except ValueError as e:
            LOGGER.warning(f"Could not load true alleles for {self.path}: {e}")
        return data
    


    @staticmethod
    def _rescale_profile(profile: np.ndarray,
                         interpolated_base_pairs: np.ndarray,
                         size_standard: str,
                         include_standard: bool) -> np.ndarray:
        selected_profile = profile if include_standard else profile[:-1]
        data = selected_profile[:, rescale_dye(interpolated_base_pairs, size_standard)]
        return data[..., np.newaxis]
    
    

    def _get_segmentation(self,
                          called_alleles,
                          shape: Tuple[int, ...]) -> np.ndarray:
        image = np.zeros(shape, dtype=np.int8)
        for marker in called_alleles:
            for allele in marker.alleles:
                image[
                    marker.dye_row,
                    slice(*tuple(np.argmin(np.abs(self.scaler - allele.bin), axis=1))),
                    0
                ] = 1
        return image
    

    def adjust_annotations(self, adjustment_type: str = 'top') -> 'SyntheticImage':
        """
        Adjust the annotation of the image or the spu annotation in case of 'adjust_spu' is True.
        If `adjustment_type` is 'top', (by default) we label the top of the peak, instead of the
        entire bin. If the type is 'complete', we find the entire peak and label this.
        Note that the original image annotations are overwritten.
        """
        profile = self.data  # force data to be read to generate annotations
        annotations = self.annotation.image
        if annotations is None:
            LOGGER.warning(f"No annotations found for file {self.path} when "
                           f"adjusting annotations.")
            return self

        for layer, dye in enumerate(profile):
            # find indices of groups of positive annotations
            _annotations, _ = np.where(annotations[layer] == 1)
            if _annotations.size == 0:  # no annotation present in this dye
                continue
            annotation_groups = np.split(_annotations, np.where(np.diff(_annotations) != 1)[0] + 1)
            for ann_group in annotation_groups:
                annotations[layer, ann_group, 0] = 0.
                peak_idx = find_peak_idx_near_or_in_range(dye, ann_group,
                                                          HIDImage.THRESHOLD)

                if peak_idx.size == 0:
                    pass
                else:
                    if adjustment_type == 'complete':
                        # find the boundary of the peak and annotate the range
                        start, end = find_peak_boundary(dye, int(peak_idx),
                                                        HIDImage.THRESHOLD)
                        annotations[layer, np.arange(start, end + 1), 0] = 1.
                    elif adjustment_type == 'top':
                        # label only the top of the peak
                        annotations[layer, peak_idx, 0] = 1.
                    else:
                        raise ValueError("Unknown adjustment type found: "
                                         f"{adjustment_type}. Please provide"
                                         " either `top` or `complete`.")
        return self

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]

    @property
    def annotation(self):
        return self._annotation

    @property
    def meta(self) -> MutableMapping[str, Any]:
        return self._meta

    @property
    def panel(self) -> Panel:
        return self._panel

    @property
    def scaler(self) -> Optional[np.ndarray]:
        if self._scaler is None:
            self._read()
        if self._scaler is None:
            return None
        return self._scaler[np.newaxis, :]

    def __repr__(self):
        return f"SyntheticImage({self.path.name})"
