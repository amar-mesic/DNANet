import logging
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Tuple

import numpy as np
import referencing

from DNAnet.data.data_models import Annotation, Panel
from DNAnet.data.data_models.base import Image
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.kit_compatibility.lane_standards import BASE_PAIR_END, BASE_PAIR_START, RESCALE_SIZE, VAL_THRESHOLD, InternalSizeStandard, get_size_standard_bps
from DNAnet.data.utils import basepair_interpolator, extract_ss_peaks, find_peak_boundary, find_peak_idx_near_or_in_range, rescale_dye
from DNAnet.utils import load_donor_alleles_synthetic_data
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
                 reference_genotype_path: str,
                 epg_to_genotypes_mapping_path: str,
                 use_cache: bool = True,
                 meta: Optional[MutableMapping[str, Any]] = None,
                 size_standard: str = InternalSizeStandard.WEN_ILS.value,
                 include_size_standard: bool = False):
        self.path = path if isinstance(path, Path) else Path(path)
        self._panel = panel
        self.reference_genotype_path = reference_genotype_path
        self.epg_to_genotypes_mapping_path = epg_to_genotypes_mapping_path
        self.use_cache = use_cache
        self._meta = meta or dict()
        self._data: Optional[np.ndarray] = None
        self.size_standard = size_standard
        self.include_size_standard = include_size_standard
        self._scaler: Optional[np.ndarray] = None

    @property
    def data(self) -> np.ndarray:
        if self.use_cache:
            if self._data is None:
                self._data = self._read()
            return self._data
        return self._read()

    def _read(self) -> Optional[np.ndarray]:
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self.profile = profile = np.load(self.path)
        # Squeeze last dimension if present (e.g., (6, N, 1) -> (6, N))
        if profile.ndim == 3 and profile.shape[-1] == 1:
            profile = np.squeeze(profile, axis=-1)
        

        size_standard_dye_lane = np.array(profile[-1])
        size_standard_peaks_idxs = extract_ss_peaks(size_standard_dye_lane)
        self.bps = bps = get_size_standard_bps(self.size_standard)

        diff = VAL_THRESHOLD + 1
        shrinkages = 0
        while shrinkages < 10:
            self.size_standard_peaks_idxs = size_standard_peaks_idxs = size_standard_peaks_idxs[-len(bps):]

            coeffs = np.polyfit(size_standard_peaks_idxs, bps, 2)
            fitted = np.polyval(coeffs, size_standard_peaks_idxs)

            diff = np.max(np.abs(fitted - bps))
            if diff < VAL_THRESHOLD:
                break
            else:
                self.bps = bps = bps[:-1]  # remove the last base pair and try again
                shrinkages += 1
                
        # if shrinkages > 0:
        #     LOGGER.info(f"Size standard for {self.path.name} was shrunk {shrinkages} times to fit the profile. "
        #                    f"Max difference: {diff:.2f} bp.")
        if diff >= VAL_THRESHOLD:
                LOGGER.warning(f"Size standard for {self.path.name} differs {diff} from the expected ")
                return None

        # returns an interpolator function that maps the indices of the size standard peaks (i.e. scan points) to the base pairs
        interpolator = basepair_interpolator(indices=size_standard_peaks_idxs,
                                   original_x_values=bps, extrapolate=False)
        self.interpolated_base_pairs = interpolator(np.arange(len(size_standard_dye_lane)))

        rescaled_indices = rescale_dye(
            self.interpolated_base_pairs,
            rescale_size=RESCALE_SIZE,
            target_range=(BASE_PAIR_START, BASE_PAIR_END),
        )

        data = self._rescale_profile(
            profile,
            rescaled_indices,
            self.include_size_standard,
        )

        self._scaler = self.interpolated_base_pairs[rescaled_indices]

        

        try:
            true_alleles = load_donor_alleles_synthetic_data(str(self.path), self._panel, self.reference_genotype_path, self.epg_to_genotypes_mapping_path)
            segmentation = self._get_segmentation(true_alleles, data.shape)
            self._annotation = Annotation(image=segmentation)
            self._meta["called_alleles"] = true_alleles
        except ValueError as e:
            LOGGER.warning(f"Could not load true alleles for {self.path}: {e}")
        return data
    
    
    

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
