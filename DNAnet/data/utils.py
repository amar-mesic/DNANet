from typing import Optional, Tuple
import logging


from sklearn import base
from DNAnet.data.kit_compatibility.lane_standards import get_size_standard_bps, ALL_SIZE_STANDARDS

import numpy as np
import scipy

LOGGER = logging.getLogger('dnanet')


# --- BEGIN MOD: Amar (2025-04-19) ---
# This is the begin and end number of base pairs for the ILS.
# Instead of hardcoding, we can use the values from lane_standard.py.

# BASE_PAIR_START, BASE_PAIR_END = 65, 475
# RESCALE_SIZE = 4096
# --- END MOD: Amar (2025-04-19) ---


def assert_image_data_valid_format(data: np.ndarray,
                                   n_color_channels: int = 3):
    """
    Makes sure that the raw image `data` conforms to the correct  format.
    """
    if len(data.shape) != 3:
        raise ValueError(
            f"The image must be 3D `(height, width, num_channels)`, not "
            f"{len(data.shape)}D. Full shape: {data.shape}"
        )
    if data.shape[-1] != n_color_channels:
        raise ValueError(
            f"Image must have {n_color_channels} color channels, not "
            f"{data.shape[-1]}. Full shape: {data.shape}"
        )
    if data.dtype != np.uint8:
        raise ValueError(
            f"dtype of `data` must be `np.uint8` to ensure consistency, "
            f"not {data.dtype}"
        )


def process_image(
        data: np.ndarray,
        channels_first: bool = False
) -> np.ndarray:
    """
    Processes the image data so that it can be fed directly to a model.

    :param data: the raw image data
    :param channels_first: whether to permute the resulting array such that
         the image channels are represented by the first axis rather than the last. The order of
         the dimensions becomes (num_channels, height, width)
    :return: A numpy array with the processed image data.
    """
    # TODO: here we can store any other preprocessing like augmentation or normalization

    # Swap the order of the axes if desired.
    if channels_first:
        return np.transpose(data, (2, 0, 1))

    # Otherwise, return the array as is.
    return data


def validate_ss_peaks(
    peaks: np.ndarray,
    expected_bps: np.ndarray,
    min_pixels_per_bp: float = 7,
    max_pixels_per_bp: float = 13
) -> bool:
    """
    Validate whether the identified peaks in the size standard are likely to
    correspond with at least one of the provided base pair arrays.
    Since we look at distances between peaks, is that why peaks will need to be one element shorter than the base pair array?

    In this method we conduct up to 3 checks:
    1. Check if the relative distances between the peaks are within the expected range of base pairs.
    2. If the first check fails, we check if the relative distances of the last 20 peaks are within the expected range.
    3. If the first two checks fail, we check if the relative distances of the peaks after a certain threshold (e.g., 4000) are within the expected range.

    :param size_standard_peaks_idxs: Indices of detected peaks in the size standard channel.
    :param expected_bps: Array or list of arrays of expected base pair positions for dye standards.
    :param min_pixels_per_bp: Minimum allowed pixels per base pair.
    :param max_pixels_per_bp: Maximum allowed pixels per base pair.
    :return: True if any check is passed, False otherwise.
    """

    distances_between_peaks = np.abs(
        np.diff((peaks,
                    scipy.ndimage.shift(peaks, shift=1, mode='nearest')
                    ), axis=0))[0, 1:]
    
    distance_basepairs = np.abs(
        np.diff((expected_bps,
                    scipy.ndimage.shift(expected_bps, shift=1, mode='nearest')
                    ), axis=0))[0, 1:]
    
    relative_distances = distances_between_peaks / distance_basepairs

    passes_validation = bool(np.all((relative_distances <= max_pixels_per_bp) & (relative_distances >= min_pixels_per_bp)))
    if passes_validation:
        return True
    # LOGGER.warning("Size standard peaks validation failed. Trying validation only scan points after 4000")
    
    # Temporary fix: only look at last 20 peaks
    # since first few peaks are often not ILS peaks, but primer flares.
    # TODO: Detect if peaks are primer flares and remove them. This would be in previous step.
    passes_validation = passes_validation or bool(np.all((relative_distances[-20:] <= max_pixels_per_bp) & (relative_distances[-20:] >= min_pixels_per_bp)))
    
    # Adjustable, but we can expect in most EPGs that the first 4000 scan points do not contain any alleles.
    threshold = 3700
    peaks_filtered = peaks[peaks >= threshold]
    distances_between_peaks_filtered = np.abs(
        np.diff((peaks_filtered,
                    scipy.ndimage.shift(peaks_filtered, shift=1, mode='nearest')
                    ), axis=0))[0, 1:]


    relative_distances_filtered = relative_distances[-distances_between_peaks_filtered.shape[0]:]

    return passes_validation or bool(np.all((relative_distances_filtered <= max_pixels_per_bp) & (relative_distances_filtered >= min_pixels_per_bp)))



def get_interpolated_basepairs(size_standard_dye_lane: np.ndarray, size_standard: str) -> (
        Optional)[np.ndarray]:
    """
    Takes the array of the size standard and detects the 19 peaks corresponding
    with the provided list of base pairs. Put these base pairs on their position
    in an array and interpolates all values in between.

    :param size_standard_dye: the values of the intenral size standard dye for this image/EPG
    :return: interpolated base pairs
    """
    # find the peaks in the size standard array
    # TODO: Detect if peaks are primer flares and remove them. This would be in previous step.
    size_standard_peaks_idxs = extract_ss_peaks(size_standard_dye_lane)

    bps = get_size_standard_bps(size_standard)
    # only take the last n peaks excluding the final peak, validate by comparing their
    # relative distances to SIZE_STANDARD_BPS
    relevant_peak_indices_from_lane_standard = size_standard_peaks_idxs[-len(bps):] # why remove last?

    # TODO: do not return none, but flag image as invalid
    if not validate_ss_peaks(relevant_peak_indices_from_lane_standard, expected_bps=bps):
        return None

    # returns an interpolator function that maps the indices of the size standard peaks (i.e. scan points) to the base pairs
    interpolator = basepair_interpolator(indices=relevant_peak_indices_from_lane_standard,
                                   original_x_values=bps)
    
    # interpolate these to the complete array (all values for indices outside
    # the size_standard_peaks_idxs range will be 0)
    # apply the interpolator function
    basepairs_interpolated = interpolator(np.arange(len(size_standard_dye_lane)))

    return basepairs_interpolated


def find_peaks_above_threshold(array: np.ndarray, threshold: int) -> \
        np.ndarray:
    """
    Find indices of peaks of an array above some threshold. This also includes
    looking for the beginning or end of flat peaks.
    """
    return (np.where((((array >= scipy.ndimage.shift(array, 1)) &
                       (array > scipy.ndimage.shift(array, -1))) |
                      ((array > scipy.ndimage.shift(array, 1)) &
                       (array >= scipy.ndimage.shift(array, -1))))
                     & (array >= threshold)))[0]


def find_peak_boundary(array: np.ndarray, idx: int, threshold: int) \
        -> Tuple[int, int]:
    """
    Find the start and end of a peak, whose peak top is located at `idx`. First
    split the array on `idx`. In the left split, look for the last index where
    the array has a value below `threshold`. In the right part, look similarly
    for the first index, in order to find the closest indices to `idx`.
    If the left split has only values above threshold, we return the beginning
    of the array. If the right split has only values above threshold, we return
    the end of the array. (TODO: improve this by fixed width/higher threshold?)
    """
    # TODO: sometimes the baseline of the array is higher than the threshold,
    # therefore we might want to adjust the threshold manually to some higher
    # value
    # TODO: check whether array[idx] is indeed a peak?
    array = array.flatten()
    # split the array on the peak index
    left_part, right_part = array[:idx], array[idx:]
    # on the left side of the peak, look for the closest index below the threshold
    start = np.where(left_part < threshold)[0][-1] + 1 if \
        any(left_part < threshold) else 0
    # on the right side of the peak, look for the closest index below the threshold
    end = np.where(right_part < threshold)[0][0] + idx - 1 if \
        any(right_part < threshold) else len(array) - 1
    return start, end


def find_peak_near_idx(array: np.ndarray, idx: int) -> np.ndarray:
    """
    Find the index of a peak in the `array` that is closest to the provided
    `idx` and has a peak height above the peak at position `idx`. Returns the
    index of the peak in the provided `array`. When two peaks have equal
    distance, the first peak index is returned.
    # TODO: it may occur that two peaks merge into each other, in that case
    # you ideally want the higher one, now we take the peak closest to `idx`.
    """
    peaks_idxs = find_peaks_above_threshold(array, array[idx])
    return peaks_idxs[np.abs(peaks_idxs - idx).argmin(), np.newaxis]


def find_peak_idx_near_or_in_range(array: np.ndarray, index_range: np.ndarray,
                                   threshold: int) -> np.ndarray:
    """
    Find a (single) peak index in `array` within the `index_range`, or just
    outside (before or after) the `index_range`. It may also be
    possible that no peak is found above the `threshold`. In that case, an
    empty array is returned.
    """
    values_in_range = array[index_range].flatten()
    if np.all(np.diff(values_in_range) > 0):
        # only increasing, search for peak near end of range
        peak_idx = find_peak_near_idx(array.flatten(), index_range[-1])
    elif np.all(np.diff(values_in_range) < 0):
        # only decreasing, search for peak near beginning of range
        peak_idx = find_peak_near_idx(array.flatten(), index_range[0])
    else:  # there must exist any (>=1) peak within the range
        peak_idx = find_peaks_above_threshold(values_in_range,
                                              threshold) + index_range[0]
        if peak_idx.size > 1:
            # multiple peaks found, return the highest
            peak_heights = array[peak_idx].flatten()
            peak_idx = peak_idx[np.argmax(peak_heights), np.newaxis]
    # return only peak index if the peak is above threshold
    return peak_idx if peak_idx.size > 0 and array[peak_idx] >= threshold else np.array([])


def extract_ss_peaks(array: np.ndarray) -> np.ndarray:
    """
    Takes an array and extracts the indices of the size standard peaks, by comparing each
    value with the neighbours and a threshold. We may find 'flat' peaks (e.g.
    [500, 520, 520, 510]) or a peak within a close distance of another
    peak, therefore we filter the found indices based on distance.
    """
    peak_idxs = find_peaks_above_threshold(array, 180)
    # the final two peaks in the size standard are often lower than the other peaks, therefore we
    # try to find those in the end of the array with a lower threshold if we haven't found them yet
    split_idx = 8200  # TODO: can we find this dynamically or something?
    if len(peak_idxs) > 0 and peak_idxs[-1] <= split_idx:
        final_peak_idxs = find_peaks_above_threshold(array[split_idx:], 120) + split_idx
        peak_idxs = np.union1d(peak_idxs, final_peak_idxs)
    # look for peak that are close (within 15 pixels) and delete the peak on the first index. This
    # may go wrong when we have a situation like [1000, 1001, 800, 800, 799], then we
    # ideally want to keep the highest peak (1001), but now this one gets deleted and we keep 1000.
    close_idxs = np.where(np.diff(peak_idxs) <= 15)[0]
    return np.delete(peak_idxs, close_idxs)


def basepair_interpolator(indices: np.ndarray,
                          original_x_values: np.ndarray,
                          extrapolate: bool = False) \
        -> scipy.interpolate.interp1d:
    """
    Generates a function whose call method uses interpolation to find the
    value of new points.

    :param indices: indices of the size standard peaks in the EPG
    :param original_x_values: the base pair values of the chosen size standard
    :param extrapolate: whether to use extrapolation or not
    """
    original_x_values = np.asarray(original_x_values)
    interp = scipy.interpolate.interp1d(indices,
                                        original_x_values,
                                        bounds_error=False,
                                        fill_value='extrapolate' if extrapolate else 0)

    return interp


def rescale_dye(basepairs: np.ndarray, size_standard: str, rescale_size: int = 4096) -> np.ndarray:
    """
    Rescale the interpolated base pairs of the size standard so that they fit between
    BASE_PAIR_START and BASE_PAIR_END, on exactly RESCALE_SIZE pixels. The output array
    should function as a translator that indicates which pixel indices (of an unscaled dye) should
    be on every pixel location.
    E.g. if the output is np.array([3825, 3826, ..]), then a pixel on index 3825 should be
    scaled to the first pixel, and a pixel on index 3826 should be scaled to the second pixel.
    """
    bps = get_size_standard_bps(size_standard)
    bp_start, bp_end = bps[0], bps[-1]
    target_linspace = np.linspace(bp_start, bp_end, rescale_size)

    # Presorting interpolated base pairs
    sort_indices = np.argsort(basepairs)
    sorted_basepairs = basepairs[sort_indices]

    # Find insertion indices
    insertion_indices = np.searchsorted(
        sorted_basepairs,
        target_linspace,
        side='left'
    )

    # Adjust indices for boundary conditions
    insertion_indices = np.clip(
        insertion_indices,
        1,
        len(sorted_basepairs) - 1
    )

    # Determine the closest index prior or after based on value proximity
    left_indices = insertion_indices - 1
    right_indices = insertion_indices

    left_deltas = np.abs(sorted_basepairs[left_indices] - target_linspace)
    right_deltas = np.abs(sorted_basepairs[right_indices] - target_linspace)

    return np.where(
        (left_deltas < right_deltas) | (left_deltas == right_deltas),
        sort_indices[left_indices],
        sort_indices[right_indices],
    )
