from scipy.ndimage import label
import numpy as np

# `compute_histograms` returns the raw counts, probability density function (PDF) 
# and cumulative distribution function (CDF) for each dye in a single EPG.
def compute_histograms(data: np.ndarray, bins: int = 256, value_range=None):
    """Compute histogram, pdf and cdf for each dye in an EPG.

    Parameters
    ----------
    data: np.ndarray
        Array of shape (num_dyes, length) containing intensities.
    bins: int
        Number of histogram bins.
    value_range: tuple or None
        Optional range passed to numpy.histogram.
    """
    num_dyes = data.shape[0]
    counts = np.zeros((num_dyes, bins), dtype=np.float64)
    for dye in range(num_dyes):
        counts[dye], bin_edges = np.histogram(data[dye], bins=bins, range=value_range)
    pdf = counts / counts.sum(axis=1, keepdims=True)
    cdf = np.cumsum(pdf, axis=1)
    return counts, pdf, cdf, bin_edges


def aggregate_histograms(dataset, bins=128):
    """
    Aggregates histogram counts, PDFs, and CDFs over all images in the dataset.

    Parameters
    ----------
    dataset : iterable
        An iterable of images, each with a .data attribute (shape: [num_dyes, length]).
    bins : int
        Number of histogram bins.

    Returns
    -------
    agg_counts : np.ndarray
        Aggregated histogram counts per dye.
    agg_pdf : np.ndarray
        Aggregated PDF per dye.
    agg_cdf : np.ndarray
        Aggregated CDF per dye.
    bin_edges : np.ndarray
        Bin edges used for the histograms.
    """
    agg_counts = None
    bin_edges = None
    for image in dataset:
        counts, pdf, cdf, bin_edges = compute_histograms(image.data, bins=bins)
        if agg_counts is None:
            agg_counts = counts
        else:
            agg_counts += counts

    agg_pdf = agg_counts / agg_counts.sum(axis=1, keepdims=True)
    agg_cdf = np.cumsum(agg_pdf, axis=1)
    return agg_counts, agg_pdf, agg_cdf, bin_edges









def find_allele_peaks(dataset):
    """Return all allele peak RFU values for all images in the dataset (all dyes at once)."""
    peaks = []
    for image in dataset:
        annot_img = image.annotation.image.flatten()
        rfu_data = image.data.flatten()
        labeled, num_features = label(annot_img)
        for region in range(1, num_features + 1):
            region_mask = labeled == region
            if np.any(region_mask):
                peak = rfu_data[region_mask].max()
                peaks.append(peak)
    return np.array(peaks)




def find_false_negative_peaks(dataset, predictions_mask, type) -> np.array:
    """Return all allele peak RFU values for all images in the dataset (all dyes at once)."""
    peaks = []
    for image, prediction in zip(dataset, predictions_mask):
        pred_img = prediction.flatten()
        annot_img = image.annotation.image.flatten()
        rfu_data = image.data.flatten()
        labeled, num_features = label(annot_img)
        for region in range(1, num_features + 1):
            region_mask = labeled == region
            if not pred_img[region_mask].any():
                peak = rfu_data[region_mask].max()
                peaks.append(peak)
    return np.array(peaks)







def find_predicted_peaks(images, predictions_mask) -> np.array:
    """Return all allele peak RFU values for all images in the dataset (all dyes at once)."""
    peaks = []
    for image, prediction in zip(images, predictions_mask):
        pred_img = prediction.flatten()
        rfu_data = image.data.flatten()
        new_peaks = rfu_data[pred_img]
        peaks.extend(new_peaks)
    return np.array(peaks)




def find_predicted_peaks_by_type(images, predictions_mask, type) -> np.array:
    """Return all false positive allele peak RFU values for all images in the dataset (all dyes at once)."""
    peaks = []
    for image, prediction in zip(images, predictions_mask):
        pred_img = prediction.flatten()
        annot_img = image.annotation.image.flatten() == 1
        rfu_data = image.data.flatten()
        if type == "true_positive":
            new_peaks = rfu_data[pred_img & annot_img]  # Get RFU values where prediction is True and annotation is True
        elif type == "false_positive":
            new_peaks = rfu_data[pred_img & ~annot_img]  # Get RFU values where prediction is True and annotation is False
        else:
            raise ValueError("Type must be 'true_positive' or 'false_positive'")
        peaks.extend(new_peaks)
    return np.array(peaks)