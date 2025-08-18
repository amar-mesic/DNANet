# DNAnet/models/segmentation/threshold.py
import json
from pathlib import Path
import numpy as np

from DNAnet.allele_callers import NearestBasePairCaller
from DNAnet.data.data_models.base import Image
from DNAnet.models.base import Model
from DNAnet.models.prediction import Prediction
from DNAnet.typing import PathLike


class ThresholdSegmentationModel(Model):
    """
    Dummy baseline that labels every point with RFU >= `threshold`
    as allele signal.
    """
    def __init__(self, threshold: float = 75, apply_allele_caller: bool = True):
        self.threshold = threshold
        self.allele_caller = NearestBasePairCaller() if apply_allele_caller else None

    def predict(self, image: Image) -> Prediction:
        # Raw profile is (dyes, width, 1); squeeze to 2‑D before thresholding.
        mask = (image.data >= self.threshold).astype(np.float32)
        prediction = Prediction(image=mask)

        if self.allele_caller:
            prediction = self.allele_caller.call_alleles(image, prediction)
        return prediction

    # Nothing to train, but keep save/load for interface completeness.
    def save(self, model_dir: PathLike):
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(model_dir) / "config.json", "w") as fh:
            json.dump({"threshold": self.threshold}, fh)

    def load(self, model_dir: PathLike):
        with open(Path(model_dir) / "config.json") as fh:
            self.threshold = json.load(fh)["threshold"]
