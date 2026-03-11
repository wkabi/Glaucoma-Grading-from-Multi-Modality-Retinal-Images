import numpy as np
import logging
import sklearn.metrics
from terminaltables import AsciiTable
from typing import List, Optional

from optic.evaluation import DatasetEvaluator

logger = logging.getLogger(__name__)


class FoveaEvaluator(DatasetEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.preds = None
        self.labels = None

    def main_metric(self):
        return "r_aed"

    def num_samples(self):
        return (
            self.labels.shape[0]
            if self.labels is not None
            else 0
        )

    def euclid_dist(self, pred_coord, gt_coord):
        ed = np.sqrt(np.sum((pred_coord - gt_coord) ** 2, axis=1))

        return ed

    def update(self, pred: np.ndarray, label: np.ndarray) -> float:
        """update

        Args:
            pred (np.ndarray): n x 3 (x, y, has_fovea)
            label (np.ndarray): n x 3 (x, y, has_fovea)

        Returns:
            float: r_aed
        """
        assert pred.shape == label.shape
        if self.preds is None:
            self.preds = pred
            self.labels = label
        else:
            self.preds = np.concatenate((self.preds, pred), axis=0)
            self.labels = np.concatenate((self.labels, label), axis=0)

        ed = self.euclid_dist(pred[:, :2], label[:, :2])
        aed = np.mean(ed)
        r_aed = 1.0 / (aed + 0.1)

        self.curr = {"r_aed": r_aed}

        return r_aed

    def curr_score(self):
        return self.curr

    def mean_score(self, all_metric=True):
        ed = self.euclid_dist(self.preds[:, :2], self.labels[:, :2])
        aed = float(np.mean(ed))
        r_aed = 1.0 / (aed + 0.1)
        if self.preds.shape[1] == 3:
            acc = (
                (self.preds[:, 2].astype("int") == self.labels[:, 2].astype("int")).astype("int").sum()
                / self.labels.shape[0]
            )
        else:
            acc = 1.0

        metric = {"acc": acc, "aed": aed, "r_aed": r_aed}

        if all_metric:
            return metric
        else:
            return metric[self.main_metric()]
