import wandb
import numpy as np
import logging
import sklearn.metrics
from terminaltables import AsciiTable
from typing import List, Optional

from optic.evaluation import DatasetEvaluator

logger = logging.getLogger(__name__)


class GradeEvaluator(DatasetEvaluator):
    def __init__(
        self,
        num_classes: int,
        classes: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if classes is None:
            self.classes = [str(i) for i in range(self.num_classes)]
        else:
            self.classes = classes
        assert (
            num_classes == len(self.classes)
        ), "Number of classes doesn't match"
        self.reset()

    def reset(self) -> None:
        self.preds = None
        self.labels = None

    def main_metric(self):
        return "kappa"

    def num_samples(self):
        return (
            self.labels.shape[0]
            if self.labels is not None
            else 0
        )

    def update(self, pred: np.ndarray, label: np.ndarray) -> float:
        """update

        Args:
            pred (np.ndarray): n x 1
            label (np.ndarray): n x 1

        Returns:
            float: acc
        """
        assert pred.shape == label.shape
        if self.preds is None:
            self.preds = pred
            self.labels = label
        else:
            self.preds = np.concatenate((self.preds, pred), axis=0)
            self.labels = np.concatenate((self.labels, label), axis=0)

        acc = (pred == label).astype("int").sum() / label.shape[0]


        self.curr = {"acc": acc}

        return acc

    def curr_score(self):
        return self.curr

    def mean_score(self, print=True, all_metric=True):
        acc = (
            (self.preds == self.labels).astype("int").sum() 
            / self.labels.shape[0]
        )
        confusion = sklearn.metrics.confusion_matrix(self.labels, self.preds)
        class_acc = []
        for i in range(self.num_classes):
            class_acc.append(
                confusion[i, i] / np.sum(confusion[i])
            )
        macc = np.array(class_acc).mean()
        kappa = sklearn.metrics.cohen_kappa_score(self.labels, self.preds)

        metric = {"acc": acc, "macc": macc, "kappa": kappa}

        columns = ["id", "Class", "acc"]
        table_data = [columns]
        for i in range(self.num_classes):
            table_data.append(
                [i, self.classes[i], "{:.4f}".format(class_acc[i])]
            )
        table_data.append(
            [None, "macc", "{:.4f}".format(macc)]
        )
        table_data.append(
            [None, "kappa", "{:.4f}".format(kappa)]
        )

        if print:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)

        if all_metric:
            return metric, table_data
        else:
            return metric[self.main_metric()], table_data

    def wandb_score_table(self):
        _, table_data = self.mean_score(print=False)
        return wandb.Table(
            columns=table_data[0],
            data=table_data[1:]
        )

    def print_error(self):
        columns = ["id", "GT", "Pred"]
        table_data = [columns] 
        for i in range(self.preds.shape[0]):
            if self.preds[i] == self.labels[i]:
                continue
            table_data.append([i, self.labels[i], self.preds[i]])
        table = AsciiTable(table_data)
        print(table.table)
