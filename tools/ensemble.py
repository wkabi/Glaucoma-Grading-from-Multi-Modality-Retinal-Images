import argparse
import os.path as osp
import numpy as np
from terminaltables import AsciiTable

from optic.utils.file_io import mkdir 
from optic.data.fundus_dataset import FundusDataset
from optic.evaluation import GradeEvaluator


def load_predicts(path):
    ids = []
    predicts = []
    with open(path, "r") as f:
        for line in f:
            fields = line.strip().split(",")
            ids.append(fields[0])
            predicts.append(
                [float(fields[1]), float(fields[2]), float(fields[3])]
            )
    predicts = np.array(predicts)
    return ids, predicts


def check(result_paths, weights):
    data_root = "./data/GAMMA_training_data/training_data"

    dataset = FundusDataset(
        data_root=data_root,
        mode="val",
        return_oct=False,
        return_mask=False
    )

    evaluator = GradeEvaluator(num_classes=3)

    print("load ground truth labels")
    gt_label = []
    for i, sample in enumerate(dataset):
        gt_label.append(sample["label"])
    gt_label = np.array(gt_label)
    # gt_label = np.expand_dims(gt_label, axis=1)

    all_predicts = []
    for i, path in enumerate(result_paths):
        print("=" * 20)
        print("load results [{}]".format(i))
        ids, predicts = load_predicts("{}-val.txt".format(path))
        pred_label = np.argmax(predicts, axis=1)
        evaluator.reset()
        evaluator.update(pred_label, gt_label)
        metric, table_data = evaluator.mean_score(print=False)
        table = AsciiTable(table_data)
        print(table.table)
        all_predicts.append(predicts * weights[i])

    print("=" * 20)
    all_predicts = np.stack(all_predicts, axis=-1)
    final_predicts = np.sum(all_predicts, axis=-1)
    final_pred_label = np.argmax(final_predicts, axis=1)
    print("*Evaluate final avg result*")
    evaluator.reset()
    evaluator.update(final_pred_label, gt_label)
    metric, table_data = evaluator.mean_score(print=False)
    table = AsciiTable(table_data)
    print(table.table)


def save_predictions(path, ids, pred_label):
    label2str = ["1,0,0", "0,1,0", "0,0,1"]
    with open(path, "w") as f:
        f.write("data,non,early,mid_advanced\n")
        for i in range(len(ids)):
            f.write("{},{}\n".format(ids[i], label2str[pred_label[i]]))


def test(result_paths, save_dir, tsharpen=1.0):
    mkdir(save_dir)
    all_predicts = []
    for i, path in enumerate(result_paths):
        ids, predicts = load_predicts(path)
        all_predicts.append(predicts ** tsharpen)

    all_predicts = np.stack(all_predicts, axis=-1)
    final_predicts = np.mean(all_predicts, axis=-1)
    final_pred_label = np.argmax(final_predicts, axis=1)
    save_predictions(
        osp.join(save_dir, "Classification_Results.csv"),
        ids, final_pred_label
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble results for classification",
    )
    parser.add_argument("results", type=str, nargs="+")
    parser.add_argument("--tsharpen", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="./submit")

    args = parser.parse_args()

    print("Ensemble the results for : ")
    for r in args.results:
        print(r)

    test(args.results, args.save_dir, tsharpen=args.tsharpen)

    print("done")
