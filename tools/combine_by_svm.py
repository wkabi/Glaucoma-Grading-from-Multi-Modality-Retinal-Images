import os.path as osp
import numpy as np
from terminaltables import AsciiTable
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from joblib import dump, load
from typing import List

from optic.evaluation import GradeEvaluator
from optic.utils.file_io import mkdir

def load_labels(path):
    ids = []
    labels = []
    with open(path, "r") as f:
        for line in f:
            fields = line.strip().split(",")
            ids.append(fields[0])
            labels.append(int(fields[1]))
    labels = np.array(labels)
    return ids, labels


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


def load_features(paths):
    features = None
    for path in paths:
        ids, predicts = load_predicts(path)
        if features is None:
            features = predicts
        else:
            features = np.concatenate((features, predicts), axis=1)
    print(features.shape)
    return ids, features


def train(
    train_feature_paths: List[str],
    train_label_path: str,
    val_feature_paths: List[str],
    val_label_path: str,
    save_model_path: str
):
    # train_label_path = "data/GAMMA_training_data/training_data/train.csv"
    # train_result_path = [
    #     "./outputs/trained_model/0a118x26-resnet50-weighted_ce-best-train.txt",
    #     "./outputs/trained_model/fsltuais-resnet3d50-ce-oct-train-best-train.txt",
    #     # "./outputs/trained_model/j6x1fimr-resnet50-weighted_ce-disc-train-best-train.txt"
    # ]

    # val_label_path = "data/GAMMA_training_data/training_data/valtest.csv"
    # val_result_path = [
    #     "./outputs/trained_model/0a118x26-resnet50-weighted_ce-best-val.txt",
    #     "./outputs/trained_model/fsltuais-resnet3d50-ce-oct-train-best-val.txt",
    #     # "./outputs/trained_model/j6x1fimr-resnet50-weighted_ce-disc-train-best-val.txt"
    # ]

    ids, train_features = load_features(train_feature_paths)
    _, train_labels = load_labels(train_label_path)
    ids, val_features = load_features(val_feature_paths)
    _, val_labels = load_labels(val_label_path)

    clf = make_pipeline(
        # StandardScaler(),
        Normalizer(norm="l2"),
        SVC(kernel="linear", gamma="auto", C=1)
    )
    clf.fit(train_features, train_labels)

    val_new_pred = clf.predict(val_features)

    print(val_new_pred.shape)

    evaluator = GradeEvaluator(3)
    evaluator.update(val_new_pred, val_labels)
    metric, table_data = evaluator.mean_score(print=False)
    table = AsciiTable(table_data)
    print(table.table)

    # dump(clf, "./outputs/trained_model/combine_svm.joblib")
    dump(clf, save_model_path)


def save_predictions(path, ids, pred_label):
    label2str = ["1,0,0", "0,1,0", "0,0,1"]
    with open(path, "w") as f:
        f.write("data,non,early,mid_advanced\n")
        for i in range(len(ids)):
            f.write("{},{}\n".format(ids[i], label2str[pred_label[i]]))


def test(test_features_paths: List[str], model_path: str, save_dir: str):
    # test_result_path = [
    #     "./outputs/trained_model/0a118x26-resnet50-weighted_ce-best-val_data.txt",
    #     "./outputs/trained_model/fsltuais-resnet3d50-ce-oct-train-best-val_data.txt",
    #     # "./outputs/trained_model/j6x1fimr-resnet50-weighted_ce-disc-train-best-val_data.txt"
    # ]
    # svm_model_path = "./outputs/trained_model/combine_svm.joblib"
    # save_dir = "./outputs/trained_model/submit17"
    mkdir(save_dir)

    print("load features")
    ids, test_features = load_features(test_features_paths)
    print("load model : ", model_path)
    clf = load(model_path)
    test_preds = clf.predict(test_features)

    save_path = osp.join(save_dir, "Classification_Results.csv")
    save_predictions(save_path, ids, test_preds)


if __name__ == "__main__":
    train_label_path = "data/GAMMA_training_data/training_data/train.csv"
    train_result_path = [
        "./outputs/trained_model/0a118x26-resnet50-weighted_ce-best-train.txt",
        "./outputs/trained_model/fsltuais-resnet3d50-ce-oct-train-best-train.txt",
        # "./outputs/trained_model/j6x1fimr-resnet50-weighted_ce-disc-train-best-train.txt"
    ]
    val_label_path = "data/GAMMA_training_data/training_data/val.csv"
    val_result_path = [
        "./outputs/trained_model/0a118x26-resnet50-weighted_ce-best-val.txt",
        "./outputs/trained_model/fsltuais-resnet3d50-ce-oct-train-best-val.txt",
        # "./outputs/trained_model/j6x1fimr-resnet50-weighted_ce-disc-train-best-val.txt"
    ]
    save_model_path = "./outputs/trained_model/combine_svm.joblib"

    test_result_path = [
        "./outputs/trained_model/0a118x26-resnet50-weighted_ce-best-val_data.txt",
        "./outputs/trained_model/fsltuais-resnet3d50-ce-oct-train-best-val_data.txt",
        # "./outputs/trained_model/j6x1fimr-resnet50-weighted_ce-disc-train-best-val_data.txt"
    ]
    save_dir = "./outputs/trained_model/submit17"
    print("training ... ")
    train(train_result_path, train_label_path, val_result_path, val_label_path, save_model_path)
    print("testing ...")
    test(test_result_path, save_model_path, save_dir)
    print("Done")
