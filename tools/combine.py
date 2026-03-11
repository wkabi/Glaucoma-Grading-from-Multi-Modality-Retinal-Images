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


    # print("=" * 20)
    # print("load results for image model")
    # image_result_path = "./outputs/trained_model/0a118x26-resnet50-weighted_ce-best-val.txt"
    # # image_result_path = "./outputs/trained_model/batch1/px2i7zwf-resnet34-ce-image-train-best-val.txt"
    # _, image_predicts = load_predicts(image_result_path)
    # image_pred_label = np.argmax(image_predicts, axis=1)
    # evaluator.update(image_pred_label, gt_label)
    # print("*Evaluate image result*")
    # metric, table_data = evaluator.mean_score(print=False)
    # table = AsciiTable(table_data)
    # print(table.table)
    # # evaluator.print_error()


    # print("=" * 20)
    # print("load results for oct model")
    # oct_result_path = "./outputs/trained_model/fsltuais-resnet3d50-ce-oct-train-best-val.txt"
    # _, oct_predicts = load_predicts(oct_result_path)
    # oct_pred_label = np.argmax(oct_predicts, axis=1)
    # print("*Evaluate oct result*")
    # evaluator.reset()
    # evaluator.update(oct_pred_label, gt_label)
    # metric, table_data = evaluator.mean_score(print=False)
    # table = AsciiTable(table_data)
    # print(table.table)

    # # image_weight = np.array([1.0, 0.5, 0.2])
    # # oct_weight = np.array([0.0, 0.5, 0.8])
    # # avg_predicts = (
    # #     np.einsum("ij,j->ij", image_predicts, image_weight)
    # #     + np.einsum("ij,j->ij", oct_predicts, oct_weight)
    # # )

    # print("=" * 20)
    # print("load results for disc image model")
    # disc_result_path = "./outputs/trained_model/j6x1fimr-resnet50-weighted_ce-disc-train-best-val.txt"
    # # image_result_path = "./outputs/trained_model/batch1/px2i7zwf-resnet34-ce-image-train-best-val.txt"
    # _, disc_predicts = load_predicts(disc_result_path)
    # disc_pred_label = np.argmax(disc_predicts, axis=1)
    # evaluator.reset()
    # evaluator.update(disc_pred_label, gt_label)
    # print("*Evaluate image result*")
    # metric, table_data = evaluator.mean_score(print=False)
    # table = AsciiTable(table_data)
    # print(table.table)

    # print("=" * 20)
    # # weight = [0.5, 0.3, 0.2]
    # avg_predicts = (
    #     weights[0] * image_predicts
    #     + weights[1] * oct_predicts
    #     + weights[2] * disc_predicts
    # )
    # avg_pred_label = np.argmax(avg_predicts, axis=1)
    # print("*Evaluate final avg result*")
    # evaluator.reset()
    # evaluator.update(avg_pred_label, gt_label)
    # metric, table_data = evaluator.mean_score(print=False)
    # table = AsciiTable(table_data)
    # print(table.table)
    # evaluator.print_error()

    # avg_predicts = np.maximum(image_predicts, oct_predicts)
    # avg_pred_label = np.argmax(avg_predicts, axis=1)
    # print("*Evaluate final max result*")
    # evaluator.reset()
    # evaluator.update(avg_pred_label, gt_label)
    # metric, table_data = evaluator.mean_score(print=False)
    # table = AsciiTable(table_data)
    # print(table.table)


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
        ids, predicts = load_predicts("{}-all.txt".format(path))
        all_predicts.append(predicts ** tsharpen)

    all_predicts = np.stack(all_predicts, axis=-1)
    final_predicts = np.mean(all_predicts, axis=-1)
    final_pred_label = np.argmax(final_predicts, axis=1)
    save_predictions(
        osp.join(save_dir, "Classification_Results.csv"),
        ids, final_pred_label
    )


if __name__ == "__main__":
    result_paths = [
        # "./outputs/trained_model/2b0428i6-multimodal-ce-train-best"
        # "outputs/trained_model/0a118x26-resnet50-weighted_ce-best",
        # "outputs/trained_model/1uhyx1nt-resnet3d50-ce-oct-train-best",
        # "outputs/fundus_img_20210815/bz2jr080-resnet50-ce-train-f0-best",
        # "outputs/fundus_img_20210815/3i9uoo5i-resnet50-ce-train-f1-best",
        # "outputs/fundus_img_20210815/3rjlsbcs-resnet50-ce-train-f2-best",
        # "outputs/fundus_img_20210815/2jkkkufc-resnet50-ce-train-f4-best",
        # "outputs/fundus_oct_20210815/1bqdh12q-resnet3d50-ce-train-f0-best",
        # "outputs/fundus_oct_20210815/eblxbagv-resnet3d50-ce-train-f1-best",
        # "outputs/fundus_oct_20210815/15yqhg6q-resnet3d50-ce-train-f2-best",
        # "outputs/trained_model/osjei66e-resnet50-ce-disc-train-best"
        "./outputs/fundus_twostream_20210816_sgd/jagusifd-twostream_model-twostream_loss-train-best",
        "./outputs/fundus_twostream_20210816_sgd/1ka2gqz9-twostream_model-twostream_loss-train-f1-best",
        # "outputs/fundus_twostream_resnet34_20210825/88nspzr3-twostream_model-twostream_loss-train-f0-best",
        # "outputs/fundus_twostream_resnet34_20210825/dvtymu5s-twostream_model-twostream_loss-train-f1-best",
        # "outputs/fundus_twostream_resnet34_20210825/1y6ua85t-twostream_model-twostream_loss-train-f3-best",
        # "outputs/fundus_twostream_resnet34_20210825/crv94j8v-twostream_model-twostream_loss-train-f2-best",
        # "outputs/fundus_twostream_resnet34_20210825/35kezrs3-twostream_model-twostream_loss-train-f4-best",
        # "outputs/fundus_twostream_resnet34_20210825/26n0988d-twostream_model-twostream_loss-train-f0-best",
        # "outputs/fundus_twostream_resnet34_sam_20210826/3cl82a4z-twostream_model-twostream_loss-train-f0-best",
        # "outputs/fundus_twostream_resnet34_sam_20210826/mxq7abdr-twostream_model-twostream_loss-train-f0-best",
        # "outputs/fundus_twostream_resnet34_sam_20210826/zjdre2pl-twostream_model-twostream_loss-train-f2-best",
        # "outputs/fundus_twostream_resnet34_sam_20210826/3e6mh3ec-twostream_model-twostream_loss-train-f3-best",
        # "outputs/fundus_twostream_resnet34_sam_20210826/fq6yeq2s-twostream_model-twostream_loss-train-f4-best",
        # "outputs/fundus_twostream_resnet34_sam_20210826/27lsdyzf-twostream_model-twostream_loss-train-f1-best",
        # "outputs/fundus_twostream_resnet50_sam_20210826/22ip0a8h-twostream_model-twostream_loss-train-f2-best",
        # "outputs/fundus_twostream_resnet50_sam_20210826/vx9gf25b-twostream_model-twostream_loss-train-f3-best",
        # "outputs/fundus_twostream_resnet50_sam_20210826/3l5gmaq4-twostream_model-twostream_loss-train-f4-best",
        # "outputs/fundus_twostream_resnet50_sam_20210826/eldh99s4-twostream_model-twostream_loss-train-f1-best",
        # "outputs/fundus_twostream_resnet50_sam_20210826/3ob3pynb-twostream_model-twostream_loss-train-f0-best",
        # "outputs/fundus_twostream_new_20210829/1wzjtfv4-twostream_resnet34-twostream_loss-train-f0-best",
        # "outputs/fundus_twostream_new_20210829/3gyduw4o-twostream_resnet50-twostream_loss-train-f0-best",
        # "outputs/fundus_twostream_new_20210829/3e5pzl4k-twostream_resnet34-twostream_loss-train-f0-best",
        # "outputs/fundus_multimodal_resnet34-34_sam_20210830/3n9pkth4-multimodal-multimodal_ce_kl-train-f0-best",
        # "outputs/fundus_multimodal_resnet34-34_sam_20210830/34pqqgrg-multimodal-multimodal_ce_kl-train-f1-best",
        # "outputs/fundus_multimodal_resnet34-34_sam_20210830/2ejeft5b-multimodal-multimodal_ce_kl-train-f2-best",
        "outputs/fundus_multimodal_resnet34-34_sam_20210830/1ia7qnp9-multimodal-multimodal_ce_kl-train-f3-best",
        # "outputs/fundus_multimodal_resnet34-34_sam_20210830/3qvibp5v-multimodal-multimodal_ce_kl-train-f0-best",
        # "outputs/fundus_multimodal_resnet34-34_sam_20210830/2k844ap6-multimodal-multimodal_ce_kl-train-f4-best",
        "outputs/fundus_multimodal_resnet34-50_sam_20210830/3d5vhpld-multimodal-multimodal_ce_kl-train-f0-best",
        # "outputs/fundus_multimodal_resnet34-50_sam_20210830/1017r6m5-multimodal-multimodal_ce_kl-train-f1-best",
        # "outputs/fundus_multimodal_resnet34-50_sam_20210830/3c3ooy85-multimodal-multimodal_ce_kl-train-f4-best",
        # "outputs/fundus_multimodal_resnet34-50_sam_20210830/15ieli8j-multimodal-multimodal_ce_kl-train-f2-best",
        # "outputs/fundus_multimodal_resnet34-50_sam_20210830/2l203ow7-multimodal-multimodal_ce_kl-train-f3-best",
        # "outputs/fundus_multimodal_resnet50-50_sam_20210830/10gmwq08-multimodal-multimodal_ce_kl-train-f1-best",
        # "outputs/fundus_multimodal_resnet50-50_sam_20210830/3pugr1xq-multimodal-multimodal_ce_kl-train-f0-best"
        "outputs/fundus_multimodal_resnet50-50_sam_20210830/1kf8yjr4-multimodal-multimodal_ce_kl-train-f2-best"
        # "outputs/fundus_multimodal_resnet50-50_sam_20210830/13jmclb7-multimodal-multimodal_ce_kl-train-f3-best"
    ]
    weights = [1.0] * len(result_paths)
    save_dir = "./outputs/trained_model/submit145"

    # check(result_paths, weights)
    test(result_paths, save_dir, tsharpen=1.0)
    print("done")
