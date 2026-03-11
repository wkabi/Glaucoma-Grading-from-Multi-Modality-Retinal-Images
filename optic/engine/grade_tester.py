from terminaltables.ascii_table import AsciiTable
import os.path as osp
import torch
import torch.nn.functional as F
import time
import logging
import json
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from optic.utils import round_dict
from optic.utils.file_io import save_list
from optic.evaluation import GradeEvaluator, AverageMeter
import optic.data.test_augment as ta
from optic.engine.base_tester import _BaseTester

logger = logging.getLogger(__name__)


class GradeTester(_BaseTester):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.build_data_loader()
        self.build_model(checkpoint=self.cfg.test.checkpoint)
        self.init_meter()
        # self.init_wandb_or_not()

    def build_data_loader(self):
        self.test_loader = instantiate(self.cfg.data.object.test.dataset)
        logger.info("Data pipeline initialized")

    def init_meter(self):
        self.classes = self.test_loader.grade_classes
        self.num_classes = len(self.classes)
        self.evaluator = GradeEvaluator(
            num_classes=self.num_classes,
            classes=self.classes,
        )
        self.batch_time_meter = AverageMeter()
        logger.info("Meters initialized")

    def reset_meter(self):
        self.evaluator.reset()
        self.batch_time_meter.reset()

    def log_iter_info(self, iter, max_iter):
        log_dict = {}
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict.update(self.evaluator.curr_score())
        logger.info(
            "Test iter[{}/{}]\t{}".format(
                iter + 1, max_iter, json.dumps(round_dict(log_dict))
            )
        )

    def log_epoch_info(self):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        metric, table_data = self.evaluator.mean_score(print=False)
        log_dict.update(metric)
        logger.info(
            "Test Epoch\t{}".format(json.dumps(round_dict(log_dict)))
        )
        table = AsciiTable(table_data)
        logger.info("\n" + table.table)

    def preprocess_oct(self, volume):
        volume = self.test_loader.preprocess_oct(volume)
        if self.test_loader.oct_depth_resize == "multisample":
            augs = []
            for i in range(volume.shape[0]):
                aug = ta.augment(
                    volume[i],
                    self.cfg.test.augment.oct
                )
                augs.append(aug)
            if isinstance(augs[0], list):
                volume = []
                for i in range(len(augs[0])):
                    samples = [augs[j][i] for j in range(len(augs))]
                    samples = np.concatenate(samples, axis=0)
                    volume.append(samples)
            else:
                volume = np.concatenate(augs, axis=0)
        else:
            volume = ta.augment(
                volume,
                self.cfg.test.augment.oct
            )
        if isinstance(volume, list):
            volume = [
                torch.from_numpy(np.expand_dims(v, axis=1)).to(self.device)
                for v in volume
            ]
        else:
            volume = torch.from_numpy(np.expand_dims(volume, axis=1)).to(self.device)
        return volume

    def preprocess_image(self, image):
        image = self.test_loader.preprocess_image(image)
        image = ta.augment(
            image,
            self.cfg.test.augment.img
        )
        if isinstance(image, list):
            image = [torch.from_numpy(x).to(self.device) for x in image]
        else:
            image = torch.from_numpy(image).to(self.device)
        return image

    @torch.no_grad()
    def test(self):
        self.reset_meter()
        self.model.eval()

        max_iter = len(self.test_loader)
        end = time.time()
        if self.cfg.test.save_prediction:
            save_predicts = []
        for i, sample in enumerate(self.test_loader):
            if self.cfg.input_type == "oct":
                inputs = self.preprocess_oct(sample["oct_img"])
                # inputs = torch.from_numpy(inputs).to(self.device)
            elif self.cfg.input_type == "image":
                inputs = self.preprocess_image(sample["img"])
            elif self.cfg.input_type == "multimodal":
                volumes = self.preprocess_oct(sample["oct_img"])
                images = self.preprocess_image(sample["img"])
                inputs = (volumes, images)
            else:
                raise Exception("Invalid inputs setting : {}".format(self.cfg.input_type))
            #inputs = torch.from_numpy(inputs).to(self.device)
            label = sample["label"]
            # forward
            # import ipdb; ipdb.set_trace()
            if isinstance(inputs, list) and self.cfg.input_type != "multimodal":
                outputs = [self.model(x) for x in inputs]
                outputs = torch.cat(outputs, dim=0)
            else:
                outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=1)
            # avg_pred = predicts[0]
            predicts = ta.fuse_predicts(predicts, reduce=self.cfg.test.augment.reduce)
            pred_label = torch.argmax(predicts, dim=0)
            self.evaluator.update(
                np.expand_dims(pred_label.detach().cpu(), axis=0),
                np.expand_dims(label, axis=0)
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter)
            if self.cfg.test.save_prediction:
                predicts = predicts.tolist()
                save_predicts.append(
                    "{:04d},{:.5f},{:.5f},{:.5f}".format(
                        sample["id"], predicts[0], predicts[1], predicts[2]
                    )
                )
            # if self.cfg.test.save_feature:
            #     self.model.forward = types.MethodType(_forward_resnet_feature, self.model)
            #     feature = self.model(inputs)
            #     feature = ta.fuse_predicts(feature, reduce=self.cfg.test.augment.reduce)
            #     self.model.forward = types.MethodType(self.model._forward_impl, self.model)

            end = time.time()
        self.log_epoch_info()
        if self.cfg.test.save_prediction:
            save_list(save_predicts, osp.join(self.work_dir, self.cfg.test.save_prediction))

    def run(self):
        self.test()
