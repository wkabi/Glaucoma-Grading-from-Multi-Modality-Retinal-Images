import logging
import torch
import torch.nn.functional as F
import time
from terminaltables.ascii_table import AsciiTable
import numpy as np
import json
import wandb
from omegaconf.dictconfig import DictConfig

from optic.utils.misc import get_lr, round_dict
from optic.modeling import TwostreamLoss, SAM
from optic.engine.grade_trainer import GradeTrainer
from optic.evaluation import AverageMeter, LossMeter, GradeEvaluator
import optic.data.test_augment as ta
from optic.utils.torch_helper import disable_bn, enable_bn

logger = logging.getLogger(__name__)


class TwostreamTainer(GradeTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def build_meter(self):
        self.classes = self.train_loader.dataset.grade_classes
        self.num_classes = len(self.classes)
        self.img_evaluator = GradeEvaluator(
            num_classes=self.num_classes,
            classes=self.classes
        )
        self.oct_evaluator = GradeEvaluator(
            num_classes=self.num_classes,
            classes=self.classes
        )
        self.evaluator = GradeEvaluator(
            num_classes=self.num_classes,
            classes=self.classes
        )
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_meter = LossMeter(
            num_terms=len(self.loss_func.names),
            names=self.loss_func.names
        )
        logger.info("Meters initialized")

    def reset_meter(self):
        self.img_evaluator.reset()
        self.oct_evaluator.reset()
        self.evaluator.reset()
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()

    def train_epoch(self, epoch):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        end = time.time()
        for i, samples in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            octs = samples["oct_img"].to(self.device)
            imgs = samples["img"].to(self.device)
            labels = samples["label"].to(self.device)
            # forward
            output_octs, output_imgs = self.model((octs, imgs))
            loss = self.loss_func((output_octs, output_imgs), labels)
            if isinstance(loss, tuple):
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            if isinstance(self.optimizer, SAM):
                self.optimizer.first_step(zero_grad=True)
                # second forward-backward step for SAM optimizer
                # disable_bn(self.model)
                loss2 = self.loss_func(self.model((octs, imgs)), labels)
                if isinstance(loss2, tuple):
                    loss2[0].backward()
                else:
                    loss2.backward()
                self.optimizer.second_step(zero_grad=True)
                # enable_bn(self.model)
            else:
                self.optimizer.step()
            # metric
            self.loss_meter.update(loss, octs.size(0))
            predict_octs = F.softmax(output_octs, dim=1)
            pred_label_octs = torch.argmax(predict_octs, dim=1)
            self.oct_evaluator.update(
                pred_label_octs.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            predict_imgs = F.softmax(output_imgs, dim=1)
            pred_label_imgs = torch.argmax(predict_imgs, dim=1)
            self.img_evaluator.update(
                pred_label_imgs.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            predicts = (predict_octs + predict_imgs) / 2
            pred_label = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                pred_label.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_epoch_info(epoch)

    def log_iter_info(self, iter, max_iter, epoch, phase="Train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_vals())
        oct_curr_score = self.oct_evaluator.curr_score()
        log_dict.update(dict(
            ("oct_{}".format(key), val) for (key, val) in oct_curr_score.items()
        ))
        img_curr_score = self.img_evaluator.curr_score()
        log_dict.update(dict(
            ("img_{}".format(key), val) for (key, val) in img_curr_score.items()
        ))
        combo_curr_score = self.evaluator.curr_score()
        log_dict.update(dict(
            ("combo_{}".format(key), val) for (key, val) in combo_curr_score.items()
        ))
        logger.info("{} Iter[{}/{}][{}]\t{}".format(
            phase, iter + 1, max_iter, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable and phase.lower() == "train":
            wandb_log_dict = {"iter": epoch * max_iter + iter}
            wandb_log_dict.update(dict(
                ("{}/Iter/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_epoch_info(self, epoch, phase="Train"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, TwostreamLoss):
            log_dict["alpha"] = self.loss_func.alpha
        metric, _ = self.oct_evaluator.mean_score(print=False)
        log_dict.update(dict(
            ("oct_{}".format(key), val) for (key, val) in metric.items()
        ))
        metric, _ = self.img_evaluator.mean_score(print=False)
        log_dict.update(dict(
            ("img_{}".format(key), val) for (key, val) in metric.items()
        ))
        metric, table_data = self.evaluator.mean_score(print=False)
        log_dict.update(dict(
            ("combo_{}".format(key), val) for (key, val) in metric.items()
        ))
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))

        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)
            if phase.lower() != "train":
                table = AsciiTable(table_data)
                logger.info("\n" + table.table)
                wandb_log_dict["{}/score_table".format(phase)] = wandb.Table(
                    columns=table_data[0], data=table_data[1:]
                )
            wandb.log(wandb_log_dict)

    def preprocess_oct(self, volume):
        volume = self.val_loader.preprocess_oct(volume)
        if self.val_loader.oct_depth_resize == "multisample":
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
        image = self.val_loader.preprocess_image(image)
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
    def eval_epoch(self, data_loader, epoch, phase="Val"):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)
        end = time.time()
        for i, samples in enumerate(data_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            octs = self.preprocess_oct(samples["oct_img"])
            imgs = self.preprocess_image(samples["img"])
            label = np.expand_dims(samples["label"], axis=0)
            # forward
            # output_octs, output_imgs = self.model((octs, imgs))
            if isinstance(octs, list):
                outputs_list = [self.model(x) for x in zip(octs, imgs)]
                oct_outputs = torch.cat([out[0] for out in outputs_list],
                                        dim=0)
                img_outputs = torch.cat([out[1] for out in outputs_list],
                                        dim=0)
                # outputs = [torch.cat(x, dim=0) for x in outputs]
            else:
                oct_outputs, img_outputs = self.model((octs, imgs))
            # For oct
            oct_predicts = F.softmax(oct_outputs, dim=1)
            oct_predicts = ta.fuse_predicts(
                oct_predicts,
                reduce=self.cfg.test.augment.reduce)
            oct_predict_labels = torch.argmax(oct_predicts, dim=0)
            self.oct_evaluator.update(
                np.expand_dims(oct_predict_labels.detach().cpu(), axis=0),
                label,
            )
            # For img
            img_predicts = F.softmax(img_outputs, dim=1)
            img_predicts = ta.fuse_predicts(
                img_predicts,
                reduce=self.cfg.test.augment.reduce)
            img_predict_labels = torch.argmax(img_predicts, dim=0)
            self.img_evaluator.update(
                np.expand_dims(img_predict_labels.detach().cpu(), axis=0),
                label,
            )
            # For combo
            outputs = torch.cat([oct_outputs, img_outputs], dim=0)
            if torch.isnan(outputs).any():
                logging.error("NAN")
            predicts = F.softmax(outputs, dim=1)
            predicts = ta.fuse_predicts(predicts, reduce=self.cfg.test.augment.reduce)
            pred_label = torch.argmax(predicts, dim=0)
            self.evaluator.update(
                np.expand_dims(pred_label.detach().cpu(), axis=0),
                label
            )
            # loss = self.loss_func((output_octs, output_imgs), labels)
            # # metric
            # self.loss_meter.update(loss, octs.size(0))
            # predict_octs = F.softmax(output_octs, dim=1)

            # predict_octs = ta.fuse_predicts(predict_octs, reduce=self.cfg.test.augment.reduce)
            # pred_label_octs = torch.argmax(predict_octs, dim=0)
            # self.oct_evaluator.update(
            #     pred_label_octs.unsqueeze(dim=0).detach().cpu().numpy(),
            #     labels.detach().cpu().numpy()
            # )
            # predict_imgs = F.softmax(output_imgs, dim=1)
            # predict_imgs = ta.fuse_predicts(predict_imgs, reduce=self.cfg.test.augment.reduce)
            # pred_label_imgs = torch.argmax(predict_imgs, dim=0)
            # self.img_evaluator.update(
            #     pred_label_imgs.unsqueeze(dim=0).detach().cpu().numpy(),
            #     labels.detach().cpu().numpy()
            # )
            # predicts = (predict_octs + predict_imgs) / 2
            # pred_label = torch.argmax(predicts, dim=0)
            # self.evaluator.update(
            #     pred_label.unsqueeze(dim=0).detach().cpu().numpy(),
            #     labels.detach().cpu().numpy()
            # )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            end = time.time()
        self.log_epoch_info(epoch, phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score(print=False, all_metric=False)[0]
