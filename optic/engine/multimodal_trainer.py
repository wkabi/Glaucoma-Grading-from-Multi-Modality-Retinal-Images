import logging
import numpy as np
import torch
import torch.nn.functional as F
import time
from terminaltables.ascii_table import AsciiTable
from hydra.utils import instantiate
import wandb
import json
from omegaconf.dictconfig import DictConfig

from optic.utils.misc import round_dict, get_lr
from optic.evaluation import GradeEvaluator, AverageMeter, LossMeter
from optic.engine.twostream_trainer import TwostreamTainer
from optic.modeling import MultimodalLoss, SAM
from optic.utils.checkpoint import load_checkpoint
import optic.data.test_augment as ta

logger = logging.getLogger(__name__)


class MultimodalTrainer(TwostreamTainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_model(self) -> None:
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        logger.info("Model initialized")
        if self.cfg.model.oct_model_checkpoint:
            load_checkpoint(
                self.cfg.model.oct_model_checkpoint,
                self.model.oct_model,
                self.device
            )
        if self.cfg.model.img_model_checkpoint:
            load_checkpoint(
                self.cfg.model.img_model_checkpoint,
                self.model.img_model,
                self.device
            )
        self.loss_func = instantiate(self.cfg.loss.object)
        self.loss_func.to(self.device)

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
        self.fuse_evaluator = GradeEvaluator(
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
        self.fuse_evaluator.reset()
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
            outputs, oct_outputs, img_outputs = self.model((octs, imgs))
            loss = self.loss_func((outputs, oct_outputs, img_outputs), labels)
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
            # oct evaluator
            oct_predicts = F.softmax(oct_outputs, dim=1)
            oct_pred_labels = torch.argmax(oct_predicts, dim=1)
            self.oct_evaluator.update(
                oct_pred_labels.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            # img evaluator
            img_predicts = F.softmax(img_outputs, dim=1)
            img_pred_labels = torch.argmax(img_predicts, dim=1)
            self.img_evaluator.update(
                img_pred_labels.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            # fuse evaluator
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            self.fuse_evaluator.update(
                pred_labels.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            # super combo
            predicts = (oct_predicts + img_predicts + predicts) / 3
            pred_labels = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                pred_labels.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_epoch_info(epoch)

    @torch.no_grad()
    def eval_epoch(self, data_loader, epoch, phase="Val"):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)
        end = time.time()
        for i, samples in enumerate(data_loader):
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            octs = self.preprocess_oct(samples["oct_img"])
            imgs = self.preprocess_image(samples["img"])
            label = np.expand_dims(samples["label"], axis=0)
            # forward
            if isinstance(octs, list):
                outputs_list = [self.model(x) for x in zip(octs, imgs)]
                outputs = torch.cat([out[0] for out in outputs_list], dim=0)
                oct_outputs = torch.cat([out[1] for out in outputs_list], dim=0)
                img_outputs = torch.cat([out[2] for out in outputs_list], dim=0)
            else:
                outputs, oct_outputs, img_outputs = self.model((octs, imgs))
            # loss = self.loss_func(outputs, labels)
            # metric
            # self.loss_meter.update(loss)
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
            # For fuse 
            predicts = F.softmax(outputs, dim=1)
            predicts = ta.fuse_predicts(
                predicts,
                reduce=self.cfg.test.augment.reduce
            )
            pred_labels = torch.argmax(predicts, dim=0)
            self.fuse_evaluator.update(
                np.expand_dims(pred_labels.detach().cpu(), axis=0),
                label
            )
            # For super combo
            outputs = torch.cat([oct_outputs, img_outputs, outputs], dim=0)
            predicts = F.softmax(outputs, dim=1)
            predicts = ta.fuse_predicts(
                predicts,
                reduce=self.cfg.test.augment.reduce
            )
            pred_labels = torch.argmax(predicts, dim=0)
            self.evaluator.update(
                np.expand_dims(pred_labels.detach().cpu(), axis=0),
                label
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            # if (i + 1) % self.cfg.log_period == 0:
            #     self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()
        self.log_epoch_info(epoch, phase)

        if self.cfg.train.best_model_metric == "fuse":
            return self.loss_meter.avg(0), self.fuse_evaluator.mean_score(print=False, all_metric=False)[0]
        else:
            return self.loss_meter.avg(0), self.evaluator.mean_score(print=False, all_metric=False)[0]

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
        fuse_curr_score = self.fuse_evaluator.curr_score()
        log_dict.update(dict(
            ("fuse_{}".format(key), val) for (key, val) in fuse_curr_score.items()
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
        if isinstance(self.loss_func, MultimodalLoss):
            log_dict["alpha"] = self.loss_func.alpha
        metric, _ = self.oct_evaluator.mean_score(print=False)
        log_dict.update(dict(
            ("oct_{}".format(key), val) for (key, val) in metric.items()
        ))
        metric, _ = self.img_evaluator.mean_score(print=False)
        log_dict.update(dict(
            ("img_{}".format(key), val) for (key, val) in metric.items()
        ))
        metric, _ = self.fuse_evaluator.mean_score(print=False)
        log_dict.update(dict(
            ("fuse_{}".format(key), val) for (key, val) in metric.items()
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

    def run(self):
        self.train()
