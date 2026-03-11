import logging
from shutil import copyfile
import os.path as osp
import time
import json
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf.dictconfig import DictConfig

from .grade_trainer import GradeTrainer
from optic.evaluation import FoveaEvaluator, LossMeter, AverageMeter
from optic.utils.misc import get_lr, round_dict
import optic.data.test_augment as ta
from optic.utils.checkpoint import (
    load_train_checkpoint, save_checkpoint, load_checkpoint
)

logger = logging.getLogger(__name__)


class FoveaTrainer(GradeTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_meter(self):
        self.evaluator = FoveaEvaluator()
        # self.loss_meter = LossMeter(
        #     num_terms=len(self.loss_func.names),
        #     names=self.loss_func.names
        # )
        self.loss_meter = LossMeter()
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        logger.info("Meters initialized")

    def train_epoch(self, epoch):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        end = time.time()
        for i, samples in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            inputs = samples["img"].to(self.device)
            labels = torch.cat(
                (samples["fovea_x"].unsqueeze(1), samples["fovea_y"].unsqueeze(1), samples["has_fovea"].unsqueeze(1)),
                dim=1
            ).float().to(self.device)
            labels = labels[:, :2]
            # foward
            outputs = self.model(inputs)
            outputs = F.sigmoid(outputs)
            loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            # pred_coord = outputs[:, :2]
            # pred_has_fovea = (F.sigmoid(outputs[:, 2]) > 0.5).float()
            # preds = torch.cat((pred_coord, pred_has_fovea.unsqueeze(1)), dim=1)
            preds = outputs
            self.evaluator.update(
                preds.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
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
        # log_dict["alpha"] = self.loss_func.alpha
        log_dict.update(self.evaluator.curr_score())
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
        # log_dict["alpha"] = self.loss_func.alpha
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

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
            self.data_time_meter.update(time.time() - end)
            orig_height, orig_width = samples["img"].shape[:2]
            inputs = self.preprocess_image(samples["img"])
            # num_samples = inputs.shape[0]
            labels = np.array([
                samples["fovea_x"] / orig_width,
                samples["fovea_y"] / orig_height,
                # samples["has_fovea"]
            ])
            labels = np.expand_dims(labels, axis=0)
            labels = torch.from_numpy(labels).float().to(self.device)
            # forard
            if isinstance(inputs, list):
                outputs = [self.model(x) for x in inputs]
                outputs = torch.cat(outputs, dim=0)
                outputs = F.sigmoid(outputs).mean(dim=0).unsqueeze(dim=0)
            else:
                outputs = self.model(inputs)
                outputs = F.sigmoid(outputs)
            loss = self.loss_func(outputs, labels)
            # metric
            self.loss_meter.update(loss)
            # pred_coord = outputs[:, :2]
            # pred_has_fovea = (F.sigmoid(outputs[:, 2]) > 0.5).type(torch.int32)
            # preds = torch.cat((pred_coord, pred_has_fovea.unsqueeze(1)), dim=1)
            preds = outputs
            self.evaluator.update(
                preds.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            self.batch_time_meter.update(time.time() - end)
        self.log_epoch_info(epoch, phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score(all_metric=False)

    def train(self):
        self.start_or_resume()
        logger.info(
            "Everything is perfect so far. Let's start training. Good luck!"
        )
        for epoch in range(self.start_epoch, self.max_epoch):
            logger.info("=" * 20)
            logger.info(" Start epoch {}".format(epoch + 1))
            logger.info("=" * 20)
            self.train_epoch(epoch)
            val_loss, val_score = self.eval_epoch(self.val_loader, epoch, phase="Val")
            # run lr scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(
                    val_loss if self.scheduler.mode == "min"
                    else val_score
                )
            else:
                self.scheduler.step()

            if self.best_score is None or val_score > self.best_score:
                self.best_score, self.best_epoch = val_score, epoch
                best_checkpoint = True
            else:
                best_checkpoint = False
            save_checkpoint(
                self.work_dir, self.model, self.optimizer, self.scheduler,
                epoch=epoch,
                best_checkpoint=best_checkpoint,
                val_score=val_score,
                keep_checkpoint_num=self.cfg.train.keep_checkpoint_num
            )
            # logging best performance on val so far
            logger.info(
                "Epoch[{}]\tBest {} on Val : {:.4f} at epoch {}".format(
                    epoch + 1, self.evaluator.main_metric(),
                    self.best_score, self.best_epoch + 1
                )
            )
            if self.cfg.wandb.enable and best_checkpoint:
                wandb.log({
                    "epoch": epoch,
                    "Val/best_epoch": self.best_epoch,
                    "Val/best_{}".format(self.evaluator.main_metric()): self.best_score
                })
        if self.cfg.wandb.enable:
            copyfile(
                osp.join(self.work_dir, "best.pth"),
                osp.join(self.work_dir, "{}-best.pth".format(wandb.run.name))
            )
            artifact = wandb.Artifact(
                type="model",
                name="%s-best" % wandb.run.name,
            )
            artifact.add_file(
                osp.join(self.work_dir, "{}-best.pth".format(wandb.run.name)),
                "{}-best.pth".format(wandb.run.name)
            )
            wandb.run.log_artifact(artifact)
