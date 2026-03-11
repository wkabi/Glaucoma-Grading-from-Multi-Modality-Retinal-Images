import os.path as osp
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import logging
import json
import wandb
from collections import OrderedDict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from optic.modeling import CompoundLoss
from optic.utils import get_lr, round_dict
from optic.utils.checkpoint import (
    load_train_checkpoint, save_checkpoint, load_checkpoint
)
from optic.evaluation import SegmentationEvaluator, AverageMeter, LossMeter
from optic.engine import _BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(_BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.work_dir = self.cfg.work_dir
        self.device = torch.device(self.cfg.device)
        self.init()
        self.init_meter()
        self.init_wandb_or_not()

    def init(self):
        # data pipeline
        self.train_loader = instantiate(self.cfg.data.train)
        self.val_loader = instantiate(self.cfg.data.val)
        logger.info("Data pipeline initialized")
        # modeling
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        self.loss_func = instantiate(self.cfg.loss.object)
        self.loss_func.to(self.device)
        logger.info("Model initialized")
        # sovler
        self.optimizer = instantiate(self.cfg.optim.object, self.model.parameters())
        self.scheduler = instantiate(self.cfg.scheduler.object, self.optimizer)
        logger.info("Solver initialized")

    def init_wandb_or_not(self):
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=self.cfg.wandb.tags,
            )
            wandb.run.name = "{}-{}-{}".format(
                self.cfg.model.name, self.cfg.loss.name, wandb.run.id
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized")

    def init_meter(self):
        self.classes = self.train_loader.dataset.classes
        self.evaluator = SegmentationEvaluator(
            classes=self.classes,
            include_background=False
        )
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        if isinstance(self.loss_func, CompoundLoss):
            self.loss_meter = LossMeter(self.loss_func.num_terms)
        else:
            self.loss_meter = LossMeter()
        logger.info("Meters initialized")

    def reset_meter(self):
        self.evaluator.reset()
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()

    def start_or_resume(self):
        if self.cfg.train.resume:
            self.start_epoch, self.best_epoch, self.best_score = (
                load_train_checkpoint(
                    self.work_dir, self.device, self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )
            )
        else:
            self.start_epoch, self.best_epoch, self.best_score = 0, -1, None
        self.max_epoch = self.cfg.train.max_epoch

    def train_epoch(self, epoch):
        self.reset_meter()
        self.model.train()

        self.lr = get_lr(self.optimizer)
        max_iter = len(self.train_loader)

        end = time.time()
        for i, samples in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            inputs, labels = samples["img"].to(self.device), samples["mask"].to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
                # For compounding loss, make sure the first term is the overall loss
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            predicts = F.softmax(outputs, dim=1)
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
            inputs, labels = samples["img"].to(self.device), samples["mask"].to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            # metric
            self.loss_meter.update(loss)
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                pred_labels.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()
        self.log_epoch_info(epoch, phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score(main=True)

    def log_iter_info(self, iter, max_iter, epoch, phase="Train"):
        log_dict = OrderedDict()
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict["lr"] = self.lr
        log_dict.update(self.loss_meter.get_vals())
        if isinstance(self.loss_func, CompoundLoss):
            log_dict["alpha"] = self.loss_func.alpha
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
        log_dict = OrderedDict()
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, CompoundLoss):
            log_dict["alpha"] = self.loss_func.alpha
        log_dict.update(self.evaluator.mean_score())
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)
        if phase.lower() != "train" and len(self.evaluator.classes) > 1:
            self.evaluator.class_score()
            if self.cfg.wandb.enable:
                df = self.evaluator.class_score(return_dataframe=True)
                table = wandb.Table(dataframe=df)
                wandb.log({
                    "epoch": epoch,
                    "{}/class_score".format(phase): table
                })

    def train(self):
        self.start_or_resume()
        logger.info("Everything is perfect so far. Let's start training. Good luck!")
        for epoch in range(self.start_epoch, self.max_epoch):
            logger.info("=================")
            logger.info(" Start epoch {}".format(epoch + 1))
            logger.info("=================")
            self.train_epoch(epoch)
            val_loss, val_score = self.eval_epoch(self.val_loader, epoch, phase="Val")
            if self.best_score is None or val_score > self.best_score:
                self.best_score, self.best_epoch = val_score, epoch
                best_checkpint = True
            else:
                best_checkpint = False
            save_checkpoint(
                self.work_dir, self.model, self.optimizer, self.scheduler,
                epoch=epoch,
                best_checkpoint=best_checkpint,
                val_score=val_score
            )
            # logging best performance on val so far
            logger.info(
                "Epoch[{}]\tBest {} on Val : {:.4f} at epoch {}".format(
                    epoch + 1, self.evaluator.main_metric(),
                    self.best_score, self.best_epoch + 1
                )
            )
            if self.cfg.wandb.enable and best_checkpint:
                wandb.log({
                    "epoch": epoch,
                    "Val/best_epoch": self.best_epoch,
                    "Val/best_{}".format(self.evaluator.main_metric()): self.best_score,
                    "Val/best_class_score": wandb.Table(dataframe=self.evaluator.class_score(return_dataframe=True))
                })
            # run lr shceduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(
                    val_loss if self.scheduler.mode == "min"
                    else val_score
                )
            else:
                self.scheduler.step()
            if isinstance(self.loss_func, CompoundLoss):
                self.loss_func.adjust_alpha(epoch)
        if self.cfg.wandb.enable:
            wandb.save(osp.join(self.work_dir, "*.pth"))

    def test(self):
        logger.info("We are almost done : final testing ...")
        self.test_loader = instantiate(self.cfg.data.test)
        # test best pth
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(" Test at best epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Best epoch[{}] :".format(epoch + 1))
        load_checkpoint(
            osp.join(self.work_dir, "best.pth"), self.model, self.device
        )
        self.eval_epoch(self.test_loader, epoch, phase="Test/Best")
        # test last pth
        epoch = self.max_epoch - 1
        logger.info("#################")
        logger.info(" Test at last epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Last epoch[{}] :".format(epoch + 1))
        load_checkpoint(
            osp.join(self.work_dir, "last.pth"), self.model, self.device
        )
        self.eval_epoch(self.test_loader, epoch, phase="Test/Last")

    def run(self):
        self.train()
        self.test()
