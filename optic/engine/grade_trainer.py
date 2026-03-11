import logging
from shutil import copyfile
import time
from terminaltables.ascii_table import AsciiTable
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import json
import os.path as osp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from optic.engine import _BaseTrainer
from optic.modeling.twostream import TwostreamLoss
from optic.modeling.multimodal import MultimodalLoss
from optic.utils import get_lr, round_dict
from optic.utils.checkpoint import (
    load_train_checkpoint, save_checkpoint, load_checkpoint
)
from optic.evaluation import AverageMeter, LossMeter, GradeEvaluator
import optic.data.test_augment as ta

logger = logging.getLogger(__name__)


class GradeTrainer(_BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.build_data_loader()
        self.build_model()
        self.build_solver()
        self.build_meter()
        self.init_wandb_or_not()

    def init_wandb_or_not(self) -> None:
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=self.cfg.wandb.tags.split(","),
            )
            wandb.run.name = "{}-{}-{}-{}".format(
                wandb.run.id,
                self.cfg.model.name,
                self.cfg.loss.name,
                (self.cfg.data.train_split if "train_split" in self.cfg.data
                 else self.cfg.data.train_mode)
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

    def build_data_loader(self) -> None:
        # data pipleline
        self.train_loader = instantiate(self.cfg.data.object.train)
        self.val_loader = instantiate(self.cfg.data.object.val.dataset)
        logger.info("Data pipeline initialized")

    def build_meter(self):
        self.classes = self.train_loader.dataset.grade_classes
        self.num_classes = len(self.classes)
        self.evaluator = GradeEvaluator(
            num_classes=self.num_classes,
            classes=self.classes
        )
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_meter = LossMeter()
        logger.info("Meters initialized")

    def reset_meter(self):
        self.evaluator.reset()
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()

    def log_iter_info(self, iter, max_iter, epoch, phase="Train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_vals())
        log_dict.update(self.evaluator.curr_score())
        logger.info("{} Iter[{}/{}][{}]\t{}".format(
            phase, iter + 1, max_iter, epoch + 1,
            json.dumps(round_dict(log_dict))
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
        metric, table_data = self.evaluator.mean_score(print=False)
        log_dict.update(metric)
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if phase.lower() != "train":
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            if phase.lower() != "train":
                wandb_log_dict["{}/score_table".format(phase)] = wandb.Table(
                    columns=table_data[0], data=table_data[1:]
                )
            wandb.log(wandb_log_dict)

    def train_epoch(self, epoch):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        end = time.time()
        for i, samples in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            # inputs = samples["img"].to(self.device)
            if self.cfg.input_type == "oct":
                inputs = samples["oct_img"].to(self.device)
            elif self.cfg.input_type == "image":
                inputs = samples["img"].to(self.device)
            else:
                raise NotImplementedError("Invalid inputs setting : {}".format(self.cfg.input_type))
            labels = samples["label"].to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # metric
            self.loss_meter.update(loss.item(), inputs.size(0))
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
            if self.cfg.input_type == "oct":
                inputs = self.preprocess_oct(samples["oct_img"])
                num_sample = inputs.shape[0]
            elif self.cfg.input_type == "image":
                inputs = self.preprocess_image(samples["img"])
                num_sample = inputs.shape[0]
            elif self.cfg.input_type == "multimodal":
                volumes = self.preprocess_oct(samples["oct_img"])
                images = self.preprocess_image(samples["img"])
                inputs = (volumes, images)
                num_sample = inputs[0].shape[0]
            else:
                raise Exception("Invalid inputs setting : {}".format(self.cfg.input_type))
            # decouple samples
            # inputs = samples["img"].to(self.device)
            # if self.cfg.input_type == "oct":
            #     inputs = samples["oct_img"].to(self.device)
            # elif self.cfg.input_type == "image":
            #     inputs = samples["img"].to(self.device)
            # else:
            #     raise("Invalid inputs setting : {}".format(self.cfg.input_type))
            # inputs = samples["oct_img"].to(self.device)
            # labels = (
            #     torch.from_numpy(np.array(samples["label"]))
            #     .repeat(num_sample).to(self.device)
            # )
            label = samples["label"]
            # forward
            outputs = self.model(inputs)
            # loss = self.loss_func(outputs, labels)
            # metric
            # self.loss_meter.update(loss)
            predicts = F.softmax(outputs, dim=1)
            predicts = ta.fuse_predicts(predicts, reduce=self.cfg.test.augment.reduce)
            pred_label = torch.argmax(predicts, dim=0)
            self.evaluator.update(
                np.expand_dims(pred_label.detach().cpu(), axis=0),
                np.expand_dims(label, axis=0)
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            # if (i + 1) % self.cfg.log_period == 0:
            #     self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()
        self.log_epoch_info(epoch, phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score(print=False, all_metric=False)[0]

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
            if isinstance(self.loss_func, (TwostreamLoss, MultimodalLoss)):
                self.loss_func.adjust_alpha(epoch)

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
                    "Val/best_{}".format(self.evaluator.main_metric()): self.best_score,
                    "Val/best_score_table": self.evaluator.wandb_score_table()
                })
        if self.cfg.wandb.enable:
            # artifact = wandb.Artifact(
            #     type="model",
            #     name="%s-best" % wandb.run.name,
            # )
            # artifact.add_file(osp.join(self.work_dir, "best.pth"), "best.pth")
            # wandb.run.log_artifact(artifact)
            copyfile(
                osp.join(self.work_dir, "best.pth"),
                osp.join(self.work_dir, "{}-best.pth".format(wandb.run.name))
            )
            # wandb.save(osp.join(self.work_dir, "*.pth"))

    def preprocess_image(self, image):
        image = self.val_loader.preprocess_image(image)
        image = ta.augment(
            image,
            self.cfg.test.augment.img
        )
        image = torch.from_numpy(image).to(self.device)
        return image

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
            volume = np.concatenate(augs, axis=0)
        else:
            volume = ta.augment(
                volume,
                self.cfg.test.augment.oct
            )
        volume = np.expand_dims(volume, axis=1)
        return torch.from_numpy(volume).to(self.device)

    @torch.no_grad()
    def test(self):
        logger.info("We are almost done : final testing ...")
        self.test_loader = instantiate(self.cfg.data.object.test.dataset)
        # test last pth
        epoch = self.max_epoch - 1
        logger.info("#" * 20)
        logger.info(" Test at last epoch {}".format(epoch + 1))
        logger.info("#" * 20)
        logger.info("Last epoch[{}] :".format(epoch + 1))
        load_checkpoint(
            osp.join(self.work_dir, "last.pth"), self.model, self.device
        )

        self.reset_meter()
        self.model.eval()
        for i, sample in enumerate(self.test_loader):
            if self.cfg.input_type == "oct":
                inputs = self.preprocess_oct(sample["oct_img"])
            elif self.cfg.input_type == "image":
                inputs = self.preprocess_image(sample["img"])
            else:
                raise("Invalid inputs setting : {}".format(self.cfg.input_type))
            inputs = torch.from_numpy(inputs).to(self.device)
            # inputs = torch.unsqueeze(inputs, dim=1)
            label = sample["label"]
            # forward
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=1)
            predicts = ta.fuse_predicts(predicts, reduce=self.cfg.test.augment.reduce)
            # avg_pred = predicts[0]
            pred_label = torch.argmax(predicts, dim=0)
            self.evaluator.update(
                np.expand_dims(pred_label.detach().cpu(), axis=0),
                np.expand_dims(label, axis=0)
            )
        self.log_test_info(epoch)

    def log_test_info(self, epoch):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        metric, table_data = self.evaluator.mean_score(print=False)
        log_dict.update(metric)
        logger.info("Test Epoch[{}]\t{}".format(epoch + 1, json.dumps(round_dict(log_dict))))
        table = AsciiTable(table_data)
        logger.info("\n" + table.table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("Test/{}".format(key), value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict["Test/score_table"] = wandb.Table(
                columns=table_data[0], data=table_data[1:]
            )
            wandb.log(wandb_log_dict)

    def run(self):
        self.train()
        # self.test()
