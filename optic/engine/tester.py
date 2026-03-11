import os.path as osp
import torch
import torch.nn.functional as F
import time
import logging
import json
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from optic.utils import round_dict
from optic.utils.wandb_helper import wandb_image_mask
from optic.utils.torch_helper import to_numpy
from optic.utils.checkpoint import load_checkpoint
from optic.evaluation import SegmentationEvaluator, AverageMeter

logger = logging.getLogger(__name__)


class Tester:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.device = torch.device(self.cfg.device)
        self.init()
        self.init_meter()
        self.init_wandb_or_not()

    def init(self):
        # data pipeline
        self.test_loader = instantiate(self.cfg.data.object.test)
        logger.info("Test data pipeline initialized")
        # modeling
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        logger.info("Model initialized")
        self.checkpoint_path = osp.join(self.work_dir, "best.pth")
        load_checkpoint(self.checkpoint_path, self.model, self.device)

    def init_meter(self):
        self.classes = self.test_loader.dataset.classes
        self.evaluator = SegmentationEvaluator(
            classes=self.classes,
            include_background=False
        )
        self.batch_time_meter = AverageMeter()
        logger.info("Meters initialized")

    def reset_meter(self):
        self.evaluator.reset()
        self.batch_time_meter.reset()

    def init_wandb_or_not(self):
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=["test"],
            )
            wandb.run.name = "{}-{}-{}".format(
                self.cfg.model.name, self.cfg.loss.name, wandb.run.id
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized")
        self.visualize = self.cfg.wandb.enable and self.cfg.test.visualize

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
        log_dict.update(self.evaluator.mean_score())
        logger.info(
            "Test Epoch\t{}".format(json.dumps(round_dict(log_dict)))
        )
        if self.cfg.wandb.enable:
            wandb_log_dict = dict(
                ("Test/{}".format(key), val) for (key, val) in log_dict.items()
            )
            wandb.log(wandb_log_dict)
        if len(self.evaluator.classes) > 1:
            self.evaluator.class_score()
            if self.cfg.wandb.enable:
                df = self.evaluator.class_score(return_dataframe=True)
                table = wandb.Table(dataframe=df)
                wandb.log({
                    "Test/class_score": table
                })

    @torch.no_grad()
    def test(self):
        self.reset_meter()
        self.model.eval()

        if self.visualize:
            image_mask_list = []

        max_iter = len(self.test_loader)
        end = time.time()
        for i, samples in enumerate(self.test_loader):
            inputs, labels = samples["img"].to(self.device), samples["mask"].to(self.device)
            # forward
            outputs = self.model(inputs)
            # metric
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
                self.log_iter_info(i, max_iter)
            end = time.time()
            if self.visualize:
                for j in range(pred_labels.shape[0]):
                    image_mask_list.append(
                        wandb_image_mask(
                            to_numpy(inputs[j]),
                            to_numpy(pred_labels[j]),
                            to_numpy(labels[j]),
                            self.classes
                        )
                    )
        self.log_epoch_info()
        if self.visualize:
            wandb.log({"segmentations": image_mask_list})
