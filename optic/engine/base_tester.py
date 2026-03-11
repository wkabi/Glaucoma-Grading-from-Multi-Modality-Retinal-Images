import os.path as osp
import wandb
import torch
import logging
from abc import ABCMeta, abstractmethod
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from optic.utils.checkpoint import (
    load_train_checkpoint, save_checkpoint, load_checkpoint
)

logger = logging.getLogger(__name__)


class _BaseTester(metaclass=ABCMeta):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.device = torch.device(self.cfg.device)

    def build_data_loader(self) -> None:
        # data pipleline
        self.test_loader = instantiate(self.cfg.data.object.test)
        logger.info("Data pipeline initialized")

    def build_model(self, checkpoint: Optional[str] = None) -> None:
        # modeling
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        logger.info("Model initialized")
        self.checkpoint_path = osp.join(
            self.work_dir, "last.pth" if checkpoint=="" else checkpoint
        )
        load_checkpoint(self.checkpoint_path, self.model, self.device)

    def init_wandb_or_not(self) -> None:
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=["test"]
            )
            wandb.run.name = "{}-{}-{}".format(
                self.cfg.model.name, self.cfg.loss.name, wandb.run.id
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized")
        self.visualize = self.cfg.wandb.enable and self.cfg.test.visualize

    @abstractmethod
    def test(self):
        pass

    def run(self):
        self.test()
