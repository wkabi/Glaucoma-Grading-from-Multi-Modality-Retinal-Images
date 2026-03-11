import wandb
import torch
import logging
from abc import ABCMeta, abstractmethod
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from optic.utils.checkpoint import (
    load_train_checkpoint, save_checkpoint, load_checkpoint
)
from optic.modeling import SAM

logger = logging.getLogger(__name__)


class _BaseTrainer(metaclass=ABCMeta):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.device = torch.device(self.cfg.device)

    def build_data_loader(self) -> None:
        # data pipleline
        self.train_loader = instantiate(self.cfg.data.object.train)
        self.val_loader = instantiate(self.cfg.data.object.val)
        logger.info("Data pipeline initialized")

    def build_model(self) -> None:
        # modeling
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        self.loss_func = instantiate(self.cfg.loss.object)
        self.loss_func.to(self.device)
        logger.info(self.loss_func)
        logger.info("Model initialized")

    def build_solver(self) -> None:
        self.optimizer = instantiate(self.cfg.optim.object, self.model.parameters())
        if isinstance(self.optimizer, SAM):
            self.scheduler = instantiate(self.cfg.scheduler.object, self.optimizer.base_optimizer)
        else:
            self.scheduler = instantiate(self.cfg.scheduler.object, self.optimizer)
        logger.info("Solver initialized")

    def init_wandb_or_not(self) -> None:
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=self.cfg.wandb.tags.split(","),
            )
            wandb.run.name = "{}-{}-{}".format(
                wandb.run.id, self.cfg.model.name, self.cfg.loss.name
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

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

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def run(self):
        self.train()
        self.test()
