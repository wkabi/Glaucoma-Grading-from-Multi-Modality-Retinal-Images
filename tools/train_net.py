from optic.engine.fovea_trainer import FoveaTrainer
import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from optic.engine import (
    Trainer, GradeTrainer, MultimodalTrainer, TwostreamTainer,
    FoveaTrainer
)
from optic.utils.misc import set_random_seed


logger = logging.getLogger(__name__)


TRAINERS = {
    "segment": Trainer,
    "grade": GradeTrainer,
    "multimodal": MultimodalTrainer,
    "twostream": TwostreamTainer,
    "fovea": FoveaTrainer
}


@hydra.main(config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    with open_dict(cfg):
        cfg.work_dir = os.getcwd()
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    set_random_seed(
        cfg.seed if cfg.seed is not None else None,
        deterministic=True if cfg.seed is not None else False
    )

    trainer = TRAINERS[cfg.task](cfg)
    trainer.run()

    logger.info("Job complete !\n")


if __name__ == "__main__":
    main()
