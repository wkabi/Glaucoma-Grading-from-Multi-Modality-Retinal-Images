import logging
import torch
from hydra.utils import instantiate

from omegaconf.dictconfig import DictConfig
from optic.engine import GradeTester
from optic.modeling import ModelWithTemperature

logger = logging.getLogger(__name__)


class CalibrateNetwork(GradeTester):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def build_data_loader(self) -> None:
        self.test_loader = instantiate(self.cfg.data.object.test.dataset)
        logger.info("Data pipeline initialized")

    @torch.no_grad()
    def test(self):
        scalded_model = ModelWithTemperature(self.model, self.cfg)
        scalded_model.set_temperature(self.test_loader, "ece")

    def run(self):
        self.test()
