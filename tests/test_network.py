import unittest
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf


class TestNetwork(unittest.TestCase):
    def test(self):
        initialize(config_path="../configs")
        cfg = compose(config_name="defaults.yaml")
        print(OmegaConf.to_yaml(cfg))

        model = instantiate(cfg.model)
        print(model)

        optimizer = instantiate(cfg.optim, model.parameters())
        print(optimizer)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
