import unittest
from hydra import compose, initialize
from omegaconf import OmegaConf


class TestNetwork(unittest.TestCase):
    def test(self):
        initialize(config_path="../configs")
        cfg = compose(config_name="defaults.yaml")
        print(OmegaConf.to_yaml(cfg))

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
