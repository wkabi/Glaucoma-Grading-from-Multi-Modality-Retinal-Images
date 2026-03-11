import unittest
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from optic.data import FundusDataset, img_keypoint_augment

class TestFundusDataset(unittest.TestCase):
    # def test_seg(self):
    #     initialize(config_path="../configs", job_name="test_fundus_dataset")
    #     cfg = compose(config_name="defaults.yaml")
    #     print(OmegaConf.to_yaml(cfg.data))

    #     train_loader = instantiate(cfg.data.train)
    #     print(train_loader)
    #     val_loader = instantiate(cfg.data.val)
    #     print(val_loader)

    #     # data_root = "./data/GAMMA_training_data/training_data"
    #     # mode = "train"

    #     # dataset = FundusDataset(data_root, mode)

    #     # for i in tqdm(range(10)):
    #     #     img, target = dataset[i]

    #     self.assertTrue(True)

    # def test_grade(self):
    #     with initialize(config_path="../configs", job_name="test_fundus_grade"):
    #         cfg = compose(
    #             config_name="defaults.yaml",
    #             overrides=["data=fundus_grade_oct"]
    #         )
    #         print(OmegaConf.to_yaml(cfg.data))

    #         train_set = instantiate(cfg.data.object.train.dataset)
    #         for i in tqdm(range(2)):
    #             sample = train_set[i]
    #             print(type(sample))

    #     self.assertTrue(True)

    def test_localization(self):
        data_root="./data/GAMMA_training_data/training_data"
        dataset = FundusDataset(
            data_root=data_root,
            mode="train",
            return_fovea=True
        )

        print("without augmentation")
        for i in range(3):
            sample = dataset[i]
            print("===", i, "===")
            print(sample.keys())
            print(type(sample["img"]))
            print(sample["img"].shape)
            print(sample["fovea_x"], sample["fovea_y"])

        dataset = FundusDataset(
            data_root=data_root,
            mode="train",
            return_fovea=True,
            transformer=img_keypoint_augment()
        )

        print("")
        print("with augmentation")
        for i in range(3):
            print("===", i, "===")
            sample = dataset[i]
            print(sample.keys())
            print(type(sample["img"]))
            print(sample["img"].shape)
            print(sample["fovea_x"], sample["fovea_y"])

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
