"""
load the fundus images from the data folder,
and extract the corresponding ground truth to generate training samples
"""
import random
import os
import os.path as osp
import h5py
import cv2
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data.dataset import Dataset
from typing import Callable, Optional, List, Any
from albumentations.augmentations.functional import normalize


class FundusDataset(Dataset):
    def __init__(self, data_root: str,
                 mode: str = "train",
                 transformer: Optional[Callable] = None,
                 oct_transformer: Optional[Callable] = None,
                 use_h5file: bool = False,
                 return_oct: bool = False,
                 return_mask: bool = False,
                 return_disc_region: bool = False,
                 return_disc_image: bool = False,
                 return_fovea: bool = False,
                 oct_depth_resize: str = "zoom",
                 oct_sample_step: int = 4,
                 oct_depth: int = 64) -> None:
        assert mode in {"train", "val", "trainval", "test", "all", "valtest"}
        super().__init__()
        # self.data_root = osp.expanduser(data_root)
        self.data_root = data_root
        self.mode = mode
        self.transformer = transformer
        self.oct_transformer = oct_transformer
        self.use_h5file = use_h5file
        self.return_oct = return_oct
        self.return_mask = return_mask
        self.oct_depth_resize = oct_depth_resize
        self.oct_sample_step = oct_sample_step
        self.oct_depth = oct_depth
        self.return_disc_region = return_disc_region
        self.return_disc_image = return_disc_image
        self.return_fovea = return_fovea

        self.img_dir = osp.join(self.data_root, "multi-modality_images")
        self.disc_img_dir = osp.join(self.data_root, "Disc_Image")
        self.mask_dir = osp.join(self.data_root, "Disc_Cup_Mask")
        self.load_list()
        if self.return_fovea:
            self.load_fovea_localization()
        self.seg_classes = ["_BG_", "OC", "OD"]
        self.grade_classes = ["0", "1", "2"]

    def load_list(self):
        split_file = osp.join(self.data_root, "{}.csv".format(self.mode))
        self.img_inds = []
        self.labels = []
        with open(split_file, "r") as f:
            for line in f:
                ind, label = line.strip().split(",")
                self.img_inds.append(int(ind))
                self.labels.append(int(label))

    def load_fovea_localization(self):
        path = osp.join(self.data_root, "fovea_localization.csv")
        self.localization = {}
        with open(path, "r") as f:
            for line in f:
                fields = line.strip().split(",")
                ind = int(fields[0])
                x = float(fields[1])
                y = float(fields[2])
                self.localization[ind] = (x, y)

    def get_img(self, ind: int) -> np.ndarray:
        img_path = osp.join(
            self.img_dir, "{:04d}".format(ind), "{:04d}.jpg".format(ind)
        )
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_oct_img(self, ind: int) -> np.ndarray:
        oct_dir = osp.join(
            self.img_dir, "{:04d}".format(ind), "{:04d}".format(ind)
        )
        oct_series_list = sorted(
            os.listdir(oct_dir), key=lambda x: int(x.split("_")[0])
        )
        oct_series_0 = cv2.imread(
            osp.join(oct_dir, oct_series_list[0]), cv2.IMREAD_GRAYSCALE
        )
        oct_img = np.zeros(
            (oct_series_0.shape[0], oct_series_0.shape[1], len(oct_series_list)),
            dtype="uint8"
        )
        for i, name in enumerate(oct_series_list):
            oct_img[:, :, i] = cv2.imread(
                osp.join(oct_dir, name), cv2.IMREAD_GRAYSCALE
            )
        return oct_img

    def get_mask(self, ind) -> np.ndarray:
        # In the ground truth, a pixel value of 0 is the optic cup (class 0),
        # a pixel value of 128 is the optic disc (class 1),
        # and a pixel value of 255 is the background (class 2).
        mask_path = osp.join(self.mask_dir, "{:04d}.png".format(ind))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.clear_mask_border(mask)
        mask[mask == 0] = 1
        mask[mask == 128] = 2
        mask[mask == 255] = 0
        return mask

    def clear_mask_border(self, mask, size=5):
        height, width = mask.shape
        mask[:size, :] = 255
        mask[height-size:, :] = 255
        mask[:, :size] = 255
        mask[:, width-size:] = 255

    def get_disc_region(self, ind, expand_ratio=0.5) -> np.ndarray:
        # pixel value of 128 is the optic disc(class 1)
        mask_path = osp.join(self.mask_dir, "{:04d}.png".format(ind))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.clear_mask_border(mask)
        mask[mask != 255] = 1
        mask[mask == 255] = 0

        height, width = mask.shape

        ind = np.nonzero(mask)
        xmin, xmax = np.min(ind[1]), np.max(ind[1])
        ymin, ymax = np.min(ind[0]), np.max(ind[0])

        expand_width = int((xmax - xmin + 1) * expand_ratio / 2)
        expand_height = int((ymax - ymin + 1) * expand_ratio / 2)

        xmin = max(xmin - expand_width, 0)
        xmax = min(xmax + expand_width, width - 1)
        ymin = max(ymin - expand_height, 0)
        ymax = min(ymax + expand_height, height - 1)

        return (xmin, ymin, xmax, ymax)

    def get_disc_img(self, ind) -> np.ndarray:
        img_path = osp.join(
            self.disc_img_dir, "{:04d}.png".format(ind)
        )
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_h5file(self, ind) -> dict:
        file_path = osp.join(self.data_root, "h5", "{:04d}.h5".format(ind))
        sample = {"id": ind}
        with h5py.File(file_path, "r") as h5f:
            sample["img"] = h5f["img"][:]
            if self.return_oct:
                sample["oct_img"] = h5f["oct_img"][:]
            if self.return_mask:
                sample["mask"] = h5f["mask"]
        return sample

    def preprocess_image(
        self,
        image,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    ):
        return normalize(image, mean, std, max_pixel_value)

    def resize_oct_depth(self, volume):
        if self.oct_depth_resize == "zoom":
            scale_z = self.oct_depth / 256
            volume = ndimage.interpolation.zoom(volume, (1.0, 1.0, scale_z))
        elif self.oct_depth_resize == "sample":
            step_size = 256 // self.oct_depth
            if self.mode in ("train", "trainval"):
                start = random.randint(0, step_size - 1)
            else:
                start = 0
            z = np.arange(start, 256, step_size)
            volume = volume[:, :, z]
        elif self.oct_depth_resize == "multisample":
            # only for test phase
            step_size = 256 // self.oct_depth
            volumes = []
            for start in range(0, step_size, self.oct_sample_step):
                z = np.arange(start, 256, step_size)
                volumes.append(volume[:, :, z])
            volume = np.stack(volumes)
        else:
            raise("Invalid oct depth resize method : {}".format(self.oct_depth_resize))

        return volume

    def preprocess_oct(self, volume, mean=0.284, std=0.087, max_pixel_value=255.0) -> np.ndarray:
        volume = self.resize_oct_depth(volume)
        # scale_z = self.oct_depth / 256
        # volume = ndimage.interpolation.zoom(volume, (1.0, 1.0, scale_z))
        volume = volume.astype(np.float32) / max_pixel_value

        volume = (volume - mean) / std

        return volume

    def __getitem__(self, i: int) -> List[Any]:
        ind = self.img_inds[i]
        label = self.labels[i]
        if self.use_h5file:
            sample = self.load_h5file(ind)
            sample["label"] = label
        else:
            img = self.get_img(ind)
            sample = {"img": img, "label": label, "id": ind}
            if self.return_oct:
                sample["oct_img"] = self.get_oct_img(ind)
            if self.return_mask:
                sample["mask"] = self.get_mask(ind)
            if self.return_disc_region:
                sample["disc_bbox"] = self.get_disc_region(ind)
            if self.return_disc_image:
                sample["img"] = self.get_disc_img(ind)
        if self.return_fovea:
            sample["fovea_x"], sample["fovea_y"] = self.localization[ind]
            sample["has_fovea"] = 1

        if self.transformer is not None:
            if self.return_mask:
                result = self.transformer(
                    image=sample["img"], mask=sample["mask"]
                )
                sample["img"] = result["image"]
                sample["mask"] = result["mask"].long()
            elif self.return_fovea:
                keypoints = [(sample["fovea_x"], sample["fovea_y"])]
                # for _ in range(5):
                while True:
                    # Try to reduce the possiblility of getting images with invisible keyponts
                    result = self.transformer(
                        image=sample["img"], keypoints=keypoints
                    )
                    if len(result["keypoints"]) > 0:
                        break
                sample["img"] = result["image"]
                if len(result["keypoints"]) > 0:
                    sample["fovea_x"], sample["fovea_y"] = result["keypoints"][0]
                    sample["fovea_x"] /= sample["img"].shape[1]
                    sample["fovea_y"] /= sample["img"].shape[2]
                    sample["has_fovea"] = 1
                else:
                    sample["fovea_x"], sample["fovea_y"] = -1, -1
                    sample["has_fovea"] = 0
            else:
                result = self.transformer(image=sample["img"])
                sample["img"] = result["image"]

        if self.oct_transformer is not None:
            volume = self.preprocess_oct(sample["oct_img"])
            result = self.oct_transformer(image=volume)
            sample["oct_img"] = torch.unsqueeze(result["image"], dim=0)

        return sample

    def __len__(self):
        return len(self.img_inds)

    def __repr__(self) -> str:
        return (
            "FundusDataset(data_root={}, mode={})\tSamples : {}".format(
                self.data_root, self.mode, self.__len__()
            )
        )
