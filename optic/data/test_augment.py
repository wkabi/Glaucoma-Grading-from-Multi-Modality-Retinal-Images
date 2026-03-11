"""Test time augmentation for image classification and segmentation tasks"""
import numpy as np
from omegaconf.dictconfig import DictConfig
import torch
import cv2
from albumentations.augmentations.geometric.functional import smallest_max_size, resize
from albumentations.augmentations.crops.functional import crop, center_crop
from albumentations.augmentations.functional import hflip
from typing import List, Union


def fivecrop(
    image: np.ndarray,
    resize_max_size: int = 256,
    crop_size: List[int] = [224, 224],
    flip: bool = True
) -> np.ndarray:
    """five crop image augment

    Args:
        image (np.ndarray): H x W x C
        resize_max_size (int, optional): Defaults to 256.
        crop_size (List[int], optional): Defaults to [224, 224].
        flip (bool): Defaults to True

    Returns:
        np.ndarray: 5/10 x C x crop_size
    """
    resized_image = smallest_max_size(image, max_size=resize_max_size, interpolation=cv2.INTER_LINEAR)
    height, width = resized_image.shape[0], resized_image.shape[1]
    crop_height, crop_width = crop_size
    assert (
        crop_height <= height and crop_width <= width
    ), "crop size {} does not match smalest_max_size {}".format(crop_size, resize_max_size)

    tl = crop(resized_image, 0, 0, crop_width, crop_height)
    tr = crop(resized_image, width - crop_width, 0,  width, crop_height)
    bl = crop(resized_image, 0, height - crop_height, crop_width, height)
    br = crop(resized_image, width - crop_width, height - crop_height, width, height)

    cc = center_crop(resized_image, crop_height, crop_width)

    if flip:
        ret = np.stack(
            (cc, tl, tr, bl, br, hflip(cc), hflip(tl), hflip(tr), hflip(bl), hflip(br))
        )
    else:
        ret = np.stack((cc, tl, tr, bl, br))

    ret = np.einsum("nijc->ncij", ret)
    return ret


def centercrop(
    image: np.ndarray,
    resize_max_size: int = 256,
    crop_size: List[int] = [224, 224],
    flip: bool = True
) -> np.ndarray:
    resized_image = smallest_max_size(image, max_size=resize_max_size, interpolation=cv2.INTER_LINEAR)
    height, width = resized_image.shape[0], resized_image.shape[1]
    crop_height, crop_width = crop_size
    assert (
        crop_height <= height and crop_width <= width
    ), "crop size {} does not match smalest_max_size {}".format(crop_size, resize_max_size)

    cc = center_crop(resized_image, crop_height, crop_width)

    if flip:
        ret = np.stack((cc, hflip(cc)))
    else:
        ret = np.expand_dims(cc, axis=0)
    ret = np.einsum("nijc->ncij", ret)

    return ret


def resize_and_centercrop(
    image: np.ndarray,
    resize_max_size: int = 256,
    crop_size: List[int] = [224, 224],
    flip: bool = True
) -> np.ndarray:
    resized_image = smallest_max_size(image, max_size=resize_max_size, interpolation=cv2.INTER_LINEAR)
    height, width = resized_image.shape[0], resized_image.shape[1]
    crop_height, crop_width = crop_size
    assert (
        crop_height <= height and crop_width <= width
    ), "crop size {} does not match smalest_max_size {}".format(crop_size, resize_max_size)

    rs = resize(resized_image, crop_height, crop_width)
    cc = center_crop(resized_image, crop_height, crop_width)

    if flip:
        ret = np.stack((rs, cc, hflip(rs), hflip(cc)))
    else:
        ret = np.stack((rs, cc))
    ret = np.einsum("nijc->ncij", ret)

    return ret


def resize_and_flip(
    image: np.ndarray,
    dst_size: List[int] = [224, 224],
    flip: bool = False
):
    height, width = dst_size
    rs = resize(image, height, width)

    if flip:
        ret = np.stack((rs, hflip(rs)))
    else:
        ret = np.expand_dims(rs, axis=0)
    ret = np.einsum("nijc->ncij", ret)

    return ret


def resize_smallest_and_flip(
    image: np.ndarray,
    dist_size: Union[List[int], int] = [224, 256],
    flip: bool = False
):
    if isinstance(dist_size, int):
        dist_size = [dist_size]
    ret = []
    for ss in dist_size:
        rs = smallest_max_size(
            image, max_size=ss, interpolation=cv2.INTER_LINEAR
        )
        if flip:
            rs = np.stack((rs, hflip(rs)))
        else:
            rs = np.expand_dims(rs, axis=0)
        rs = np.einsum("nijc->ncij", rs)
        ret.append(rs)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 3:
        mean = image.mean()
        std = image.std()
        denominator = np.reciprocal(std, dtype=np.float32)
        out = (image - mean) * denominator
    else:
        mean = image.mean(axis=[1, 2, 3])

    return out


def geometric_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute geometric mean along given dimension.
    This implementation assume values are in range (0...1) (Probabilities)
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce
    Returns:
        Tensor
    """
    return x.log().mean(dim=dim).exp()


def fuse_predicts(x: torch.Tensor, reduce: str = "mean"):
    if reduce == "mean":
        return x.mean(dim=0)
    elif reduce == "max":
        return x.max(dim=0)[0]
    elif reduce == "gmean":
        return geometric_mean(x, dim=0)
    elif reduce == "tsharpen":
        return (x ** 0.5).mean(dim=0)
    else:
        raise NotImplementedError("Invalid reduce method : {}".format(reduce))



def augment(input: np.ndarray, aug_cfg: DictConfig):
    # import ipdb; ipdb.set_trace()
    if aug_cfg.method == "fivecrop":
        return fivecrop(
            input,
            resize_max_size=aug_cfg.resize_small_size,
            crop_size=aug_cfg.crop_size,
            flip=aug_cfg.flip
        )
    elif aug_cfg.method == "centercrop":
        return centercrop(
            input,
            resize_max_size=aug_cfg.resize_small_size,
            crop_size=aug_cfg.crop_size,
            flip=aug_cfg.flip
        )
    elif aug_cfg.method == "resize_and_centercrop":
        return resize_and_centercrop(
            input,
            resize_max_size=aug_cfg.resize_small_size,
            crop_size=aug_cfg.crop_size,
            flip=aug_cfg.flip
        )
    elif aug_cfg.method == "resize":
        return resize_and_flip(
            input,
            dst_size=aug_cfg.crop_size,
            flip=aug_cfg.flip
        )
        # ret = resize(
        #     input,
        #     height=aug_cfg.crop_size[0],
        #     width=aug_cfg.crop_size[1],
        # )
        # ret = np.expand_dims(ret, axis=0)
        # ret = np.einsum("nijc->ncij", ret)
        # return ret
    elif aug_cfg.method == "resize_smallest":
        return resize_smallest_and_flip(
            input,
            dist_size=aug_cfg.resize_small_size,
            flip=aug_cfg.flip
        )
    else:
        raise NotImplementedError(
            "Invalid agument method : {}".format(aug_cfg.method)
        )
