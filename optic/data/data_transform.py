from albumentations.augmentations.crops.transforms import RandomCrop, RandomResizedCrop
from albumentations.augmentations.geometric.functional import scale
from albumentations.augmentations.geometric.resize import Resize, SmallestMaxSize
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.core.composition import OneOf, Sequential
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def fundus_augment(size, is_train: bool = True) -> A.Compose:
    height, width = size
    if is_train:
        transformer = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf([
                A.RandomGamma(),
                A.RandomBrightnessContrast(),
                ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
            A.ShiftScaleRotate(),
            A.Resize(height, width, always_apply=True),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.Resize(height, width, always_apply=True),
            A.Normalize(),
            ToTensorV2()
        ])

    return transformer


def img_augment(is_train: bool = True) -> A.Compose:
    """augmentation for global image
    """
    if is_train:
        transformer = A.Compose([
            # A.ShiftScaleRotate(p=0.5),
            A.OneOf(
                [
                    A.Sequential([
                        A.SmallestMaxSize(max_size=256),
                        A.RandomCrop(height=224, width=224),
                        # A.RandomResizedCrop(
                        #     height=224, width=224, scale=(0.5, 1.0),
                        #     ratio=(0.75, 1.33)
                        # )
                    ]),
                    A.Sequential([
                        A.SmallestMaxSize(max_size=480),
                        A.RandomCrop(height=224, width=224),
                        # A.RandomResizedCrop(
                        #     height=224, width=224, scale=(0.5, 1.0),
                        #     ratio=(0.75, 1.33)
                        # )
                    ])
                ],
                p=1
            ),
            # A.SmallestMaxSize(max_size=256),
            # A.RandomResizedCrop(
            #     height=224, width=224, scale=(0.90, 1.0), ratio=(0.90, 1.1)
            # ),
            A.OneOf(
                [
                    A.RandomGamma(),
                    A.RandomBrightnessContrast()
                ],
                p=0.5
            ),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            # A.SmallestMaxSize(max_size=256),
            # A.CenterCrop(height=224, width=224),
            A.Resize(height=224, width=224),
            A.Normalize(),
            ToTensorV2()
        ])
    return transformer


def img_keypoint_augment(is_train: bool = True) -> A.Compose:
    if is_train:
        transformer = A.Compose(
            [
                A.OneOf(
                    [
                        A.Sequential([
                            A.SmallestMaxSize(max_size=224),
                            A.RandomCrop(height=224, width=224),
                        ]),
                        A.Sequential([
                            A.SmallestMaxSize(max_size=256),
                            A.RandomCrop(height=224, width=224),
                        ]),
                        A.Sequential([
                            A.SmallestMaxSize(max_size=384),
                            A.RandomCrop(height=224, width=224),
                        ])
                    ],
                    p=1
                ),
                # A.OneOf(
                #     [
                #         A.RandomGamma(),
                #         A.RandomBrightnessContrast()
                #     ],
                #     p=0.5
                # ),
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Normalize(),
                ToTensorV2()
            ],
            keypoint_params=A.KeypointParams(format="xy")
        )
    else:
        transformer = A.Compose(
            [
                A.Resize(height=224, width=224),
                A.Normalize(),
                ToTensorV2()
            ],
            keypoint_params=A.KeypointParams(format="xy")
        )
    return transformer


class OneVolumeNormalize(A.ImageOnlyTransform):
    def __init__(
        self,
        always_apply=False,
        p=1.0
    ):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        image = image.astype(np.float32)
        mean = image.mean()
        std = image.std()
        denominator = np.reciprocal(std, dtype=np.float32)
        out = (image - mean) * denominator
        return out


def oct_augment(is_train: bool = True) -> A.Compose:
    """get data-augment transformer for oct data

    Args:
        is_train (bool, optional): train mode or not. Defaults to True.

    Returns:
        A.Compose: The callable transformer
    """
    if is_train:
        transformer = A.Compose([
            A.RandomBrightnessContrast(),
            A.OneOf(
                [
                    A.Sequential([
                        A.SmallestMaxSize(max_size=224),
                        A.RandomCrop(height=224, width=224),
                        # A.RandomResizedCrop(
                        #     height=224, width=224, scale=(0.5, 1.0),
                        #     ratio=(0.75, 1.33)
                        # )
                    ]),
                    A.Sequential([
                        A.SmallestMaxSize(max_size=480),
                        A.RandomResizedCrop(
                            height=224, width=224, scale=(0.5, 1.0),
                            ratio=(0.75, 1.33)
                        )
                    ])
                ],
                p=1
            ),
            # A.SmallestMaxSize(max_size=256),
            # A.RandomResizedCrop(height=224, width=224, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
            A.HorizontalFlip(),
            # A.VerticalFlip(),
            # A.Normalize(),
            # OneVolumeNormalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.Resize(height=224, width=224),
            # OneVolumeNormalize(),
            # A.Normalize(),
            ToTensorV2()
        ])

    return transformer
