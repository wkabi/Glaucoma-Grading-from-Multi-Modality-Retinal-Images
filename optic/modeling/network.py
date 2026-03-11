import os
import os.path as osp
import torch
from torch import nn
import logging

from torch.functional import norm

from optic.modeling import resnet
from optic.modeling import resnet3d

logger = logging.getLogger(__name__)


def build_resnet(
    encoder_name="resnet50",
    num_classes=2,
    pretrained=True,
    has_dropout=True,
    has_batchnorm=True,
) -> nn.Module:
    """

    Args:
        encoder_name (str, optional): [description]. Defaults to "resnet50".
        num_classes (int, optional): [description]. Defaults to 2.
        pretrained (bool, optional): [description]. Defaults to True.
        checkpiont (bool, optional): checkpoint path.

    Returns:
        nn.Module: [description]
    """
    # model = models.resnet50(pretrained=pretrained)
    norm_layer = nn.Identity if not has_batchnorm else None

    model = eval("resnet." + encoder_name)(
        pretrained=pretrained,
        has_dropout=has_dropout,
        norm_layer=norm_layer
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def build_resnet3d(
    encoder_name="resnet50",
    num_classes=2,
    pretrained=True,
    has_dropout=True,
    has_batchnorm=True,
    shortcut_type="B",
    pretrained_dataset="23dataset"
) -> nn.Module:
    assert shortcut_type in {"A", "B"}
    assert pretrained_dataset in {"8dataset", "23dataset"}
    norm_layer = nn.Identity if not has_batchnorm else None
    model = eval("resnet3d." + encoder_name)(
        num_classes=num_classes,
        has_dropout=has_dropout,
        norm_layer=norm_layer,
        shortcut_type=shortcut_type
    )

    if pretrained:
        cached_dir = osp.join(os.getenv("HOME"), ".cache/optic/resnet3d")
        if pretrained_dataset == "8dataset":
            checkpoint_path = osp.join(cached_dir, encoder_name + ".pth")
        else:
            checkpoint_path = osp.join(cached_dir, encoder_name + "_23dataset.pth")
        assert osp.exists(checkpoint_path), "Pretained checkpoint not existed : {}".format(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        keys = list(checkpoint["state_dict"].keys())
        for key in keys:
            checkpoint["state_dict"][key[len("module."):]] = checkpoint["state_dict"].pop(key)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info("Succeed to load weights from {}".format(checkpoint_path))
        if missing_keys:
            logger.warn("Missing keys : {}".format(missing_keys))
        if unexpected_keys:
            logger.warn("Unexpected keys : {}".format(unexpected_keys))

    return model
