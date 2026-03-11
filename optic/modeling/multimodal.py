import logging
import torch
import torch.nn as nn
from typing import Tuple, List
import torch.nn.functional as F

from optic.utils.torch_helper import kl_div

logger = logging.getLogger(__name__)


class MultimodalModel(nn.Module):
    def __init__(
        self,
        oct_model: nn.Module,
        img_model: nn.Module,
        num_classes: int,
        fuse_method: str = "cat",
        has_dropout: bool = True,
        from_scratch: bool = False
    ):
        super().__init__()
        self.oct_model = oct_model
        self.oct_feature_size = self.oct_model.fc.in_features
        self.img_model = img_model
        self.img_feature_size = self.img_model.fc.in_features
        self.num_classes = num_classes

        self.fuse_method = fuse_method
        self.has_dropout = has_dropout
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        if self.fuse_method == "cat":
            self.fuse_fc = nn.Sequential(
                nn.Dropout(p=0.5) if self.has_dropout else nn.Identity(),
                nn.Linear(
                    self.oct_feature_size + self.img_feature_size,
                    self.num_classes
                )
            )
            # self.fuse_fc = nn.Linear(
            #     self.oct_feature_size + self.img_feature_size, num_classes
            # )
        elif self.fuse_method == "add":
            assert (
                self.img_feature_size == self.oct_feature_size
            ), "The feature size of img and oct is not matched for add fusing"
            self.fuse_fc = nn.Sequential(
                nn.Dropout(p=0.5) if self.has_dropout else nn.Identity(),
                nn.Linear(self.oct_feature_size, self.num_classes)
            )
        else:
            raise NotImplementedError("Unsupported fuse method : {}".format("self.fuse_method"))

        if from_scratch:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm3d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Tuple[torch.Tensor]):
        volumes, images = inputs
        oct_feature, oct_output = self.oct_model.forward_feature_logit(volumes)
        img_feature, img_output = self.img_model.forward_feature_logit(images)

        if self.fuse_method == "cat":
            x = torch.cat((oct_feature, img_feature), dim=1)
        elif self.fuse_method == "add":
            x = oct_feature + img_feature
        else:
            raise NotImplementedError("Unsupported fuse method : {}".format("self.fuse_method"))

        fuse_output = self.fuse_fc(x)

        return fuse_output, oct_output, img_output


class MultimodalLoss(nn.Module):
    def __init__(
        self,
        weight: List[float] = [1.0, 1.0],
        alpha: float = 1.0,
        max_alpha: float = 10.0,
        temp: float = 1.0,
        step_size: int = 0,
        step_factor: float = 1.0
    ):
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.temp = temp
        self.step_size = step_size
        self.step_factor = step_factor

    @property
    def names(self):
        return "loss", "oct_loss", "img_loss", "fuse_loss"

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.step_factor, self.max_alpha)
            logger.info(
                "{}: Adjust the alpha of KL : {:.3g} -> {:.3g}".format(self.__class__, curr_alpha, self.alpha)
            )


class MultimodalCELoss(MultimodalLoss):
    def forward(self, outputs: Tuple[torch.Tensor], labels):
        fuse_outputs, oct_outputs, img_outputs = outputs

        oct_loss = F.cross_entropy(oct_outputs, labels)
        img_loss = F.cross_entropy(img_outputs, labels)
        fuse_loss = F.cross_entropy(fuse_outputs, labels)

        loss = (
            self.weight[0] * oct_loss
            + self.weight[1] * img_loss
            + self.alpha * fuse_loss
        )

        return loss, oct_loss, img_loss, fuse_loss


class MultimodalCeKlLoss(MultimodalLoss):
    def __init__(
        self,
        weight: List[float] = [1.0, 1.0],
        alpha: float = 1.0,
        max_alpha: float = 10.0,
        temp: float = 1.0,
        step_size: int = 0,
        step_factor: float = 1.0,
        kl_lambda: float = 0.1
    ) -> None:
        super().__init__(
            weight, alpha, max_alpha, temp, step_size, step_factor
        )
        self.kl_lambda = kl_lambda

    @property
    def names(self):
        return "loss", "oct_loss", "img_loss", "fuse_loss", "kl_loss"

    def forward(self, outputs: Tuple[torch.Tensor], labels):
        fuse_outputs, oct_outputs, img_outputs = outputs

        oct_loss = F.cross_entropy(oct_outputs, labels)
        img_loss = F.cross_entropy(img_outputs, labels)
        fuse_loss = F.cross_entropy(fuse_outputs, labels)

        loss_dist = kl_div(
            F.log_softmax(oct_outputs / self.temp, dim=1).exp(),
            F.log_softmax(img_outputs / self.temp, dim=1).exp()
        )

        loss = (
            self.weight[0] * oct_loss
            + self.weight[1] * img_loss
            + self.alpha * fuse_loss
            + self.kl_lambda * loss_dist
        )

        return loss, oct_loss, img_loss, fuse_loss, loss_dist
