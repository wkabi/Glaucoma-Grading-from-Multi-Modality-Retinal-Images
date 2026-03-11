import torch
from torch import nn
from typing import Optional, List


def weighted_ce(reduction="mean", weight: Optional[List] = None):
    return nn.CrossEntropyLoss(
        weight=torch.FloatTensor(weight), reduction=reduction
    )
