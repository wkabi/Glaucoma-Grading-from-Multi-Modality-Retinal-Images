import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import Optional

from optic.utils.torch_helper import worker_init_fn


def data_pipeline(
    dataset: Optional[Dataset] = None,
    batch_size: int = 4,
    number_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = torch.cuda.is_available()
):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=number_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        # drop_last=True
    )

    return data_loader
