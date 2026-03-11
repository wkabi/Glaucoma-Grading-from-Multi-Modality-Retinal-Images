import os.path as osp
from typing import Callable, Optional

from optic.data import FundusDataset


class FundusDatasetFolders(FundusDataset):
    def __init__(self, data_root: str,
                 split: str,
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
                 oct_depth: int = 64) -> None:
        self.split = split
        super().__init__(
            data_root,
            mode=mode,
            transformer=transformer,
            oct_transformer=oct_transformer,
            use_h5file=use_h5file,
            return_oct=return_oct,
            return_mask=return_mask,
            return_disc_region=return_disc_region,
            return_disc_image=return_disc_image,
            return_fovea=return_fovea,
            oct_depth_resize=oct_depth_resize,
            oct_depth=oct_depth
        )

    def load_list(self):
        split_file = osp.join(self.data_root, "{}.csv".format(self.split))
        self.img_inds = []
        self.labels = []
        with open(split_file, "r") as f:
            for line in f:
                ind, label = line.strip().split(",")
                self.img_inds.append(int(ind))
                self.labels.append(int(label))

    def __repr__(self) -> str:
        return (
            "FundusDataset(data_root={}, mode={}, split={})\tSamples : {}".format(
                self.data_root, self.mode, self.split, self.__len__()
            )
        )