import os.path as osp
from tqdm import tqdm
import h5py
import argparse

from optic.data.fundus_dataset import FundusDataset
from optic.utils.file_io import mkdir

parser = argparse.ArgumentParser(description="convert the original dataset to new format")
parser.add_argument("--data-root", type=str,
                    default="./data/GAMMA_training_data/val_data")
args = parser.parse_args()

# data_root = "./data/GAMMA_training_data/val_data"

print(args.data_root)

dataset = FundusDataset(
    data_root=args.data_root,
    mode="all",
    return_oct=True,
    return_mask=False
)

mkdir(osp.join(args.data_root, "h5"))

for i, sample in tqdm(enumerate(dataset)):
    img = sample["img"]
    # label = sample["label"]
    oct_img = sample["oct_img"]
    # mask = sample["mask"]
    ind = sample["id"]
    out_path = osp.join(args.data_root, "h5", "{:04d}.h5".format(ind))
    f = h5py.File(out_path, "w")
    f.create_dataset("img", data=img, compression="gzip")
    # f.create_dataset("label", data=label, compression="gzip")
    f.create_dataset("oct_img", data=oct_img, compression="gzip")
    # f.create_dataset("mask", data=oct_img, compression="gzip")
    f.close()
