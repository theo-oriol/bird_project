import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from .transforms import make_transforms
load_dotenv()


CONFIG_DIR = "configs"
FOLD_DIR =f"data/cross"
SOURCE_IMG = os.environ["SOURCE_IMG"]

path_out_json = os.path.join(CONFIG_DIR,"labelname.json")
with open(path_out_json, "r") as f:
    LABEL_COLS = json.load(f)  

def load_csv(path):
    """
    Reads a fold CSV and returns (image_names, labels as float32 normalized rows).
    Labels are kept as soft floats — binarization happens in the dataset if needed.
    """
    df = pd.read_csv(path)
    X  = df["name_of_img"].values
    y  = df[LABEL_COLS].values.astype(np.float32)
    # normalize rows to sum to 1 (same as original _normalize_rows)
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    y = y / row_sums
    return X, y


class BirdDataset(Dataset):
    """
    Single-image dataset. One row = one image + label vector.

    Args:
        paths      : array of image filenames (just the basename)
        labels     : (N, num_classes) float32 array, normalized rows
        img_dir    : folder where images live
        transform  : torchvision transform
        binarize   : if True, converts soft labels to binary (for BCE tasks)
    """

    def __init__(self, paths, labels, img_dir, cfg, transform=None, is_train=False, binarize=False):
        assert len(paths) == len(labels)
        self.paths     = paths
        self.labels    = labels
        self.img_dir   = img_dir
        self.cfg = cfg  
        self.transform_function = transform
        self.binarize  = binarize
        self.is_train = is_train

        RGBA_TRANSFORMS = {"maskonly", "gaussian_blur", "shuffle_mask_pixels", "greyscale_view"}
        self.additional = set(self.cfg.data.get("additional_transforms") or [])
        self.needs_rgba = bool(self.additional & RGBA_TRANSFORMS)

        if "greyscale_view" in self.additional:
            img_dir_Back_name = os.path.basename(img_dir)
            img_dir_Side_name = os.path.basename(img_dir).replace("Back", "Side")
            img_dir_Belly_name = os.path.basename(img_dir).replace("Back", "Belly")
            self.transform_back = self.transform_function(cfg=cfg,is_train=is_train, view="Back", meta_dir=os.path.join(CONFIG_DIR,"meta_data","grayscale",img_dir_Back_name))
            self.transform_side = self.transform_function(cfg=cfg,is_train=is_train, view="Side", meta_dir=os.path.join(CONFIG_DIR,"meta_data","grayscale",img_dir_Side_name))
            self.transform_belly = self.transform_function(cfg=cfg,is_train=is_train, view="Belly", meta_dir=os.path.join(CONFIG_DIR,"meta_data","grayscale",img_dir_Belly_name))
        else :
            self.transform = self.transform_function(cfg=cfg, is_train=is_train, view=None, meta_dir=None)
            
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.paths[idx])
        
        if "Belly" in self.paths[idx]:
            img_path = img_path.replace("Back", "Belly")
        elif "Side" in self.paths[idx]:
            img_path = img_path.replace("Back", "Side")

        img = Image.open(img_path).convert("RGBA" if self.needs_rgba else "RGB")

        if "greyscale_view" in self.additional:
            if "Back" in self.paths[idx]:
                img = self.transform_back(img)
            elif "Side" in self.paths[idx]:
                img = self.transform_side(img)
            elif "Belly" in self.paths[idx]:
                img = self.transform_belly(img)
            else : 
                raise ValueError(f"Unknown view in path '{self.paths[idx]}'")
        else :
            img = self.transform(img)

        label = self.labels[idx].copy()
        if self.binarize:
            label = (label > 0).astype(np.float32)

        return img, label, self.paths[idx]


def build_dataloaders(cfg, dataset_dir, img_folder):
    """
    Builds train and val dataloaders from config.
    Returns (train_loader, val_loader, train_labels) — train_labels
    is returned raw for weighted loss computation.
    """
    fold     = cfg.data.fold
    data_dir = os.path.join(FOLD_DIR, dataset_dir)        # path to the fold CSVs
    img_dir  = os.path.join(SOURCE_IMG ,img_folder)           # path to the images
    binarize = cfg.data.get("binarize", False)

    train_paths, train_labels = load_csv(
        os.path.join(data_dir, f"train_fold_{fold}.csv")
    )
    valid_paths, valid_labels = load_csv(
        os.path.join(data_dir, f"valid_fold_{fold}.csv")
    )
    # ########################################################################################
    # print("TEST VERSION !!!!!!!!!!!!!!")
    # train_paths = train_paths[:500]
    # train_labels = train_labels[:500]
    # valid_paths = valid_paths[:500]
    # valid_labels = valid_labels[:500]
    # #########################################################################################
    ds_train = BirdDataset(
        train_paths, train_labels, img_dir, cfg,
        transform=make_transforms,
        is_train = True,
        binarize=binarize,
    )
    ds_val = BirdDataset(
        valid_paths, valid_labels, img_dir, cfg,
        transform=make_transforms,
        is_train = False,
        binarize=binarize,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, train_labels