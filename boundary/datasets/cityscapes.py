#!/usr/bin/env python3

import os
import re

import numpy as np
from PIL import Image
import torch

from .base_cityscapes import BaseDataset


class CityscapesEdgeDetection(BaseDataset):
    NUM_CLASS = 19

    def __init__(
        self,
        root="data/cityscapes/",
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super(CityscapesEdgeDetection, self).__init__(
            root, split, mode, transform, target_transform, **kwargs
        )
        assert os.path.exists(root), (
            "Please download the dataset and place it under: %s" % root
        )

        self.images, self.masks = _get_cityscapes_pairs(root, split)
        if split != "vis":
            assert len(self.images) == len(self.masks)
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")

        if self.mode == "testval":
            img_size = torch.from_numpy(np.array(img.size))
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index]), img_size

        mask = Image.open(self.masks[index])

        if self.mode == "vis":
            img_size = torch.from_numpy(np.array(img.size))
            if self.transform is not None:
                img = self.transform(img)
            mask = self._mask_transform(mask)
            return img, mask, os.path.basename(self.images[index]), img_size

        # synchrosized transform
        if self.mode == "train":
            img, mask = self._sync_transform(img, mask)
        else:
            assert self.mode == "val"
            img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def _mask_transform(self, mask):
        mask = np.unpackbits(np.array(mask), axis=2)[:, :, -1:-20:-1]
        mask = torch.from_numpy(np.array(mask)).float()
        mask = mask.permute(2, 0, 1)  # channel first

        return mask

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_cityscapes_pairs(folder="data/cityscapes", split="train"):
    def get_path_pairs_json(folder, split_f):

        import json

        img_paths = []
        edge_paths = []

        with open(split_f, "r") as f:
            split = json.load(f)
            assert len(split) > 0

            for data in split:
                img_path = os.path.join(folder, data["img"])
                edge_path = os.path.join(folder, data["gtProc_edge"])

                assert os.path.exists(img_path)
                assert os.path.exists(edge_path)

                img_paths.append(img_path)
                edge_paths.append(edge_path)

        return img_paths, edge_paths

    def get_path_pairs(folder, split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, "r") as lines:
            for line in lines:
                ll_str = re.split(" ", line)
                imgpath = folder + ll_str[0].rstrip()
                maskpath = folder + ll_str[1].rstrip()
                if os.path.isfile(imgpath):
                    img_paths.append(imgpath)
                else:
                    print("cannot find the image:", imgpath)
                if os.path.isfile(maskpath):
                    mask_paths.append(maskpath)
                else:
                    print("cannot find the mask:", maskpath)
        return img_paths, mask_paths

    if split == "train":
        split_f = os.path.join(folder, "edge_splits", "train.json")
        img_paths, mask_paths = get_path_pairs_json(folder, split_f)
    elif split == "val":
        split_f = os.path.join(folder, "edge_splits", "val.json")
        img_paths, mask_paths = get_path_pairs_json(folder, split_f)
    elif split == "test":
        split_f = os.path.join(folder, "edge_splits", "test.json")
        img_paths, mask_paths = get_path_pairs_json(folder, split_f)
    else:
        # FIXME: how do we make vis.txt
        split_f = os.path.join(folder, "edge_splits", "vis.json")
        img_paths, mask_paths = get_path_pairs_json(folder, split_f)

    return img_paths, mask_paths
