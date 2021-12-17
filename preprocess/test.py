#!/usr/bin/env python3

"""Testing `cityscapes-preprocess`
"""

import glob
import os

import numpy as np
from PIL import Image

from datasets.cityscapes_labels import (
    labels,
    trainId2label,
    id2label,
    name2label,
    label2trainId,
)
from preprocess.mask2edge import mask2edge, mask2edge_fast

# NOTE: same as trainId?
label_mapping = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}


def convert_label2trainId(label_img: np.ndarray) -> np.ndarray:
    """labelid2trainid function
    """

    if len(label_img.shape) == 2:
        h, w = label_img.shape
    elif len(label_img.shape) == 3:
        h, w, c = label_img.shape
        assert c == 1, f"ERR: input label has {c} channels which should be 1"
    else:
        raise ValueError()

    # 1. create an array populated with 255
    trainId_img = 255 * np.ones((h, w), dtype=np.uint8)  # 8-bit array

    # 2. map all pixels in the `label_mapping` dict
    for labelId, trainId in label_mapping.items():
        idx = label_img == labelId
        trainId_img[idx] == trainId

    return trainId_img


def main():
    """python version of `demo_preproc`
    """

    # FIXME: parameters should be converted to `argparse`

    # setup directories
    data_root = 'data_orig'
    output_root = 'data_proc'
    img_suffix = '_leftImg8bit.png'
    color_suffix = '_gtFine_color.png'
    labelIds_suffix = '_gtFine_labelIds.png'
    instIds_suffix = '_gtFine_instanceIds.png'
    trainIds_suffix = '_gtFine_trainIds.png'
    polygons_suffix = '_gtFine_polygons.json'
    edge_suffix = '_gtFine_edge.bin'  # this is the output

    cityscapes_root = 'data/cityscapes'
    gtFine_dir = 'gtFine'
    img_dir = 'leftImg8bit'
    out_dir = 'gtProc'

    # setup parameters
    radius = 2
    num_categories = len(label_mapping)  # 19

    # 0. setup parallel workers
    # FIXME: threading or multiprocessing (make sure to limit threads for numpy)

    # 1. generate output directories

    # 2. generate preprocessed dataset
    # splits = ['train', 'val', 'test']  # FIXME: test split doesn't have GT (dummy anno)
    splits = ['train', 'val']
    for split in splits:

        img_split_path = os.path.join(cityscapes_root, img_dir, split)
        gtFine_split_path = os.path.join(cityscapes_root, gtFine_dir, split)
        # minor checks
        assert os.path.exists(img_split_path)
        assert os.path.exists(gtFine_split_path)

        cities = os.listdir(img_split_path)
        _cities = os.listdir(gtFine_split_path)
        # minor checks
        assert len(cities) == len(_cities)
        assert len(set(cities) - set(_cities)) == 0

        for city in cities:
            img_city_path = os.path.join(img_split_path, city)
            gtFine_city_path = os.path.join(gtFine_split_path, city)
            # minor checks
            assert os.path.exists(img_city_path)
            assert os.path.exists(gtFine_city_path)

            save_root = os.path.join(cityscapes_root, out_dir, split, city)
            # TODO: make dirs if it doesn't exist

            img_paths = glob.glob(os.path.join(img_city_path, "*.png"))
            assert len(img_paths) > 0

            for _img_path in img_paths:
                # strip the prefix (to save as split txt file)
                img_path = os.path.relpath(_img_path, cityscapes_root)
                data_name = os.path.basename(img_path)[:-len(img_suffix)]

                # 3. generate and write data
                # 3.1. copy image and gt files to output directory
                # FIXME: instead of duplicating the GTs, separate the directories

                # 3.2. [if not 'test']
                # 3.2.1. transform label id map to train id map and save (segmentation map)
                labelId_path = os.path.join(gtFine_city_path, f"{data_name}{labelIds_suffix}")
                assert os.path.exists(labelId_path)
                labelId_map = np.array(Image.open(labelId_path))

                # save trainIds
                trainId_map = convert_label2trainId(labelId_map)
                trainId_img = Image.fromarray(trainId_map, 'L')
                trainId_img.save(os.path.join(save_root, f"{data_name}{trainIds_suffix}"))

                # 3.2.2. transform color map to edge map and write
                edge_map = mask2edge(labelId_map, radius=radius, ignore_labels=[2, 3], edge_type="regular")
                h, w = labelId_map.shape
                cat_edge_map = np.zeros((h, w), dtype=np.uint32)
                cat_edge_map = cat_edge_map.flatten()  # FIXME: is this necessary?
                for cat_idx in range(0, num_categories):
                    mask_map = trainId_map == cat_idx
                    if (mask_map is True).any():  # FIXME: does this work?
                        edge_idx = mask2edge_fast(
                            cat_mask=mask_map,
                            candidate_edge=edge_map,
                            radius=radius,
                            edge_type="regular",
                        )
                        edge_idx = edge_idx.flatten()
                        # bit manipulation
                        cat_edge_map[edge_idx] = cat_edge_map[edge_idx] + 2**(cat_idx)

                cat_edge_map = cat_edge_map.reshape(h, w)

                # 4. save list of images for the split as a txt file
                # hdf5?
                # how to write binary file?


if __name__ == "__main__":

    # print(trainId2label)
    # print(label2trainId)
    # Need to remove 255 and -1
    # this should be the same as `label_mapping` used in cityscapes-preprocessing
    new_label2trainId = {l: t for l, t in label2trainId.items() if t != 255 and t >= 0}

    # print(new_label2trainId)
    # print(new_label2trainId == label_mapping)

    main()
