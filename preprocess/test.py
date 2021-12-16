#!/usr/bin/env python3

"""Testing `cityscapes-preprocess`

TODO:
- converting MATLAB codes to Python
"""

import numpy as np

from datasets.cityscapes_labels import (
    labels,
    trainId2label,
    id2label,
    name2label,
    label2trainId,
)

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


def seg2edge(
    seg,
    radius,
    ignore_label,
    edge_type,
):
    """python version of `seg2edge` subroutine

    This function takes an input segment and produces binary boundaries.
    Multi-channel input segments are supported by the function.
    """
    ...


def seg2edge_fast(
    seg,
    candidate_edge,
    radius,
    ignore_label,
    edge_type,
):
    """python version of `seg2edge_fast` subroutine

    Fast version of `seg2edge` by only considering pixels in `candidate_edge`.
    """
    ...


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

    # setup parameters
    num_class = 19
    radius = 2

    # 0. setup parallel workers

    # 1. generate output directories

    # 2. generate preprocessed dataset
    splits = ['train', 'val', 'test']
    for split in splits:
        ...

        # for each city

        # for each image

        # 3. generate and write data
        # 3.1. copy image and gt files to output directory
        # 3.2. [if not 'test']
        # 3.2.1. transform label id map to train id map and save (segmentation map)
        # 3.2.2. transform color map to edge map and write


if __name__ == "__main__":

    # print(trainId2label)
    # print(label2trainId)
    # Need to remove 255 and -1
    # this should be the same as `label_mapping` used in cityscapes-preprocessing
    new_label2trainId = {l: t for l, t in label2trainId.items() if t != 255 and t >= 0}

    # print(new_label2trainId)
    # print(new_label2trainId == label_mapping)
