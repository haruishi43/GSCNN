#!/usr/bin/env python3

import numpy as np

__all__ = [
    'convert_label2trainId',
    'label_mapping',
]

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
    """python version of `labelid2trainid` function
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
        trainId_img[idx] = trainId

    return trainId_img


if __name__ == "__main__":

    import os
    from PIL import Image
    from datasets.cityscapes_labels import (
        labels,
        trainId2label,
        id2label,
        name2label,
        label2trainId,
    )

    # print(trainId2label)
    # print(label2trainId)
    # Need to remove 255 and -1
    # this should be the same as `label_mapping` used in cityscapes-preprocessing
    new_label2trainId = {l: t for l, t in label2trainId.items() if t != 255 and t >= 0}

    # print(new_label2trainId)
    # print(new_label2trainId == label_mapping)

    label_path = os.path.join(
        './preprocess/data/',
        'cropped_aachen_000000_000019_labelIds.png',
    )
    label_img = Image.open(label_path)
    label = np.array(label_img)

    train = convert_label2trainId(label)

    for i in range(len(label_mapping)):
        print(i, np.any(train == i))
