#!/usr/bin/env python3

import math
import os

from typing import List

import numpy as np


def mask2edge(
    mask,
    radius,
    ignore_labels: List[int] = [2, 3],
    edge_type: str = "regular",  # choice: 'regular', 'inner', 'outer'
):
    """python version of `seg2edge` subroutine

    This function takes an input segment and produces binary boundaries.
    Multi-channel input segments are supported by the function.
    """

    # 1. get dimensions
    # h, w, c = mask.shape
    assert len(mask.shape) == 2, f"ERR: only accepts 2-dim masks, but got {mask.shape}"
    h, w = mask.shape

    # 2. set the considered neighborhood
    search_radius = int(max(math.ceil(radius), 1))
    _x = np.linspace(0, w - 1, w, dtype=np.int64)
    _y = np.linspace(0, h - 1, h, dtype=np.int64)
    _rx = np.linspace(-search_radius, search_radius, search_radius * 2 + 1, dtype=np.int64)
    _ry = np.linspace(-search_radius, search_radius, search_radius * 2 + 1, dtype=np.int64)
    X, Y = np.meshgrid(_x, _y)
    rx, ry = np.meshgrid(_rx, _ry)

    # 3. columize everything (flatten)
    X = X.flatten()
    Y = Y.flatten()
    rx = rx.flatten()
    ry = ry.flatten()
    # mask = mask.flatten()

    # 4. build circular neighborhood
    neighbor_idxs = np.sqrt(rx**2 + ry**2) <= radius
    rx = rx[neighbor_idxs]
    ry = ry[neighbor_idxs]
    num_img_px = len(X)

    # 5. compute gaussian weight
    edge_idx = np.zeros(num_img_px, dtype=bool)
    for x, y in zip(rx, ry):
        X_neighbor = X + x
        Y_neighbor = Y + y
        valid_idx = np.where(
            (X_neighbor >= 0)
            & (X_neighbor < w)
            & (Y_neighbor >= 0)
            & (Y_neighbor < h)
        )[0]  # NOTE: it's a tuple...

        X_center = X[valid_idx]
        Y_center = Y[valid_idx]
        X_neighbor = X_neighbor[valid_idx]
        Y_neighbor = Y_neighbor[valid_idx]
        L_center = mask[Y_center, X_center]
        L_neighbor = mask[Y_neighbor, X_neighbor]

        if edge_type == "regular":
            diff_idx = np.where(L_center != L_neighbor)[0]
        elif edge_type == "inner":
            # TODO: understand what 'inner' does
            diff_idx = np.where(
                (L_center != L_neighbor)
                & (L_center != 0)  # FIXME: what? why 0?
                & (L_neighbor == 0)
            )[0]
        elif edge_type == "outer":
            # TODO: understand what 'outer' does
            diff_idx = np.where(
                (L_center != L_neighbor)
                & (L_center == 0)
                & (L_neighbor != 0)
            )[0]
        else:
            raise ValueError()

        L_center_edge = L_center[diff_idx]
        L_neighbor_edge = L_neighbor[diff_idx]
        use_idx = np.ones(diff_idx.shape, dtype=bool)

        assert L_center_edge.shape == L_neighbor_edge.shape == use_idx.shape

        for label in ignore_labels:
            ignore_idx = np.where((L_center_edge == label) | (L_neighbor_edge == label))
            use_idx[ignore_idx] = False

        assert use_idx.shape == L_center_edge.shape == L_neighbor_edge.shape

        diff_gt_idx = diff_idx[use_idx]
        edge_idx[valid_idx[diff_gt_idx]] = True

    return edge_idx.reshape(h, w)


def mask2edge_fast(
    mask,
    candidate_edge,
    radius,
    ignore_label,
    edge_type,
):
    """python version of `seg2edge_fast` subroutine

    Fast version of `seg2edge` by only considering pixels in `candidate_edge`.
    """

    # 1. get dimensions
    assert len(mask.shape) == 2, f"ERR: only accepts 2-dim masks, but got {mask.shape}"
    h, w = mask.shape

    # 2. set the considered neighborhood

    # 3. columize everything (flatten)

    # 4. build circular neighborhood

    # 5. compute gaussian weight


if __name__ == "__main__":

    from PIL import Image

    mask_path = os.path.join(
        './preprocess/data/',
        'cropped_aachen_000000_000019_labelIds.png',
    )
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img)
