#!/usr/bin/env python3

import sys

import numpy as np

from preprocess.mask2edge import mask2edge, mask2edge_fast


def default_encoding(
    labelId_map: np.ndarray,
    trainId_map: np.ndarray,
    num_categories: int,
    radius: int,
):
    edge_map = mask2edge(
        labelId_map,
        radius=radius,
        ignore_labels=[2, 3],
        edge_type="regular",
    )

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

    return cat_edge_map


def rgb_encoding(
    labelId_map: np.ndarray,
    trainId_map: np.ndarray,
    num_categories: int,
    radius: int,
):
    edge_map = mask2edge(
        labelId_map,
        radius=radius,
        ignore_labels=[2, 3],
        edge_type="regular",
    )

    h, w = labelId_map.shape
    cat_edge_b = np.zeros((h, w), dtype=np.uint8)
    cat_edge_g = np.zeros((h, w), dtype=np.uint8)
    cat_edge_r = np.zeros((h, w), dtype=np.uint8)
    cat_edge_png = np.zeros((h, w, 3), dtype=np.uint8)
    cat_edge_b = cat_edge_b.flatten()
    cat_edge_g = cat_edge_g.flatten()
    cat_edge_r = cat_edge_r.flatten()
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
            if cat_idx >= 0 and cat_idx < 8:
                cat_edge_b[edge_idx] = cat_edge_b[edge_idx] + 2**(cat_idx)
            elif cat_idx >= 8 and cat_idx < 16:
                cat_edge_g[edge_idx] = cat_edge_g[edge_idx] + 2**(cat_idx - 8)
            elif cat_idx >= 16 and cat_idx < 24:
                cat_edge_r[edge_idx] = cat_edge_r[edge_idx] + 2**(cat_idx - 16)
            else:
                raise ValueError()

    cat_edge_b = cat_edge_b.reshape(h, w)
    cat_edge_g = cat_edge_g.reshape(h, w)
    cat_edge_r = cat_edge_r.reshape(h, w)
    cat_edge_png[:, :, 0] = cat_edge_r
    cat_edge_png[:, :, 1] = cat_edge_g
    cat_edge_png[:, :, 2] = cat_edge_b

    return cat_edge_png


def loading_edge_bin(bin_path, h, w, num_categories):
    b = np.fromfile(bin_path, dtype=np.uint32)
    if b.dtype.byteorder == '>' or \
       (b.dtype.byteorder == '=' and sys.byteorder == 'big'):
        b = b[:, ::-1]

    b = b.reshape(h, w)[:, :, None]  # reshape and make it 3 channels
    ub = np.unpackbits(
        b.view(np.uint8),
        axis=2,
        count=num_categories,
        bitorder='little',
    )
    return ub


def loading_edge_rgb():
    ...


def loading_edge_tif():
    ...
