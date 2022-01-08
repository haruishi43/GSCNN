#!/usr/bin/env python3

import glob
import json
import os

import numpy as np
from PIL import Image

from preprocess.label2trainId import convert_label2trainId, label_mapping
from preprocess.mask2edge import mask2edge, mask2edge_fast


def preproc_original():
    """python version of `demo_preproc`

    NOTE: kept close to the original MATLAB code
    """

    # suffix
    img_suffix = '_leftImg8bit.png'
    color_suffix = '_gtFine_color.png'
    labelIds_suffix = '_gtFine_labelIds.png'
    instIds_suffix = '_gtFine_instanceIds.png'
    trainIds_suffix = '_gtFine_trainIds.png'
    polygons_suffix = '_gtFine_polygons.json'
    edge_bin_suffix = '_gtProc_edge.bin'
    edge_tif_suffix = '_gtProc_edge.tif'
    edge_png_suffix = '_gtProc_edge.png'

    cityscapes_root = 'data/cityscapes'
    img_dir = 'leftImg8bit'
    gtFine_dir = 'gtFine'
    out_dir = 'gtProc'
    split_dir = 'edge_splits'

    # setup parameters
    radius = 2
    num_categories = len(label_mapping)  # 19
    encode_type = "tif"  # ('rgb', 'bin', 'tif')

    if encode_type == "rgb":
        edge_suffix = edge_png_suffix
    elif encode_type == "bin":
        edge_suffix = edge_bin_suffix
    elif encode_type == "tif":
        edge_suffix = edge_tif_suffix
    else:
        raise ValueError()

    # 0. setup parallel workers
    # TODO: threading or multiprocessing (make sure to limit threads for numpy)
    # currently it's pretty slow

    # 1. generate output directories
    assert os.path.exists(cityscapes_root), f"ERR: {cityscapes_root} not found"
    assert os.path.exists(os.path.join(cityscapes_root, img_dir))
    assert os.path.exists(os.path.join(cityscapes_root, gtFine_dir))

    # 2. generate preprocessed dataset
    # splits = ['train', 'val', 'test']
    splits = ['train', 'val']
    for split in splits:

        split_files = []

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
            if not os.path.exists(save_root):
                print(f"Making directory: {save_root}")
                os.makedirs(save_root, exist_ok=True)

            img_paths = glob.glob(os.path.join(img_city_path, "*.png"))
            assert len(img_paths) > 0

            for _img_path in img_paths:
                # strip the prefix (to save as split txt file)
                img_path = os.path.relpath(_img_path, cityscapes_root)
                data_name = os.path.basename(img_path)[:-len(img_suffix)]

                split_info = {
                    'img': os.path.join(img_dir, split, city, f"{data_name}{img_suffix}"),
                    'gtFine_labelIds': os.path.join(gtFine_dir, split, city, f"{data_name}{labelIds_suffix}"),
                    'gtProc_trainIds': os.path.join(out_dir, split, city, f"{data_name}{trainIds_suffix}"),
                    'gtProc_edge': os.path.join(out_dir, split, city, f"{data_name}{edge_suffix}")
                }
                split_files.append(split_info)

                # 3. generate and write data
                # 3.1. copy image and gt files to output directory
                # NOTE: we skip this since copying files is redundant

                # 3.2. [if not 'test'] -> we skip `test` alltogether
                # 3.2.1. transform label id map to train id map and save (segmentation map)
                labelId_path = os.path.join(gtFine_city_path, f"{data_name}{labelIds_suffix}")
                assert os.path.exists(labelId_path)
                labelId_map = np.array(Image.open(labelId_path))

                # save trainIds
                trainId_map = convert_label2trainId(labelId_map)
                trainId_img = Image.fromarray(trainId_map, 'L')
                trainId_img.save(os.path.join(save_root, f"{data_name}{trainIds_suffix}"))

                # 3.2.2. transform color map to edge map and write
                edge_map = mask2edge(
                    labelId_map,
                    radius=radius,
                    ignore_labels=[2, 3],
                    edge_type="regular",
                )
                h, w = labelId_map.shape

                if encode_type == "rgb":
                    cat_edge_b = np.zeros((h, w), dtype=np.uint8)
                    cat_edge_g = np.zeros((h, w), dtype=np.uint8)
                    cat_edge_r = np.zeros((h, w), dtype=np.uint8)
                    cat_edge_png = np.zeros((h, w, 3), dtype=np.uint8)
                    cat_edge_b = cat_edge_b.flatten()
                    cat_edge_g = cat_edge_g.flatten()
                    cat_edge_r = cat_edge_r.flatten()

                    for cat_idx in range(0, num_categories):
                        mask_map = trainId_map == cat_idx
                        if np.any(mask_map):
                            edge_idx = mask2edge_fast(
                                cat_mask=mask_map,
                                candidate_edge=edge_map,
                                radius=radius,
                                edge_type="regular",
                            )
                            edge_idx = edge_idx.flatten()
                            # bit manipulation
                            # save as RGB image? up to 24 categories
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
                    # save as 3 channel uint8 (.png)
                    cat_edge_img = Image.fromarray(cat_edge_png)
                    cat_edge_img.save(os.path.join(save_root, f"{data_name}{edge_png_suffix}"))
                else:  # bin or tif
                    cat_edge_map = np.zeros((h, w), dtype=np.uint32)
                    cat_edge_map = cat_edge_map.flatten()  # FIXME: is this necessary?

                    for cat_idx in range(0, num_categories):
                        mask_map = trainId_map == cat_idx
                        if np.any(mask_map):
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

                    if encode_type == "bin":
                        # save directly to a binary file
                        # unfortunately, it strips shape information
                        cat_edge_map.tofile(os.path.join(save_root, f"{data_name}{edge_bin_suffix}"), dtype=np.uint32)
                    elif encode_type == "tif":
                        # save using PIL (uses I-mode and needs to be .tif)
                        # I-mode doesn't support uint32
                        cat_edge_map = cat_edge_map.view(np.int32)
                        cat_edge_img = Image.fromarray(cat_edge_map)
                        cat_edge_img.save(os.path.join(save_root, f"{data_name}{edge_tif_suffix}"))
                    else:
                        raise ValueError()

        # 4. save list of images for the split as a txt file
        split_path = os.path.join(cityscapes_root, split_dir, f"{split}.json")
        if not os.path.exists(os.path.join(cityscapes_root, split_dir)):
            os.makedirs(os.path.join(cityscapes_root, split_dir), exist_ok=True)
        with open(split_path, 'w') as f:
            json.dump(split_files, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OG preprocessing")
    parser.add_argument("--hoge")

    preproc_original()
