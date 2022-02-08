##############################################################################
# Config
# Config is used to set dataset path for training and testing
##############################################################################

import os
import os.path as osp

import torch

from sseg.utils.AttrDict import AttrDict


__C = AttrDict()
# Consumers can get config by:
# from fast_rcnn_config import cfg
cfg = __C
__C.EPOCH = 0
# Use Class Uniform Sampling to give each class proper sampling
__C.CLASS_UNIFORM_PCT = 0.0
# Use class weighted loss per batch to increase loss for low pixel count classes per batch
__C.BATCH_WEIGHTING = False
# Border Relaxation Count
__C.BORDER_WINDOW = 1
# Number of epoch to use before turn off border restriction
__C.REDUCE_BORDER_EPOCH = -1
# Comma Seperated List of class id to relax
__C.STRICTBORDERCLASS = None

# Attribute Dictionary for Dataset
__C.DATASET = AttrDict()
# Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = osp.join(os.getcwd(), "data/cityscapes")
# Number of splits to support
__C.DATASET.CV_SPLITS = 3

__C.MODEL = AttrDict()
__C.MODEL.BN = "regularnorm"
__C.MODEL.BNFUNC = torch.nn.BatchNorm2d
__C.MODEL.BIGMEMORY = False


def assert_and_infer_cfg(args, make_immutable=True, train_mode=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """

    if args.syncbn:
        import encoding

        __C.MODEL.BN = "syncnorm"
        # __C.MODEL.BNFUNC = torch.nn.SyncBatchNorm  # FIXME: requires DDP
        __C.MODEL.BNFUNC = encoding.nn.SyncBatchNorm
    else:
        __C.MODEL.BNFUNC = torch.nn.BatchNorm2d
        print("Using regular batch norm")

    if not train_mode:
        cfg.immutable(True)
        return

    if args.batch_weighting:
        __C.BATCH_WEIGHTING = True

    if make_immutable:
        cfg.immutable(True)
