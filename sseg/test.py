#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division

import argparse
from functools import partial
import logging
import math
import os
import sys

import torch
import numpy as np
from PIL import Image

from sseg.config import cfg, assert_and_infer_cfg
import datasets
import sseg.loss as loss
import network
import sseg.optimizer as optimizer

from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
from utils.f_boundary import eval_mask_boundary

torch.autograd.set_detect_anomaly(True)


# Argument Parser
parser = argparse.ArgumentParser(description="GSCNN")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--arch", type=str, default="network.gscnn.GSCNN")
parser.add_argument("--dataset", type=str, default="cityscapes")
parser.add_argument("--cv", type=int, default=0, help="cross validation split")
parser.add_argument(
    "--joint_edgeseg_loss", action="store_true", default=True, help="joint loss"
)
parser.add_argument(
    "--img_wt_loss",
    action="store_true",
    default=False,
    help="per-image class-weighted loss",
)
parser.add_argument(
    "--batch_weighting",
    action="store_true",
    default=False,
    help="Batch weighting for class",
)
parser.add_argument(
    "--eval_thresholds",
    type=str,
    default="0.0005,0.001875,0.00375,0.005",
    help="Thresholds for boundary evaluation",
)
parser.add_argument("--rescale", type=float, default=1.0, help="Rescaled LR Rate")
parser.add_argument("--repoly", type=float, default=1.5, help="Rescaled Poly")

parser.add_argument(
    "--edge_weight", type=float, default=1.0, help="Edge loss weight for joint loss"
)
parser.add_argument(
    "--seg_weight",
    type=float,
    default=1.0,
    help="Segmentation loss weight for joint loss",
)
parser.add_argument(
    "--att_weight", type=float, default=1.0, help="Attention loss weight for joint loss"
)
parser.add_argument(
    "--dual_weight", type=float, default=1.0, help="Dual loss weight for joint loss"
)

parser.add_argument("--evaluate", action="store_true", default=False)

parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument("--sgd", action="store_true", default=True)
parser.add_argument("--sgd_finetuned", action="store_true", default=False)
parser.add_argument("--adam", action="store_true", default=False)
parser.add_argument("--amsgrad", action="store_true", default=False)

parser.add_argument(
    "--trunk",
    type=str,
    default="resnet101",
    help="trunk model, can be: resnet101 (default), resnet50",
)
parser.add_argument("--max_epoch", type=int, default=250)  # 175
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument(
    "--color_aug", type=float, default=0.25, help="level of color augmentation"
)
parser.add_argument("--rotate", type=float, default=0, help="rotation")
parser.add_argument("--gblur", action="store_true", default=True)
parser.add_argument("--bblur", action="store_true", default=False)
parser.add_argument(
    "--lr_schedule", type=str, default="poly", help="name of lr schedule: poly"
)
parser.add_argument(
    "--poly_exp", type=float, default=1.0, help="polynomial LR exponent"
)
parser.add_argument("--bs_mult", type=int, default=1)
parser.add_argument("--bs_mult_val", type=int, default=2)
parser.add_argument("--crop_size", type=int, default=720, help="training crop size")
parser.add_argument(
    "--pre_size",
    type=int,
    default=None,
    help="resize image shorter edge to this before augmentation",
)
parser.add_argument(
    "--scale_min",
    type=float,
    default=0.5,
    help="dynamically scale training images down to this size",
)
parser.add_argument(
    "--scale_max",
    type=float,
    default=2.0,
    help="dynamically scale training images up to this size",
)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--snapshot", type=str, default=None)
parser.add_argument("--restore_optimizer", action="store_true", default=False)
parser.add_argument(
    "--exp", type=str, default="default", help="experiment directory name"
)
parser.add_argument("--tb_tag", type=str, default="", help="add tag to tb dir")
parser.add_argument("--ckpt", type=str, default="logs/ckpt")
parser.add_argument("--tb_path", type=str, default="logs/tb")
parser.add_argument(
    "--syncbn", action="store_true", default=True, help="Synchronized BN"
)
parser.add_argument(
    "--dump_augmentation_images",
    action="store_true",
    default=False,
    help="",
)
parser.add_argument(
    "--test_mode",
    action="store_true",
    default=False,
    help="minimum testing (1 epoch run ) to verify nothing failed",
)
parser.add_argument("-wb", "--wt_bound", type=float, default=1.0)
parser.add_argument("--maxSkip", type=int, default=0)
args = parser.parse_args()
args.best_record = {
    "epoch": -1,
    "iter": 0,
    "val_loss": 1e10,
    "acc": 0,
    "acc_cls": 0,
    "mean_iu": 0,
    "fwavacc": 0,
}

# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1
# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
    print("Total world size: ", int(os.environ["WORLD_SIZE"]))


def main():
    """
    Main Function

    """

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    criterion, criterion_val = loss.get_loss(args)
    net = network.get_net(args, criterion)
    optim, scheduler = optimizer.get_optimizer(args, net)

    torch.cuda.empty_cache()

    # Early evaluation for benchmarking
    default_eval_epoch = 1
    validate(val_loader, net, criterion_val, optim, default_eval_epoch, writer)
    evaluate(val_loader, net)


def validate(val_loader, net, criterion, optimizer, curr_epoch, writer):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.eval()
    val_loss = AverageMeter()
    mf_score = AverageMeter()
    IOU_acc = 0
    dump_images = []
    heatmap_images = []
    for vi, data in enumerate(val_loader):
        input, mask, edge, img_names = data
        assert len(input.size()) == 4 and len(mask.size()) == 3
        assert input.size()[2:] == mask.size()[1:]
        h, w = mask.size()[1:]

        batch_pixel_size = input.size(0) * input.size(2) * input.size(3)
        input, mask_cuda, edge_cuda = input.cuda(), mask.cuda(), edge.cuda()

        with torch.no_grad():
            seg_out, edge_out = net(input)  # output = (1, 19, 713, 713)

        if args.joint_edgeseg_loss:
            loss_dict = criterion((seg_out, edge_out), (mask_cuda, edge_cuda))
            val_loss.update(sum(loss_dict.values()).item(), batch_pixel_size)
        else:
            val_loss.update(criterion(seg_out, mask_cuda).item(), batch_pixel_size)

        seg_predictions = seg_out.data.max(1)[1].cpu()
        edge_predictions = edge_out.max(1)[0].cpu()

        _edge = edge.max(1)[0]

        # NOTE: save images on the fly
        save_root = os.path.join(args.exp_path, "all_images")
        os.makedirs(save_root, exist_ok=True)
        # segmentation
        for idx, data in enumerate(zip(mask, seg_predictions, img_names)):
            gt_pil = args.dataset_cls.colorize_mask(data[0].cpu().numpy())
            pred = data[1].cpu().numpy()
            predictons_pil = args.dataset_cls.colorize_mask(pred)
            img_name = data[2]

            prediction_fn = "{}_pred_mask.png".format(img_name)
            gt_fn = "{}_gt_mask.png".format(img_name)
            scene_name = img_name.split("_")[0]
            save_dir = os.path.join(save_root, scene_name)
            os.makedirs(save_dir, exist_ok=True)

            predictons_pil.save(os.path.join(save_dir, prediction_fn))
            gt_pil.save(os.path.join(save_dir, gt_fn))
        # edge
        for idx, data in enumerate(zip(_edge, edge_predictions, img_names)):
            gt_pil = args.dataset_cls.colorize_mask(data[0].cpu().numpy())
            pred = data[1].cpu().numpy()
            pred = (pred / pred.max()) * 255
            pred_pil = Image.fromarray(pred.astype(np.uint8))
            img_name = data[2]

            prediction_fn = "{}_pred_edge.png".format(img_name)
            gt_fn = "{}_gt_edge.png".format(img_name)
            scene_name = img_name.split("_")[0]
            save_dir = os.path.join(save_root, scene_name)
            os.makedirs(save_dir, exist_ok=True)

            pred_pil.save(os.path.join(save_dir, prediction_fn))
            gt_pil.save(os.path.join(save_dir, gt_fn))

        # Logging
        if vi % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d" % (vi + 1, len(val_loader)))
        if vi > 10 and args.test_mode:
            break

        # Image Dumps
        if vi < 10:
            dump_images.append([mask, seg_predictions, img_names])
            heatmap_images.append([_edge, edge_predictions, img_names])

        IOU_acc += fast_hist(
            seg_predictions.numpy().flatten(),
            mask.numpy().flatten(),
            args.dataset_cls.num_classes,
        )

        del seg_out, edge_out, vi, data

    if args.local_rank == 0:
        evaluate_eval(
            args,
            net,
            optimizer,
            val_loss,
            mf_score,
            IOU_acc,
            dump_images,
            heatmap_images,
            writer,
            curr_epoch,
            args.dataset_cls,
        )

    return val_loss.avg


def evaluate(val_loader, net):
    """
    Runs the evaluation loop and prints F score
    val_loader: Data loader for validation
    net: thet network
    return:
    """
    net.eval()
    for thresh in args.eval_thresholds.split(","):
        mf_score1 = AverageMeter()
        mf_pc_score1 = AverageMeter()
        ap_score1 = AverageMeter()
        ap_pc_score1 = AverageMeter()
        Fpc = np.zeros((args.dataset_cls.num_classes))
        Fc = np.zeros((args.dataset_cls.num_classes))
        for vi, data in enumerate(val_loader):
            input, mask, edge, img_names = data
            assert len(input.size()) == 4 and len(mask.size()) == 3
            assert input.size()[2:] == mask.size()[1:]
            h, w = mask.size()[1:]

            batch_pixel_size = input.size(0) * input.size(2) * input.size(3)
            input, mask_cuda, edge_cuda = input.cuda(), mask.cuda(), edge.cuda()

            with torch.no_grad():
                seg_out, edge_out = net(input)

            seg_predictions = seg_out.data.max(1)[1].cpu()
            edge_predictions = edge_out.max(1)[0].cpu()

            logging.info("evaluating: %d / %d" % (vi + 1, len(val_loader)))
            _Fpc, _Fc = eval_mask_boundary(
                seg_predictions.numpy(),
                mask.numpy(),
                args.dataset_cls.num_classes,
                bound_th=float(thresh),
            )
            Fc += _Fc
            Fpc += _Fpc

            del seg_out, edge_out, vi, data

        logging.info("Threshold: " + thresh)
        logging.info("F_Score: " + str(np.sum(Fpc / Fc) / args.dataset_cls.num_classes))
        logging.info("F_Score (Classwise): " + str(Fpc / Fc))


if __name__ == "__main__":
    main()
