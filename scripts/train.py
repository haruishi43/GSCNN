#!/usr/bin/env python3

import argparse
import logging
import os

import torch
import numpy as np

from sseg.config import cfg, assert_and_infer_cfg
from sseg import datasets, network, loss, optimizer
from sseg.utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
from sseg.utils.f_boundary import eval_mask_boundary
from sseg.utils.debug import timer

# torch.autograd.set_detect_anomaly(True)


# Argument Parser
parser = argparse.ArgumentParser(description="GSCNN")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--arch", type=str, default="sseg.network.gscnn.GSCNN")
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
    default="0.00088,0.001875,0.00375,0.005",
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
parser.add_argument(
    "--crop_size",
    type=int,
    default=800,  # orig: 720, paper: 800, DSN: 832
    help="training crop size",
)
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
    "--mf_nproc",
    type=int,
    default=16,
    help="validation mf num processes",
)
parser.add_argument(
    "--no_val_mf",
    action="store_true",
    default=False,
    help="validates mf",
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

    if args.evaluate:
        # Early evaluation for benchmarking
        default_eval_epoch = 1
        validate(val_loader, net, criterion_val, optim, default_eval_epoch, writer)
        evaluate(val_loader, net, args)
        return

    # Main Loop
    for epoch in range(args.start_epoch, args.max_epoch):
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.EPOCH = epoch
        cfg.immutable(True)

        train(train_loader, net, optim, epoch, writer)
        validate(val_loader, net, criterion_val, optim, epoch, writer)

        scheduler.step()


@timer(precision=6)
def train(train_loader, net, optimizer, curr_epoch, writer):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """
    net.train()

    train_main_loss = AverageMeter()
    train_edge_loss = AverageMeter()
    train_seg_loss = AverageMeter()
    train_att_loss = AverageMeter()
    train_dual_loss = AverageMeter()
    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        if i == 0:
            print("running....")

        inputs, mask, edge, _img_name = data

        if torch.sum(torch.isnan(inputs)) > 0:
            import pdb

            pdb.set_trace()

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs, mask, edge = inputs.cuda(), mask.cuda(), edge.cuda()

        if i == 0:
            print("forward done")

        optimizer.zero_grad()

        main_loss = None
        loss_dict = None

        if args.img_wt_loss:
            main_loss = net(inputs, gts=mask)

        elif args.joint_edgeseg_loss:
            loss_dict = net(inputs, gts=(mask, edge))

            if args.seg_weight > 0:
                log_seg_loss = loss_dict["seg_loss"].mean().clone().detach_()
                train_seg_loss.update(log_seg_loss.item(), batch_pixel_size)
                main_loss = loss_dict["seg_loss"]

            if args.edge_weight > 0:
                log_edge_loss = loss_dict["edge_loss"].mean().clone().detach_()
                train_edge_loss.update(log_edge_loss.item(), batch_pixel_size)
                if main_loss is not None:
                    main_loss += loss_dict["edge_loss"]
                else:
                    main_loss = loss_dict["edge_loss"]

            if args.att_weight > 0:
                log_att_loss = loss_dict["att_loss"].mean().clone().detach_()
                train_att_loss.update(log_att_loss.item(), batch_pixel_size)
                if main_loss is not None:
                    main_loss += loss_dict["att_loss"]
                else:
                    main_loss = loss_dict["att_loss"]

            if args.dual_weight > 0:
                log_dual_loss = loss_dict["dual_loss"].mean().clone().detach_()
                train_dual_loss.update(log_dual_loss.item(), batch_pixel_size)
                if main_loss is not None:
                    main_loss += loss_dict["dual_loss"]
                else:
                    main_loss = loss_dict["dual_loss"]

        else:
            main_loss = net(inputs, gts=mask)

        main_loss = main_loss.mean()
        log_main_loss = main_loss.clone().detach_()

        train_main_loss.update(log_main_loss.item(), batch_pixel_size)

        main_loss.backward()

        optimizer.step()

        if i == 0:
            print("step 1 done")

        curr_iter += 1

        if args.local_rank == 0:
            msg = "[epoch {}], [iter {} / {}], [train main loss {:0.6f}], [seg loss {:0.6f}], [edge loss {:0.6f}], [lr {:0.6f}]".format(
                curr_epoch,
                i + 1,
                len(train_loader),
                train_main_loss.avg,
                train_seg_loss.avg,
                train_edge_loss.avg,
                optimizer.param_groups[-1]["lr"],
            )

            logging.info(msg)

            # Log tensorboard metrics for each iteration of the training phase
            writer.add_scalar("training/loss", (train_main_loss.val), curr_iter)
            writer.add_scalar(
                "training/lr", optimizer.param_groups[-1]["lr"], curr_iter
            )
            if args.joint_edgeseg_loss:

                writer.add_scalar("training/seg_loss", (train_seg_loss.val), curr_iter)
                writer.add_scalar(
                    "training/edge_loss", (train_edge_loss.val), curr_iter
                )
                writer.add_scalar("training/att_loss", (train_att_loss.val), curr_iter)
                writer.add_scalar(
                    "training/dual_loss", (train_dual_loss.val), curr_iter
                )
        if i > 5 and args.test_mode:
            return


@timer(precision=6)
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
        input, mask_cuda = input.cuda(), mask.cuda()

        with torch.no_grad():
            seg_out, edge_out = net(input)  # output = (1, 19, 713, 713)

        val_loss.update(criterion(seg_out, mask_cuda).item(), batch_pixel_size)

        seg_predictions = seg_out.data.max(1)[1].cpu()
        edge_predictions = edge_out.max(1)[0].cpu()

        # Logging
        if vi % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d" % (vi + 1, len(val_loader)))
        if vi > 10 and args.test_mode:
            break
        _edge = edge.max(1)[0]

        # Image Dumps
        if vi < 10:
            dump_images.append([mask, seg_predictions, img_names])
            heatmap_images.append([_edge, edge_predictions, img_names])

        IOU_acc += fast_hist(
            seg_predictions.numpy().flatten(),
            mask.numpy().flatten(),
            args.dataset_cls.num_classes,
        )

        if not args.no_val_mf:
            # FIXME: slow
            Fpc, Fc = eval_mask_boundary(
                seg_predictions.numpy(),
                mask.numpy(),
                args.dataset_cls.num_classes,
                num_proc=args.mf_nproc,
                bound_th=0.00088,  # FIXME: hardcoded for now
            )
            mf_score.update(np.sum(Fpc / Fc) / args.dataset_cls.num_classes)

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


def evaluate(val_loader, net, args):
    """
    Runs the evaluation loop and prints F score
    val_loader: Data loader for validation
    net: thet network
    return:
    """
    net.eval()
    for i, thresh in enumerate(args.eval_thresholds.split(",")):
        Fpc = np.zeros((args.dataset_cls.num_classes))
        Fc = np.zeros((args.dataset_cls.num_classes))
        # val_loader.sampler.set_epoch(i + 1)
        evaluate_F_score(val_loader, net, thresh, Fpc, Fc)


@timer(precision=6)
def evaluate_F_score(val_loader, net, thresh, Fpc, Fc):
    for vi, data in enumerate(val_loader):
        input, mask, edge, img_names = data
        assert len(input.size()) == 4 and len(mask.size()) == 3
        assert input.size()[2:] == mask.size()[1:]
        input = input.cuda()

        with torch.no_grad():
            seg_out, _ = net(input)

        seg_predictions = seg_out.data.max(1)[1].cpu()

        print("evaluating: %d / %d" % (vi + 1, len(val_loader)))
        _Fpc, _Fc = eval_mask_boundary(
            seg_predictions.numpy(),
            mask.numpy(),
            args.dataset_cls.num_classes,
            bound_th=float(thresh),
        )
        Fc += _Fc
        Fpc += _Fpc

        del seg_out, vi, data

    # if args.apex:
    #     Fc_tensor = torch.cuda.FloatTensor(Fc)
    #     torch.distributed.all_reduce(Fc_tensor, op=torch.distributed.ReduceOp.SUM)
    #     Fc = Fc_tensor.cpu().numpy()
    #     Fpc_tensor = torch.cuda.FloatTensor(Fpc)
    #     torch.distributed.all_reduce(Fpc_tensor, op=torch.distributed.ReduceOp.SUM)
    #     Fpc = Fpc_tensor.cpu().numpy()

    if args.local_rank == 0:
        logging.info("Threshold: " + thresh)
        logging.info("F_Score: " + str(np.sum(Fpc / Fc) / args.dataset_cls.num_classes))
        logging.info("F_Score (Classwise): " + str(Fpc / Fc))

    return Fpc


# def old_evaluate(val_loader, net):
#     """
#     Runs the evaluation loop and prints F score
#     val_loader: Data loader for validation
#     net: thet network
#     return:
#     """
#     net.eval()
#     for thresh in args.eval_thresholds.split(","):
#         mf_score1 = AverageMeter()
#         mf_pc_score1 = AverageMeter()
#         ap_score1 = AverageMeter()
#         ap_pc_score1 = AverageMeter()
#         Fpc = np.zeros((args.dataset_cls.num_classes))
#         Fc = np.zeros((args.dataset_cls.num_classes))
#         for vi, data in enumerate(val_loader):
#             input, mask, edge, img_names = data
#             assert len(input.size()) == 4 and len(mask.size()) == 3
#             assert input.size()[2:] == mask.size()[1:]
#             h, w = mask.size()[1:]

#             batch_pixel_size = input.size(0) * input.size(2) * input.size(3)
#             input, mask_cuda, edge_cuda = input.cuda(), mask.cuda(), edge.cuda()

#             with torch.no_grad():
#                 seg_out, edge_out = net(input)

#             seg_predictions = seg_out.data.max(1)[1].cpu()
#             edge_predictions = edge_out.max(1)[0].cpu()

#             logging.info("evaluating: %d / %d" % (vi + 1, len(val_loader)))
#             _Fpc, _Fc = eval_mask_boundary(
#                 seg_predictions.numpy(),
#                 mask.numpy(),
#                 args.dataset_cls.num_classes,
#                 bound_th=float(thresh),
#             )
#             Fc += _Fc
#             Fpc += _Fpc

#             del seg_out, edge_out, vi, data

#         logging.info("Threshold: " + thresh)
#         logging.info("F_Score: " + str(np.sum(Fpc / Fc) / args.dataset_cls.num_classes))
#         logging.info("F_Score (Classwise): " + str(Fpc / Fc))


if __name__ == "__main__":
    main()
