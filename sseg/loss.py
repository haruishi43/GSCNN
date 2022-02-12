#!/usr/bin/env python3

"""loss.py"""

# FIXME: see
# https://github.com/NVIDIA/semantic-segmentation/blob/main/loss/utils.py
# for updates

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from sseg.config import cfg
from sseg.my_functionals.DualTaskLoss import DualTaskLoss


def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """

    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=args.dataset_cls.num_classes,
            ignore_index=args.dataset_cls.ignore_label,
            upper_bound=args.wt_bound,
        ).cuda()
    elif args.joint_edgeseg_loss:
        criterion = JointEdgeSegLoss(
            classes=args.dataset_cls.num_classes,
            ignore_index=args.dataset_cls.ignore_label,
            upper_bound=args.wt_bound,
            edge_weight=args.edge_weight,
            seg_weight=args.seg_weight,
            att_weight=args.att_weight,
            dual_weight=args.dual_weight,
        ).cuda()
    else:
        criterion = CrossEntropyLoss2d(
            size_average=True,
            ignore_index=args.dataset_cls.ignore_label,
        ).cuda()

    criterion_val = CrossEntropyLoss2d(
        size_average=True,
        ignore_index=args.dataset_cls.ignore_label,
    ).cuda()

    return criterion, criterion_val


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]  # get mask
        if isinstance(targets, (tuple, list)):
            targets = targets[0]  # get mask
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(
        self,
        classes,
        weight=None,
        ignore_index=255,
        norm=False,
        upper_bound=1.0,
    ):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(
            weight,
            reduction="mean",
            ignore_index=ignore_index,
        )
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        bins = torch.histc(
            target,
            bins=self.num_classes,
            min=0.0,
            max=self.num_classes,
        )
        hist_norm = bins.float() / bins.sum()
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound * (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound * (1.0 - hist_norm)) + 1.0
        return hist

    def forward(self, inputs, targets):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]  # get mask
        if isinstance(targets, (tuple, list)):
            targets = targets[0]  # get mask

        if self.batch_weights:
            weights = self.calculate_weights(targets)
            self.nll_loss.weight = weights

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(targets)
                self.nll_loss.weight = weights

            loss += self.nll_loss(
                F.log_softmax(
                    inputs[i].unsqueeze(0),
                    dim=1,
                ),
                targets[i].unsqueeze(0),
            )
        return loss


class JointEdgeSegLoss(nn.Module):
    def __init__(
        self,
        classes,
        ignore_index=255,
        upper_bound=1.0,
        edge_weight=1,
        seg_weight=1,
        att_weight=1,
        dual_weight=1,
    ):
        super(JointEdgeSegLoss, self).__init__()

        self.num_classes = classes
        self.seg_loss = ImageBasedCrossEntropyLoss2d(
            classes=classes,
            ignore_index=ignore_index,
            upper_bound=upper_bound,
        ).cuda()

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight

        self.dual_task = DualTaskLoss(cuda=True)  # FIXME: set to cuda for now

    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = target_t == 1
        neg_index = target_t == 0
        ignore_index = target_t > 1

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()

        # FIXME: already applies sigmoid
        # https://github.com/nv-tlabs/GSCNN/issues/62
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        # "this loss combines a Sigmoid layer and the BCELoss in one single class."
        # FIXME: why isn't `target_trans` used?
        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, size_average=True
        )
        return loss

    def edge_attention(self, input, target, edge):
        filler = torch.ones_like(target) * 255
        return self.seg_loss(
            input,
            torch.where(edge.max(1)[0] > 0.8, target, filler),
        )

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        losses["seg_loss"] = self.seg_weight * self.seg_loss(segin, segmask)
        losses["edge_loss"] = self.edge_weight * 20 * self.bce2d(edgein, edgemask)

        # dual task regularizer (dual loss is left, att loss is right)
        # FIXME: segin is ALL nan
        losses["dual_loss"] = self.dual_weight * self.dual_task(segin, segmask)
        losses["att_loss"] = self.att_weight * self.edge_attention(
            segin, segmask, edgein
        )

        return losses
