#!/usr/bin/env python3

import torch

import torchvision

# FIXME: only torch>1.10
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from pytorch_memlab import MemReporter, profile, set_target_gpu

from network.wider_resnet import wider_resnet38_a2


@profile
def profile_resnet(gpu_id: int = 0):
    set_target_gpu(gpu_id)
    resnet = torchvision.models.resnet50().cuda(gpu_id)


@profile
def profile_wider_resnet(gpu_id: int = 0):
    set_target_gpu(gpu_id)
    wider_resnet = wider_resnet38_a2().cuda(gpu_id)


def detailed_resnet(gpu_id: int = 0):
    loss = torch.nn.CrossEntropyLoss()

    resnet = torchvision.models.resnet50().cuda(gpu_id)
    reporter = MemReporter(resnet)
    reporter.report(verbose=True)
    inp = torch.randn(4, 3, 224, 224).cuda(gpu_id)
    out = resnet(inp)
    gt = torch.randn(4, 1000).cuda(gpu_id)
    out = loss(out, gt)
    out.backward()
    reporter.report(verbose=True)


def get_output(gpu_id: int = 0):

    resnet = torchvision.models.resnet50()

    # NOTE: new way of making an extractor from backbones
    resnet2 = create_feature_extractor(resnet, return_nodes={"layer4": "feat"})

    wider = wider_resnet38_a2()

    inp = torch.randn(4, 3, 224, 224)
    out_resnet = resnet2(inp)["feat"]

    out_wider = wider(inp)

    print(out_resnet.shape)
    print(out_wider.shape)


if __name__ == "__main__":

    # profile_resnet()
    # profile_wider_resnet()
    # detailed_resnet(0)
    get_output()
