#!/usr/bin/env python3

import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

# from sseg.config import cfg
# from sseg.network import SEresnext
from sseg.network import Resnet
from sseg.network.mynn import initialize_weights, Norm2d
from sseg.my_functionals import GatedSpatialConv as gsc


class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices).long()
            x = x.index_select(axis, Variable(indices))
        return x


class MyIdentity(nn.Module):
    def __init__(self, axis, offset):
        super(MyIdentity, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        return x


class SideOutputCrop(nn.Module):
    """
    This is the original implementation ConvTranspose2d (fixed) and crops
    """

    def __init__(
        self, num_output, kernel_sz=None, stride=None, upconv_pad=0, do_crops=True
    ):
        super(SideOutputCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(
            num_output, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True
        )

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(
                1,
                out_channels=1,
                kernel_size=kernel_sz,
                stride=stride,
                padding=upconv_pad,
                bias=False,
            )
            # doing crops
            if self._do_crops:
                self.crops = Crop(2, offset=kernel_sz // 4)
            else:
                self.crops = MyIdentity(None, None)
        else:
            self.upsample = False

    def forward(self, res, reference=None):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)
            side_output = self.crops(side_output, reference)

        return side_output


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise "output stride of {} not supported".format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True),
            )
        )
        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_dim,
                        reduction_dim,
                        kernel_size=3,
                        dilation=r,
                        padding=r,
                        bias=False,
                    ),
                    Norm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(
            img_features, x_size[2:], mode="bilinear", align_corners=True
        )
        out = img_features

        edge_features = F.interpolate(
            edge, x_size[2:], mode="bilinear", align_corners=True
        )
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class GSCNN(nn.Module):
    """
    ResNet101 version of GSCNN

    structure: [3, 4, 23, 3]
    channels = [256, 512, 1024, 2048]
    """

    def __init__(self, num_classes, trunk=None, criterion=None, pretrained=True):

        super(GSCNN, self).__init__()
        self.criterion = criterion
        self.num_classes = num_classes

        resnet = Resnet.resnet101(pretrained=pretrained)
        self.r_conv1 = resnet.conv1
        self.r_bn1 = resnet.bn1
        self.r_relu = resnet.relu
        self.r_pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        del resnet

        # NOTE: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        # I see most projects use `align_corners=True`
        self.interpolate = F.interpolate

        self.dsn0 = nn.Conv2d(64, 1, 1)  # NOTE: not used
        self.dsn1 = nn.Conv2d(256, 1, 1)
        self.dsn2 = nn.Conv2d(512, 1, 1)
        self.dsn4 = nn.Conv2d(2048, 1, 1)

        self.res1 = Resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=8)

        # Bottlenecks to reduce feature dim?
        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280 + 256, 256, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.final_seg)

    def forward(self, inp, gts=None):

        x_size = inp.size()

        # first conv
        m0 = self.r_conv1(inp)
        m0 = self.r_bn1(m0)
        m0 = self.r_relu(m0)

        # layer 1
        m1 = self.layer1(self.r_pool(m0))
        # layer 2
        m2 = self.layer2(m1)
        # layer 3
        m3 = self.layer3(m2)
        # layer 4
        m4 = self.layer4(m3)

        s1 = F.interpolate(
            self.dsn1(m1), x_size[2:], mode="bilinear", align_corners=True
        )
        s2 = F.interpolate(
            self.dsn2(m2), x_size[2:], mode="bilinear", align_corners=True
        )
        s4 = F.interpolate(
            self.dsn4(m4), x_size[2:], mode="bilinear", align_corners=True
        )

        m1f = F.interpolate(m0, x_size[2:], mode="bilinear", align_corners=True)

        im_arr = inp.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            # https://github.com/nv-tlabs/GSCNN/issues/22
            # does canny work on non-uint8 images?
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        cs = self.res1(m1f)
        cs = F.interpolate(cs, x_size[2:], mode="bilinear", align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, s1)
        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size[2:], mode="bilinear", align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s2)
        cs = self.res3(cs)
        cs = F.interpolate(cs, x_size[2:], mode="bilinear", align_corners=True)
        cs = self.d3(cs)
        cs = self.gate3(cs, s4)
        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:], mode="bilinear", align_corners=True)
        # FIXME: https://github.com/nv-tlabs/GSCNN/issues/62
        # loss function already has sigmoid
        edge_out = self.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        # aspp
        x = self.aspp(m4, acts)
        # FIXME: some values of x is NaN
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(m1)
        dec0_up = self.interpolate(
            dec0_up, m1.size()[2:], mode="bilinear", align_corners=True
        )

        # FIXME: dec0_up is ALL NaN
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final_seg(dec0)
        # FIXME: dec1 is ALL NaN
        seg_out = self.interpolate(dec1, x_size[2:], mode="bilinear")

        if self.training:
            return self.criterion((seg_out, edge_out), gts)
        else:
            return seg_out, edge_out
