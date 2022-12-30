# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import math
import numpy as np

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, kaiming_init, constant_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint, BaseModule
from mmseg.utils import get_root_logger

from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck, ResLayer

from ..utils.make_divisible import make_divisible, find_nearest_divisible_factor


class M_Module(BaseModule):
    def __init__(self,
                 inplanes,
                 planes,
                 paths=3,
                 groups=4,
                 expand_ratio=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="PReLU"),
                 order=('conv', 'norm', 'act'),
                 drop_prob=0.0,
                 with_cp=False,
                 init_cfg=None):
        super(M_Module, self).__init__(init_cfg)

        self.expand_ratio = expand_ratio
        self.paths = paths
        self.groups = groups
        self.with_cp = with_cp
        width = inplanes * expand_ratio // paths

        if self.expand_ratio > 1:
            self.expand_conv = ConvModule(
                inplanes,
                inplanes * (expand_ratio - 1),
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order,
                groups=inplanes)

        self.convs = nn.ModuleList()
        self.dcns = nn.ModuleList()
        
        for i in range(paths):
            if i == 0:
                self.convs.append(nn.Identity())
                self.dcns.append(nn.Identity())
            else:
                self.convs.append(
                    ConvModule(
                        width,
                        width,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        groups=groups,
                        order=order))
                self.dcns.append(
                    ConvModule(
                        width,
                        width,
                        3,
                        padding=1,
                        conv_cfg=dict(type="DCN"),
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        groups=groups,
                        order=order))

        self.compress_conv = \
            ConvModule(
                width * paths,
                planes,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order)

        self.drop_path = nn.Dropout2d(drop_prob) if drop_prob > 0.0 else nn.Identity()

        if inplanes != planes:
            self.res = \
                ConvModule(
                    inplanes,
                    planes,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
        else:
            self.res = nn.Identity()

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):

            if self.expand_ratio > 1:
                out = torch.cat([x, self.expand_conv(x)], 1)
            else:
                out = x

            spx = torch.chunk(out, self.paths, 1)
            sp = self.convs[0](spx[0])
            sp_outs = self.dcns[0](sp)
            for i in range(1, self.paths):
                sp = self.convs[i](spx[i] + sp)
                sp_out = self.dcns[i](sp)
                sp_outs = torch.cat([sp_outs, sp_out], 1)

            out = self.compress_conv(sp_outs)

            out = self.drop_path(out)

            out = out + self.res(x)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


# class ASPP(nn.ModuleList):
#     """Atrous Spatial Pyramid Pooling (ASPP) Module.
#
#     Args:
#         dilations (tuple[int]): Dilation rate of each layer.
#         in_channels (int): Input channels.
#         channels (int): Channels after modules, before conv_seg.
#         conv_cfg (dict|None): Config of conv layers.
#         norm_cfg (dict|None): Config of norm layers.
#         act_cfg (dict): Config of activation layers.
#     """
#
#     def __init__(self,
#                  inplanes,
#                  planes,
#                  dilations=(1, 6, 12, 18),
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN', requires_grad=True),
#                  act_cfg=dict(type="PReLU"),
#                  order=('conv', 'norm', 'act'),
#                  upsample_cfg=dict(mode='bilinear', align_corners=False)):
#         super(ASPP, self).__init__()
#         self.upsample_cfg = upsample_cfg
#         self.aspp_image_pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             ConvModule(
#                 inplanes,
#                 inplanes,
#                 1,
#                 groups=inplanes,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg))
#
#         self.aspp = nn.ModuleList()
#         for dilation in dilations:
#             self.aspp.append(
#                 ConvModule(
#                     inplanes,
#                     inplanes,
#                     1 if dilation == 1 else 3,
#                     dilation=dilation,
#                     padding=0 if dilation == 1 else dilation,
#                     groups=inplanes,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg,
#                     order=order))
#
#         self.aspp_bottleneck = ConvModule(
#             (len(dilations) + 1) * inplanes,
#             planes,
#             1,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg,
#             order=order)
#
#     def forward(self, x):
#         """Forward function."""
#         out = self.aspp_image_pool(x)
#
#         aspp_outs = [
#             nn.Upsample(
#                 size=x.size()[2:],
#                 mode=self.upsample_cfg['mode'],
#                 align_corners=self.upsample_cfg['align_corners'])(out)]
#         for aspp_module in self.aspp:
#             aspp_outs.append(aspp_module(x))
#         aspp_outs = torch.cat(aspp_outs, dim=1)
#
#         output = self.aspp_bottleneck(aspp_outs)
#
#         return output


class CT_Module(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 compress_ratio=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="Sigmoid"),
                 order=('conv', 'norm', 'act'),
                 init_cfg=None
                 ):
        super(CT_Module, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        self.compress_ratio = compress_ratio

        kernel_size = 1

        if compress_ratio == 1:
            if in_channels < out_channels:
                mid_channels = find_nearest_divisible_factor(in_channels, out_channels)
                self.expand_conv = \
                    ConvModule(
                        in_channels,
                        mid_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
                self.cheap_conv = \
                    ConvModule(
                        mid_channels,
                        out_channels - mid_channels,
                        1,
                        groups=mid_channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
            else:
                self.compress_conv = \
                    ConvModule(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
        else:
            mid_channels = out_channels // compress_ratio
            if in_channels < out_channels:
                self.expand_conv = \
                    ConvModule(
                        in_channels,
                        mid_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
            else:
                self.compress_conv = \
                    ConvModule(
                        in_channels,
                        mid_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order)
            self.cheap_conv = \
                ConvModule(
                    mid_channels,
                    out_channels - mid_channels,
                    1,
                    groups=mid_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

    def forward(self, x):
        """Forward function."""
        if self.compress_ratio == 1:
            if self.in_channels < self.out_channels:
                out = self.expand_conv(x)
                out = torch.cat([out, self.cheap_conv(out)], 1)
            else:
                out = self.compress_conv(x)
        else:
            if self.in_channels < self.out_channels:
                out = self.expand_conv(x)
                out = torch.cat([out, self.cheap_conv(out)], 1)
            else:
                out = self.compress_conv(x)
                out = torch.cat([out, self.cheap_conv(out)], 1)

        return out


class Down_Layer(BaseModule):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=2,
                 dilation=1,
                 down_type="stride",
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="PReLU"),
                 order=('conv', 'norm', 'act'),
                 drop_prob=0.0,
                 with_cp=False,
                 init_cfg=None):
        super(Down_Layer, self).__init__(init_cfg)

        assert down_type in [None, "stride", "focus"]

        self.stride = stride
        self.dilation = dilation
        self.down_type = down_type
        self.with_cp = with_cp

        if down_type == "focus" and dilation == 1:
            self.down_conv = \
                ConvModule(
                    inplanes * 4,
                    planes,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
        else:
            self.down_conv = \
                ConvModule(
                    inplanes,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        self.drop_path = nn.Dropout2d(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        def _inner_forward(x):
            if self.down_type == "focus" and self.dilation == 1:
                x = torch.cat([x[..., ::2, ::2],
                               x[..., 1::2, ::2],
                               x[..., ::2, 1::2],
                               x[..., 1::2, 1::2]], dim=1)
                out = self.down_conv(x)
            else:
                out = self.down_conv(x)

            out = self.drop_path(out)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class Stage(BaseModule):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=2,
                 dilation=1,
                 m_module_paths=3,
                 m_module_groups=3,
                 m_module_expand_ratio=4,
                 num_blocks=3,
                 down_type=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="PReLU"),
                 order=('conv', 'norm', 'act'),
                 drop_prob=0.0,
                 with_cp=False,
                 init_cfg=None):
        super(Stage, self).__init__(init_cfg)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.with_cp = with_cp
        
        if stride > 1 or inplanes != planes:
            self.down_layer = \
                Down_Layer(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dilation,
                    down_type=down_type,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order,
                    drop_prob=drop_prob,
                    with_cp=with_cp,
                    init_cfg=init_cfg)

        self.m_layer = nn.Sequential(*[
            M_Module(
                planes,
                planes,
                paths=m_module_paths,
                groups=m_module_groups,
                expand_ratio=m_module_expand_ratio,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order,
                drop_prob=drop_prob,
                with_cp=with_cp,
                init_cfg=init_cfg) for _ in range(num_blocks)])

    def forward(self, x):
        def _inner_forward(x):
            if self.stride > 1 or self.inplanes != self.planes:
                x = self.down_layer(x)

            out = self.m_layer(x)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class BT_Block(BaseModule):
    def __init__(self,
                 slave_out_channels,
                 master_out_channels,
                 ct_module_compress_ratios=(4, 4),  # s2m, m2s
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Sigmoid'),
                 bt_mode='dual_mul',
                 order=("conv", "norm", "act"),
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 slave_wh=2,
                 master_wh=2,
                 drop_prob=-1,
                 with_cp=False,
                 init_cfg=None):
        super(BT_Block, self).__init__(init_cfg)
        assert bt_mode in ["dual_mul", "dual_add",
                           "s2m_mul", "s2m_add",
                           "m2s_mul", "m2s_add",
                           None,
                           "s2m_mul_e", "s2m_add_e",
                           "m2s_mul_e", "m2s_add_e",
                           "dual_mul_e", "dual_add_e"]

        self.bt_mode = bt_mode
        self.with_cp = with_cp
        self.upsample_cfg = upsample_cfg

        self.s_scale = int(math.log(slave_wh, 2))
        self.m_scale = int(math.log(master_wh, 2))

        if self.s_scale > self.m_scale:
            self.scale_convs = nn.Sequential(*[
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order) for _ in range(self.s_scale - self.m_scale)])
        else:
            self.scale_convs = nn.Identity()

        if bt_mode == "dual_mul" or bt_mode == "dual_add":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)

        # elif bt_mode == "dual_add_mul":
        #     self.s2m_conv = \
        #         CT_Module(
        #             slave_out_channels,
        #             master_out_channels,
        #             compress_ratio=ct_module_compress_ratios[0],
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #             order=order)
        #     self.m2s_conv = \
        #         CT_Module(
        #             master_out_channels,
        #             slave_out_channels,
        #             compress_ratio=ct_module_compress_ratios[1],
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=dict(type='Sigmoid'),
        #             order=order)

        elif bt_mode == "s2m_mul" or bt_mode == "s2m_add":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)

        elif bt_mode == "m2s_mul" or bt_mode == "m2s_add":
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)

        elif bt_mode == "dual_mul_e" or bt_mode == "dual_add_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)
            self.unify_conv = \
                DepthwiseSeparableConvModule(
                    slave_out_channels,
                    master_out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.gather_convs = \
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        # elif bt_mode == "dual_add_mul_e":
        #     self.s2m_conv = \
        #         CT_Module(
        #             slave_out_channels,
        #             master_out_channels,
        #             compress_ratio=ct_module_compress_ratios[0],
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #             order=order)
        #     self.m2s_conv = \
        #         CT_Module(
        #             master_out_channels,
        #             slave_out_channels,
        #             compress_ratio=ct_module_compress_ratios[1],
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=dict(type='Sigmoid'),
        #             order=order)
        #     self.unify_conv = \
        #         DepthwiseSeparableConvModule(
        #             slave_out_channels,
        #             master_out_channels,
        #             1,
        #             padding=0,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #             order=order)
        #     self.gather_convs = \
        #         DepthwiseSeparableConvModule(
        #             master_out_channels,
        #             master_out_channels,
        #             3,
        #             padding=1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #             order=order)

        elif bt_mode == "s2m_mul_e" or bt_mode == "s2m_add_e":
            self.s2m_conv = \
                CT_Module(
                    slave_out_channels,
                    master_out_channels,
                    compress_ratio=ct_module_compress_ratios[0],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)
            self.unify_conv = \
                DepthwiseSeparableConvModule(
                    slave_out_channels,
                    master_out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.gather_convs = \
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        elif bt_mode == "m2s_mul_e" or bt_mode == "m2s_add_e":
            self.m2s_conv = \
                CT_Module(
                    master_out_channels,
                    slave_out_channels,
                    compress_ratio=ct_module_compress_ratios[1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid') if "mul" in bt_mode else act_cfg,
                    order=order)
            self.unify_conv = \
                DepthwiseSeparableConvModule(
                    slave_out_channels,
                    master_out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            self.gather_convs = \
                DepthwiseSeparableConvModule(
                    master_out_channels,
                    master_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)

        else:
            self.s2m_conv = nn.Identity()
            self.m2s_conv = nn.Identity()

        self.drop_path = nn.Dropout2d(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            x_slave, x_master = x[0], x[1]

            if self.bt_mode == "dual_mul" or self.bt_mode == "dual_add":
                out_master = self.s2m_conv(x_slave)
                out_master = nn.Upsample(size=x_master.size()[2:],
                                         mode=self.upsample_cfg["mode"],
                                         align_corners=self.upsample_cfg["align_corners"])(out_master)
                out_master = self.drop_path(out_master)

                if "mul" in self.bt_mode:
                    out_master = x_master * out_master
                else:
                    out_master = x_master + out_master

                out_slave = self.m2s_conv(self.scale_convs(x_master))
                out_slave = self.drop_path(out_slave)
                if "mul" in self.bt_mode:
                    out_slave = x_slave * out_slave
                else:
                    out_slave = x_slave + out_slave

                results = [out_slave, out_master]

            # elif self.bt_mode == "dual_add_mul":
            #     out_master = self.s2m_conv(x_slave)
            #     out_master = nn.Upsample(size=x_master.size()[2:],
            #                              mode=self.upsample_cfg["mode"],
            #                              align_corners=self.upsample_cfg["align_corners"])(out_master)
            #     out_master = self.drop_path(out_master)
            #     out_master = x_master + out_master
            #
            #     out_slave = self.m2s_conv(self.scale_convs(x_master))
            #     out_slave = self.drop_path(out_slave)
            #     out_slave = x_slave * out_slave
            #
            #     results = [out_slave, out_master]

            elif self.bt_mode == "s2m_mul" or self.bt_mode == "s2m_add":
                out_master = self.s2m_conv(x_slave)
                out_master = nn.Upsample(size=x_master.size()[2:],
                                         mode=self.upsample_cfg["mode"],
                                         align_corners=self.upsample_cfg["align_corners"])(out_master)
                if "mul" in self.bt_mode:
                    out_master = x_master * out_master
                else:
                    out_master = x_master + out_master
                out_master = self.drop_path(out_master)

                out_slave = self.drop_path(x_slave)

                results = [out_slave, out_master]

            elif self.bt_mode == "m2s_mul" or self.bt_mode == "m2s_add":
                out_slave = self.m2s_conv(self.scale_convs(x_master))
                if "mul" in self.bt_mode:
                    out_slave = x_slave * out_slave
                else:
                    out_slave = x_slave + out_slave
                out_slave = self.drop_path(out_slave)

                out_master = self.drop_path(x_master)

                results = [out_slave, out_master]

            elif self.bt_mode == "dual_mul_e" or self.bt_mode == "dual_add_e":
                out_master = self.s2m_conv(x_slave)
                out_master = nn.Upsample(size=x_master.size()[2:],
                                         mode=self.upsample_cfg["mode"],
                                         align_corners=self.upsample_cfg["align_corners"])(out_master)
                out_master = self.drop_path(out_master)

                if "mul" in self.bt_mode:
                    out_master = x_master * out_master
                else:
                    out_master = x_master + out_master

                out_slave = self.m2s_conv(self.scale_convs(x_master))
                out_slave = self.drop_path(out_slave)
                if "mul" in self.bt_mode:
                    out_slave = x_slave * out_slave
                else:
                    out_slave = x_slave + out_slave

                out_slave = nn.Upsample(size=out_master.size()[2:],
                                        mode=self.upsample_cfg["mode"],
                                        align_corners=self.upsample_cfg["align_corners"])(out_slave)
                out_slave = self.unify_conv(out_slave)
                out_slave = self.drop_path(out_slave)

                results = self.gather_convs(out_master + out_slave)
                # results = self.aspp_block(out_master + out_slave)

                results = [self.drop_path(results)]

            # elif self.bt_mode == "dual_add_mul_e":
            #     out_master = self.s2m_conv(x_slave)
            #     out_master = nn.Upsample(size=x_master.size()[2:],
            #                              mode=self.upsample_cfg["mode"],
            #                              align_corners=self.upsample_cfg["align_corners"])(out_master)
            #     out_master = self.drop_path(out_master)
            #     out_master = x_master + out_master
            #
            #     out_slave = self.m2s_conv(self.scale_convs(x_master))
            #     out_slave = self.drop_path(out_slave)
            #     out_slave = x_slave * out_slave
            #
            #     out_slave = nn.Upsample(size=out_master.size()[2:],
            #                             mode=self.upsample_cfg["mode"],
            #                             align_corners=self.upsample_cfg["align_corners"])(out_slave)
            #     out_slave = self.unify_conv(out_slave)
            #     out_slave = self.drop_path(out_slave)
            #
            #     results = self.gather_convs(out_master + out_slave)
            #
            #     results = [self.drop_path(results)]

            elif self.bt_mode == "s2m_mul_e" or self.bt_mode == "s2m_add_e":
                out_master = self.s2m_conv(x_slave)
                out_master = nn.Upsample(size=x_master.size()[2:],
                                         mode=self.upsample_cfg["mode"],
                                         align_corners=self.upsample_cfg["align_corners"])(out_master)
                out_master = self.drop_path(out_master)

                if "mul" in self.bt_mode:
                    out_master = x_master * out_master
                else:
                    out_master = x_master + out_master

                out_slave = nn.Upsample(size=out_master.size()[2:],
                                        mode=self.upsample_cfg["mode"],
                                        align_corners=self.upsample_cfg["align_corners"])(x_slave)

                out_slave = self.unify_conv(out_slave)
                out_slave = self.drop_path(out_slave)

                results = self.gather_convs(out_master + out_slave)

                results = [self.drop_path(results)]

            elif self.bt_mode == "m2s_mul_e" or self.bt_mode == "m2s_add_e":

                out_slave = self.m2s_conv(self.scale_convs(x_master))
                out_slave = self.drop_path(out_slave)
                if "mul" in self.bt_mode:
                    out_slave = x_slave * out_slave
                else:
                    out_slave = x_slave + out_slave

                out_slave = nn.Upsample(size=x_master.size()[2:],
                                        mode=self.upsample_cfg["mode"],
                                        align_corners=self.upsample_cfg["align_corners"])(out_slave)
                out_slave = self.unify_conv(out_slave)
                out_slave = self.drop_path(out_slave)

                results = self.gather_convs(x_master + out_slave)

                results = [self.drop_path(results)]

            else:
                results = [self.drop_path(self.s2m_conv(x_slave)),
                           self.drop_path(self.m2s_conv(x_master))]

            return results

        if self.with_cp and (x[0].requires_grad or x[1].requires_grad):
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)

        return outs


@BACKBONES.register_module()
class TwistNet_ResNet(BaseModule):
    # The Hybrid Bilateral Network
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 # ------------------Semantic Part----------------------
                 slave_depth,
                 slave_strides=(1, 2, 2, 2),
                 slave_dilations=(1, 1, 1, 1),
                 slave_avg_down=False,
                 slave_conv_cfg=None,
                 slave_norm_cfg=dict(type='BN', requires_grad=True),
                 slave_norm_eval=False,
                 slave_act_cfg=dict(type='ReLU'),
                 slave_drop_prob=-1,
                 slave_init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),

                 # --------------------Detail Part----------------------
                 master_channels=(128, 128, 256, 512),
                 master_blocks=(2, 2, 2, 2),
                 master_strides=(2, 2, 1, 1),
                 master_dilations=(1, 1, 2, 4),
                 m_module_paths=3,
                 m_module_groups=4,
                 m_module_expand_ratio=4,
                 ct_module_compress_ratios=(1, 1),# s2m, m2s
                 bt_modes=('dual_mul', 'dual_mul', 'dual_mul', 'dual_mul_e'),
                 down_type="stride",
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 master_order=('conv', 'norm', 'act'),
                 master_conv_cfg=None,
                 master_norm_cfg=dict(type='BN', requires_grad=True),
                 master_norm_eval=False,
                 master_act_cfg=dict(type='PReLU'),
                 master_drop_prob=-1,

                 # ---------------------Common Part----------------------
                 with_cp=False,
                 multi_modals=2):
        super(TwistNet_ResNet, self).__init__(init_cfg=None)

        # -------------------Slave----------------------
        if slave_depth not in self.arch_settings:
            raise KeyError(f'invalid slave_depth {slave_depth} for resnet')
        assert len(slave_strides) == len(slave_dilations)
        stem_channels = 64
        self.slave_norm_eval = slave_norm_eval
        self.block, self.stage_blocks = self.arch_settings[slave_depth]
        self.slave_init_cfg = slave_init_cfg

        # --------------------Master----------------------
        assert len(master_channels) == len(master_blocks) \
               == len(master_strides) == len(slave_strides)
        temp = []
        for master_channel in master_channels:
            mid_planes = make_divisible(master_channel, np.lcm(m_module_paths, m_module_groups))
            temp.append(mid_planes)
        master_channels = temp
        
        self.master_norm_eval = master_norm_eval

        # ------------------Common----------------------
        self.with_cp = with_cp
        self.multi_modals = multi_modals

        # -------------------Slave-----------------------
        self.slave_stages = []

        # --------------------Master----------------------
        self.master_stages = []
        self.bt_stages = []

        self.slave_stem = nn.Sequential(
            ConvModule(
                multi_modals * 3,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                conv_cfg=None,
                norm_cfg=slave_norm_cfg,
                act_cfg=dict(type='ReLU'),
                bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.master_stem = \
            ConvModule(
                multi_modals * 3,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                conv_cfg=None,
                norm_cfg=master_norm_cfg,
                act_cfg=dict(type='ReLU'),
                bias=False)

        slave_wh = 4
        master_wh = 2

        _slave_inplanes = stem_channels
        _slave_planes = stem_channels * self.block.expansion
        for i in range(len(self.stage_blocks)):
            slave_stage = self.make_slave_stage(
                block=self.block,
                num_blocks=self.stage_blocks[i],
                inplanes=_slave_inplanes,
                planes=_slave_planes // self.block.expansion,
                stride=slave_strides[i],
                dilation=slave_dilations[i],
                style='pytorch',
                avg_down=slave_avg_down,
                with_cp=with_cp,
                conv_cfg=slave_conv_cfg,
                norm_cfg=slave_norm_cfg,
                contract_dilation=True,
                drop_prob=slave_drop_prob)
            stage_name = f'slave_stage_{i + 1}'
            self.add_module(stage_name, slave_stage)
            self.slave_stages.append(stage_name)

            master_stage = Stage(
                inplanes=master_channels[i - 1] if i != 0 else stem_channels,
                planes=master_channels[i],
                stride=master_strides[i],
                dilation=master_dilations[i],
                m_module_paths=m_module_paths,
                m_module_groups=m_module_groups,
                m_module_expand_ratio=m_module_expand_ratio,
                num_blocks=master_blocks[i],
                down_type=down_type,
                conv_cfg=master_conv_cfg,
                norm_cfg=master_norm_cfg,
                act_cfg=master_act_cfg,
                order=master_order,
                drop_prob=master_drop_prob,
                with_cp=with_cp,
                init_cfg=None)
            master_stage_name = f'master_stage_{i + 1}'
            self.add_module(master_stage_name, master_stage)
            self.master_stages.append(master_stage_name)

            slave_wh *= slave_strides[i]
            master_wh *= master_strides[i]

            bt_stage = BT_Block(
                slave_out_channels=_slave_planes,
                master_out_channels=master_channels[i],
                ct_module_compress_ratios=ct_module_compress_ratios,
                conv_cfg=master_conv_cfg,
                norm_cfg=master_norm_cfg,
                act_cfg=master_act_cfg,
                bt_mode=bt_modes[i],
                order=master_order,
                upsample_cfg=upsample_cfg,
                slave_wh=slave_wh,
                master_wh=master_wh,
                drop_prob=master_drop_prob,
                with_cp=with_cp,
                init_cfg=None)
            bt_stage_name = f'master_bt_stage_{i + 1}'
            self.add_module(bt_stage_name, bt_stage)
            self.bt_stages.append(bt_stage_name)

            _slave_inplanes = _slave_planes
            _slave_planes *= 2

        self._freeze_stages()

    def make_slave_stage(self, **kwargs):
        return ResLayer(**kwargs)

    def _freeze_stages(self):
        pass

    def init_weights(self):
        # ------------------Semantic Part----------------------
        if self.slave_init_cfg is not None:
            logger = get_root_logger()
            load_checkpoint(self, self.slave_init_cfg['checkpoint'], strict=False, logger=logger)
        else:
            for name, m in self.named_modules():
                if "stem" in name and "master" not in name:
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                        constant_init(m, 1)

        # --------------------Detail Part----------------------
        for name, m in self.named_modules():
            if "master" in name and "stem" not in name:
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        x_slave = self.slave_stem(x)
        x_master = self.master_stem(x)

        outs = []
        for i in range(len(self.stage_blocks)):
            slave_name = self.slave_stages[i]
            slave_stage = getattr(self, slave_name)
            x_slave = slave_stage(x_slave)

            master_stage_name = self.master_stages[i]
            master_stage = getattr(self, master_stage_name)
            x_master = master_stage(x_master)

            if i < len(self.stage_blocks) - 1:
                outs.append(x_master)

            bt_stage_name = self.bt_stages[i]
            bt_stage = getattr(self, bt_stage_name)
            bt_outs = bt_stage([x_slave, x_master])

            if len(bt_outs) > 1:
                x_slave, x_master = bt_outs[0], bt_outs[1]
            else:
                x_master = bt_outs[0]
                outs.append(x_master)

        return tuple(outs)

    def train(self, mode=True):
        super(TwistNet_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.slave_norm_eval:
            for name, m in self.named_modules():
                if "slave" in name:
                    if isinstance(m, _BatchNorm):
                        m.eval()

        if mode and self.master_norm_eval:
            for name, m in self.named_modules():
                if "master" in name:
                    if isinstance(m, _BatchNorm):
                        m.eval()
