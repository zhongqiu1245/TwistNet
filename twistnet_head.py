import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import ConvModule

from .decode_head import BaseDecodeHead

from ..utils.make_divisible import make_divisible, find_nearest_divisible_factor
from ..builder import HEADS


class M_Module(nn.Module):
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
                 drop_prob=0.0):
        super(M_Module, self).__init__()

        self.expand_ratio = expand_ratio
        self.paths = paths
        self.groups = groups

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


class Up_Layer(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 skipplanes,
                 kernel_size=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="PReLU"),
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 drop_prob=0.0):
        super(Up_Layer, self).__init__()

        self.upsample_cfg = upsample_cfg

        self.up_conv = ConvModule(
            inplanes + skipplanes,
            planes,
            kernel_size,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=order)

        self.drop_path = nn.Dropout2d(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, skip, x):
        x = nn.Upsample(size=skip.size()[2:],
                        mode=self.upsample_cfg["mode"],
                        align_corners=self.upsample_cfg["align_corners"])(x)

        out = self.up_conv(torch.cat([skip, x], 1))

        out = self.drop_path(out)

        return out


class Stage(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 skipplanes,
                 m_module_paths=3,
                 m_module_groups=3,
                 m_module_expand_ratio=4,
                 num_blocks=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type="PReLU"),
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 drop_prob=0.0):
        super(Stage, self).__init__()

        self.up_layer = \
            Up_Layer(inplanes,
                     planes,
                     skipplanes,
                     conv_cfg=conv_cfg,
                     norm_cfg=norm_cfg,
                     act_cfg=act_cfg,
                     order=order,
                     upsample_cfg=upsample_cfg,
                     drop_prob=drop_prob)

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
                drop_prob=drop_prob) for _ in range(num_blocks)])

    def forward(self, skip, x):

        out = self.up_layer(skip, x)

        out = self.m_layer(out)

        return out


@HEADS.register_module()
class TwistNet_Head(BaseDecodeHead):

    def __init__(self,
                 m_module_paths=3,
                 m_module_groups=3,
                 m_module_expand_ratio=4,
                 blocks=(2, 2, 2, 2),
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(mode='bilinear'),
                 drop_prob=-1,
                 **kwargs):
        super(TwistNet_Head, self).__init__(**kwargs)

        temp = []
        for in_channel in self.in_channels:
            mid_planes = make_divisible(in_channel, np.lcm(m_module_paths, m_module_groups))
            temp.append(mid_planes)
        self.in_channels = temp

        skip_channels = self.in_channels[1:]

        assert self.channels == self.in_channels[-1]

        upsample_cfg['align_corners'] = self.align_corners

        self.stages = nn.ModuleList()
        for i in range(len(skip_channels)):
            self.stages.append(
                Stage(
                    inplanes=self.in_channels[i],
                    planes=self.in_channels[i + 1],
                    skipplanes=skip_channels[i],
                    m_module_paths=m_module_paths,
                    m_module_groups=m_module_groups,
                    m_module_expand_ratio=m_module_expand_ratio,
                    num_blocks=blocks[i],
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    order=order,
                    upsample_cfg=upsample_cfg,
                    drop_prob=drop_prob))

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        x = inputs[::-1]
        output = x[0]

        for i in range(len(self.stages)):
            output = self.stages[i](x[i + 1], output)

        output = self.cls_seg(output)

        return output
