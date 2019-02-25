#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2019-02-25 18:50
# * Last modified : 2019-02-25 18:50
# * Filename      : deeperlab.py
# * Description   : detailed see https://arxiv.org/abs/1902.05093
# **********************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
# import .depend
from base_model import xception
from config import config
from seg_opr.seg_oprs import ConvBnRelu

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'deeperlab':
            inplanes = 728
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 3, 6, 12]
        elif output_stride == 8:
            dilations = [1, 6, 12, 18]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        return x



class space_to_dense(nn.Module):
    def __init__(self,stride):
        super(space_to_dense,self).__init__()
        self.stride = stride
    def forward(self, input):
        assert len(input.shape) == 4,"input tensor must be 4 dimenson"
        stride = self.stride
        B,C,W,H = input.shape
        assert (W %stride == 0 and H %stride == 0),"the W = {} or H = {} must be divided by {}".format(W,H,stride)
        ws = W // stride
        hs = H // stride
        x = input.view(B, C, hs, stride, ws, stride).transpose(3, 4).contiguous()
        x = x.view(B, C, hs*ws, stride * stride).transpose(2, 3).contiguous()
        x = x.view(B, C, stride * stride, hs, ws).transpose(1, 2).contiguous()
        x = x.view(B, stride * stride * C, hs, ws)
        return x
class dense_to_space(nn.Module):
    def __init__(self,stride):
        super(dense_to_space,self).__init__()
        self.stride = stride
        self.ps = torch.nn.PixelShuffle(stride)
    def forward(self, input):
        return self.ps(input)
class deeperlab(nn.Module):
    def __init__(self, inplane,outplane,criterion=None, aux_criterion=None, area_alpa=None,
                 pretrained_model=None,
                 norm_layer=nn.BatchNorm2d,detection = False):
        super(deeperlab,self).__init__()
        self.backbone =xception.xception71(pretrained_model,inplane=inplane,norm_layer=norm_layer,bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,inplace =True)
        self.business_layer = []
        self.s2d = space_to_dense(4)
        self.d2s  = torch.nn.PixelShuffle(upscale_factor=4)

        self.aspp = ASPP("deeperlab",8,norm_layer)
        self.conv1 = ConvBnRelu(128, 32,1,1,0,norm_layer=norm_layer,bn_eps= config.bn_eps)
        self.conv2 = ConvBnRelu(768,4096,3,1,1,norm_layer=norm_layer,bn_eps=config.bn_eps)
        self.conv3 = ConvBnRelu(4096,4096,3,1,1,norm_layer=norm_layer,bn_eps=config.bn_eps)

        self.seg_conv = deeperlab_seg_head(256,outplane,4,norm_layer= norm_layer)
        self.business_layer.append(self.s2d)
        self.business_layer.append(self.d2s)
        self.business_layer.append(self.aspp)
        self.business_layer.append(self.conv1)
        self.business_layer.append(self.conv2)
        self.business_layer.append(self.conv3)
        self.business_layer.append(self.seg_conv)
        self.criterion = criterion
    def forward(self, input,label=None, aux_label=None):
        low_level,high_level= self.backbone(input)
        high_level = self.aspp(high_level)

        low_level  = self.conv1(low_level)
        low_level = self.s2d(low_level)
        decode = torch.cat((high_level,low_level),dim = 1)
        decode = self.conv2(decode)
        decode = self.conv3(decode)

        decode = self.d2s(decode)
        pre = self.seg_conv(decode)
        if label is not None :
            loss = self.criterion(pre,label)
            return loss
        return F.log_softmax(pre,dim=1)
class deeperlab_seg_head(nn.Module):
    def __init__(self,inplane,outplane,scale = 4 ,norm_layer=nn.BatchNorm2d):
        super(deeperlab_seg_head,self).__init__()
        self.conv = ConvBnRelu(inplane,256,7,1,3,norm_layer=norm_layer,bn_eps=config.bn_eps)
        self.conv_seg = nn.Conv2d(256, outplane, kernel_size=1,stride=1, padding=0)
        self.scale = scale
    def forward(self, x):
        x = self.conv(x)
        x = self.conv_seg(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear',
                          align_corners=True)
        return x

if __name__ == '__main__':
    # input = torch.randn(2,1,16,16)
    # for k in range(2):
    #     for i in range(16):
    #         for j in range(16):
    #             input[k][0][i][j] =k*256 + i*16+j
    # s2d = space_to_dense(2)
    # d2s = dense_to_space(2)
    # x = (s2d(input))
    # print(d2s(x))
    #model = deeperlab(3,21,pretrained_model="../pretrain/xception-71.pth")
    # criterion = nn.CrossEntropyLoss(reduction='mean',
    #                                 ignore_index=255)
    # x = torch.randn(1,2,5,5)
    # label = torch.ones(1,5,5).long()
    # loss = criterion(x,label)
    # print(loss.backward())
    # xxx
    # print(loss.requires_grad)
    # loss = loss.view(-1)
    # print(loss)
    # loss = loss[torch.argsort(loss,descending = True)][0:5].mean()
    # print(loss.requires_grad)
    pass



