#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2019-02-22 13:53
# * Last modified : 2019-02-22 13:53
# * Filename      : dfn.py
# * Description   : 
# **********************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# from torchvision.models import resnet50, resnet101, resnet152

from config import config
from base_model import resnet101

#resnet struct for here is get block which name is cov1,cov2,conv3,conv4
#suppose input tensor size is (3,224,224)
#cov1 {256,56,56} 1/4
#cov2 {512,28,28} 1/8
#cov3 {1024,14,14} 1/16
#conv4 {2048,7,7} 1/32

#all here is realize from the face++ Discriminative feature network for semantic segmentation

from seg_opr.seg_oprs import ConvBnRelu, BNRefine, RefineResidual, \
    ChannelAttention
from seg_opr.loss_opr import SigmoidFocalLoss
#global avg target (bs,c,h,w)->(bs,h*w,1,1)
#dfn about Smooth NetWork ---> crossentropy
#dfn about Border Network ---> sigmoid_focal_loss


class DFN(nn.Module):
    def __init__(self, out_planes, criterion, aux_criterion, alpha,
                 pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(DFN, self).__init__()
        self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=False, stem_width=64)
        self.business_layer = []

        smooth_inner_channel = 512
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(2048, smooth_inner_channel, 1, 1, 0,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )
        self.business_layer.append(self.global_context)

        stage = [2048, 1024, 512, 256]
        self.smooth_pre_rrbs = []
        self.cabs = []
        self.smooth_aft_rrbs = []
        self.smooth_heads = []

        #every stage we have dfn layer output?
        # top to bottom 2048 -> 256
        for i, channel in enumerate(stage):
            self.smooth_pre_rrbs.append(
                RefineResidual(channel, smooth_inner_channel, 3, has_bias=False,
                               has_relu=True, norm_layer=norm_layer))
            self.cabs.append(
                ChannelAttention(smooth_inner_channel * 2,
                                 smooth_inner_channel, 1))
            self.smooth_aft_rrbs.append(
                RefineResidual(smooth_inner_channel, smooth_inner_channel, 3,
                               has_bias=False,
                               has_relu=True, norm_layer=norm_layer))
            self.smooth_heads.append(
                DFNHead(smooth_inner_channel, out_planes, scale=2 ** (5 - i),
                        norm_layer=norm_layer))

        stage.reverse()
        border_inner_channel = 21
        self.border_pre_rrbs = []
        self.border_aft_rrbs = []
        self.border_heads = []


        for i, channel in enumerate(stage):
            self.border_pre_rrbs.append(
                RefineResidual(channel, border_inner_channel, 3, has_bias=False,
                               has_relu=True, norm_layer=norm_layer))
            self.border_aft_rrbs.append(
                RefineResidual(border_inner_channel, border_inner_channel, 3,
                               has_bias=False,
                               has_relu=True, norm_layer=norm_layer))
            self.border_heads.append(
                DFNHead(border_inner_channel, 1, 4, norm_layer=norm_layer))

        self.smooth_pre_rrbs = nn.ModuleList(self.smooth_pre_rrbs)
        self.cabs = nn.ModuleList(self.cabs)
        self.smooth_aft_rrbs = nn.ModuleList(self.smooth_aft_rrbs)
        self.smooth_heads = nn.ModuleList(self.smooth_heads)
        self.border_pre_rrbs = nn.ModuleList(self.border_pre_rrbs)
        self.border_aft_rrbs = nn.ModuleList(self.border_aft_rrbs)
        self.border_heads = nn.ModuleList(self.border_heads)

        #smooth model
        self.business_layer.append(self.smooth_pre_rrbs)
        self.business_layer.append(self.cabs)
        self.business_layer.append(self.smooth_aft_rrbs)
        self.business_layer.append(self.smooth_heads)

        #border_layer model
        self.business_layer.append(self.border_pre_rrbs)
        self.business_layer.append(self.border_aft_rrbs)
        self.business_layer.append(self.border_heads)

        self.criterion = criterion
        self.aux_criterion = aux_criterion
        self.alpha = alpha

    def forward(self, data, label=None, aux_label=None):
        blocks = self.backbone(data)


        '''
        >> Block we have 4 conv1 conv2 conv3 conv4
        conv1 -> 256 1/4
        conv2 -> 512 1/8
        conv3 ->1024 1/16
        conv4 ->2048 1/32
        '''
        blocks.reverse()

        #understanding the global context meaning:

        global_context = self.global_context(blocks[0])
        #global: ->(bs,512,1,1)
        #equal to squeeze bar
        global_context = F.interpolate(global_context,
                                       size=blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)
        #this part using for smooth model ,this part sometime is important, U must carefully

        last_fm = global_context
        #last_fm --->(bs,512,1/32,1/32)
        pred_out = []

        for i, (fm, pre_rrb,
                cab, aft_rrb, head) in enumerate(zip(blocks,
                                                     self.smooth_pre_rrbs,
                                                     self.cabs,
                                                     self.smooth_aft_rrbs,
                                                     self.smooth_heads)):
            #step RRB model
            fm = pre_rrb(fm)
            #CAB para: low_level,hight_level
            fm = cab(fm, last_fm)
            fm = aft_rrb(fm)
            #RRB model ,next using to next low level , we must up sample

            #what is head mean?
            pred_out.append(head(fm))
            if i != 3:
                last_fm = F.interpolate(fm, scale_factor=2, mode='bilinear',
                                        align_corners=True)


        #every conv# have a predict label (bs,num_label,input_size,input_size)
        blocks.reverse()
        #change blocks from bottom to top
        last_fm = None
        boder_out = []
        #struct like follow border
        '''
        conv1 ---RRB------------>head
                       |
        conv2 ---RRB---+--RRB--->head
                            |
        conv3 ---RRB---+--RRB--->head
                            |
        conv4 ---RRB---+--RRB--->head
        '''
        for i, (fm, pre_rrb,
                aft_rrb, head) in enumerate(zip(blocks,
                                                self.border_pre_rrbs,
                                                self.border_aft_rrbs,
                                                self.border_heads)):
            fm = pre_rrb(fm)
            if last_fm is not None:
                fm = F.interpolate(fm, scale_factor=2 ** i, mode='bilinear',
                                   align_corners=True)
                last_fm = last_fm + fm
                last_fm = aft_rrb(last_fm)

            else:
                last_fm = fm
            boder_out.append(head(last_fm))

        #if train: loss have 4 layer
        #else print the smooth_layer.
        #this method is good , we must learning this step
        if label is not None and aux_label is not None:
            loss0 = self.criterion(pred_out[0], label)
            loss1 = self.criterion(pred_out[1], label)
            loss2 = self.criterion(pred_out[2], label)
            loss3 = self.criterion(pred_out[3], label)

            aux_loss0 = self.aux_criterion(boder_out[0], aux_label)
            aux_loss1 = self.aux_criterion(boder_out[1], aux_label)
            aux_loss2 = self.aux_criterion(boder_out[2], aux_label)
            aux_loss3 = self.aux_criterion(boder_out[3], aux_label)

            loss = loss0 + loss1 + loss2 + loss3
            aux_loss = aux_loss0 + aux_loss1 + aux_loss2 + aux_loss3
            return loss + self.alpha * aux_loss

        return F.log_softmax(pred_out[-1], dim=1)

4
class DFNHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm2d):
        #remeber the scale means  upsample times
        super(DFNHead, self).__init__()
        self.rrb = RefineResidual(in_planes, out_planes * 9, 3, has_bias=False,
                                  has_relu=False, norm_layer=norm_layer)
        self.conv = nn.Conv2d(out_planes * 9, out_planes, kernel_size=1,
                              stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        x = self.rrb(x)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear',
                          align_corners=True)

        return x


if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=255)
    aux_criterion = SigmoidFocalLoss(ignore_label=255, gamma=2.0, alpha=0.25)
    model = DFN(21, criterion,aux_criterion,0.25,"/home/qlt/qiulingteng/seg/TorchSeg-master/model/dfn_v1/pretrain/resnet101_torch.pth")
    print(model)
    # model = DFN(21, criterion,aux_criterion,0.25)
    xxx
    print(model)
    xxx
    model(torch.randn(2,3,448,448))
    # print(model)
