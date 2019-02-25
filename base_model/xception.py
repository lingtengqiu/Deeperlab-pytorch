from __future__ import print_function, division, absolute_import
from utils.pyt_utils import load_model
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Xception','xception71']



class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,norm_layer = nn.BatchNorm2d,eps = 1e-5,momentum = 0.1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = norm_layer(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(norm_layer(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(norm_layer(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(norm_layer(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self,inplane=3,norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1,inplace =True):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()


        self.conv1 = nn.Conv2d(inplane, 32, 3,2, 1, bias=False)
        self.bn1 = norm_layer(32,eps=bn_eps, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=inplace)

        self.conv2 = nn.Conv2d(32,64,3,padding =1,bias=False)
        self.bn2 = norm_layer(64,eps=bn_eps, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=inplace)

        # firs two conv

        #do relu here

        #1/2
        self.block1=Block(64,128,2,2,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=False,grow_first=True)
        #1/4
        self.block2=Block(128,256,2,2,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)
        #1/8
        self.block3=Block(256,728,2,2,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=True)

        #1/16
        self.block12=Block(728,1024,2,2,norm_layer = norm_layer,eps = bn_eps,momentum = bn_momentum,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = norm_layer(1536,eps=bn_eps, momentum=bn_momentum)
        self.relu3 = nn.ReLU(inplace=inplace)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = norm_layer(2048,eps=bn_eps, momentum=bn_momentum)

        self.relu4 = nn.ReLU(inplace)

        # #------- initweights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        #1/2
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #1/4
        x = self.block1(x)
        low_feature = x
        #1/8
        x = self.block2(x)
        #1/16
        x = self.block3(x)

        #mid_layer
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.relu3(x)
        #1/16
        return low_feature,x

        # x = self.block12(x)
        # #1/32
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        #
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu4(x)
    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        return x
def xception71(pretrained_model=None,**kwargs):
    model = Xception(**kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model
if __name__ == '__main__':
    pass