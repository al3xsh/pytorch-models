"""
resnet_blocks_v2.py

this uses pytorch to implement deep residual neural networks from:

"Deep residual learning for image recognition" (He, et al., 2016)
"Identity mappings in deep residual neural networks" (He, et al., 2016)

this version uses the improved residual blocks with preactivation from the 
second paper which have been shown to improve results (especially in even 
deeper networks such as resnet 1001 with over a thousand layers) 

this module provides the two types of blocks used in creating deep residual 
neural networks - 

the basic block (useful for smaller resnets and resnet 110 on cifar10)
the bottleneck block (useful for larger resnets such as resnet 1001 applied 
to cifar 10 & resnets applied to huge datasets such as imagenet)

author:     alex shenfield
date:       22/11/2017

"""

import torch.nn as nn

#
# helper functions to simplify the creation of residual deep neural networks
#

# return a 3x3 convolution with implicit zero padding padding of 1 pixel on 
# each side (this is the basic convolutional unit within the resnet 
# architecture)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    

# basic residual block with preactivation from "Identity mappings in deep 
# residual neural networks" 
class PreactivationBasicBlock(nn.Module):
    expansion = 1

    # this is where all the layers are defined
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        
        # what this does is make sure we call all the methods in the right 
        # order
        # see https://stackoverflow.com/a/27134600
        
        # python 3 allows use of super like this ...
        super().__init__()
        
        # define our layers        
        self.bn1   = nn.BatchNorm2d(inplanes)        
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2   = nn.BatchNorm2d(planes)        
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        self.downsample = downsample
        self.stride = stride

    # this where all the layers are put together
    def forward(self, x):
        
        # x is our input 
        residual = x

        # build the convolutional path (using preactivation based resnet 
        # architecture style ...)
        out = self.bn1(x)
        out = self.relu1(out)
        
        # if we are down sampling in this block, we include the "downsample"
        # block in the shortcut path - this is the 1x1 convolution with stride
        # of two preceeded by batch norm and relu
        if self.downsample is not None:
            residual = self.downsample(out)
        
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        # so we add the residual (shortcut path) and return the result
        out += residual
        return out
    

# bottleneck residual block with preactivation from "Identity mappings in deep 
# residual neural networks" 
# - used in resnet 1001 and imagenet resnets
class PreactivationBottleneckBlock(nn.Module):
    
    # this controls the re-expansion of the bottleneck
    expansion = 4

    # this is where all the layers are defined
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        
        # what this does is make sure we call all the methods in the right 
        # order
        # see https://stackoverflow.com/a/27134600
        
        # python 3 allows use of super like this ...
        super().__init__()
        
        # define our layers      
        
        # bottleneck constrict
        self.bn1   = nn.BatchNorm2d(inplanes)        
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, 
                               kernel_size=1, stride=stride, bias=False)
        
        # 3x3 convolution
        self.bn2   = nn.BatchNorm2d(planes)        
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        # bottleneck expand
        self.bn3   = nn.BatchNorm2d(planes)        
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 
                               kernel_size=1, bias=False)
        
        self.downsample = downsample
        self.stride = stride

    # this where all the layers are put together
    def forward(self, x):
        
        # x is our input 
        residual = x

        # build the convolutional path (using preactivation based resnet 
        # architecture style ...)
        out = self.bn1(x)
        out = self.relu1(out)
        
        # if we are down sampling in this block, we include the "downsample"
        # block in the shortcut path - this is the 1x1 convolution with stride
        # of two preceeded by batch norm and relu
        if self.downsample is not None:
            residual = self.downsample(out)
        
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        # so we add the residual (shortcut path) and return the result
        out += residual
        return out


