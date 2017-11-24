"""
cifar_resnet_v2.py

this uses pytorch to implement deep residual neural networks from:

"Deep residual learning for image recognition" (He, et al., 2016)
"Identity mappings in deep residual neural networks" (He, et al., 2016)

this version uses the improved residual blocks with preactivation from the 
second paper which have been shown to improve results (especially in even 
deeper networks such as resnet 1001 with over a thousand layers) 

author:     alex shenfield
date:       22/11/2017

"""

import torch.nn as nn
import math

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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        # 3x3 convolution
        self.bn2   = nn.BatchNorm2d(planes)        
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        # bottleneck expand
        self.bn3   = nn.BatchNorm2d(inplanes)        
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, 
                               planes * self.expansion, 
                               kernel_size=1, 
                               bias=False)
        
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


# build a version of the deep residual neural network architecture from 
# "Identity mappings in deep residual neural networks"suitable for training 
# on cifar 10
#
# the key differences between this and the one used for training on imagenet 
# are:
# 1) the use of maxpooling in the initial block, 
# 2) the number of filters used in the convolutional blocks, 
# 3) the number of layers,
# 4) the 7x7 average pooling before the final fully connected layer,
# 5) the fact that the imagenet resnet architecture uses bottleneck blocks 
#    throughout to reduce computation time
class ResNetCifar10(nn.Module):

    def __init__(self, block, depth, num_classes=10):
        
        # what this does is make sure we call all the methods in the right 
        # order
        # see https://stackoverflow.com/a/27134600
        
        # python 3 allows use of super like this ...                
        super().__init__()
        
        # the depth of the cifar 10 model using the basic blocks should be 
        # 2 (because of the initial layer and the final classification layer)
        # + a mulitple of 6 (because we have 3 separate blocks of residual 
        # units with differing numbers of filters each with 2n convolutional 
        # units)
        assert (depth - 2) % 6 == 0, 'depth should be 6n + 2'
        layer_blocks = (depth - 2) // 6
        print ('Preactivation based resnet for Cifar10 (using basic blocks)')
        print('Depth: {}'.format(depth))
        print('Layers for each block: {}'.format(layer_blocks))
        
        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=16, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               bias=False)
        
        # layer up the residual blocks
        self.inplanes = 16
        self.layer1 = self.layer(block, 16, layer_blocks, stride=1)
        self.layer2 = self.layer(block, 32, layer_blocks, stride=2)
        self.layer3 = self.layer(block, 64, layer_blocks, stride=2)
        
        # final batch norm and relu (according to the torch version) - I think
        # this is 128 ... though the torch version uses different numbers of
        # filters
        # -> check the paper!
        self.bn_end   = nn.BatchNorm2d(64)        
        self.relu_end = nn.ReLU(inplace=True)
        
        # add global average pooling (on the output size of 8) and the fully 
        # connected linear layer
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # initialise all the weights abd biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    # lets layer up the residual blocks ...
    def layer(self, block, planes, blocks, stride=1):
        
        # down sample on the shortcut path if needed 
        #
        # (i.e. the first residual block in each set - apart from the very 
        # first set, where the input dimension - 16 - is the same as the 
        # output dimension of the convolutional path - 16 - because of the 
        # very first convolutional layer)
        downsample = None
        if (stride != 1) or (self.inplanes != (planes * block.expansion)):
            
            # down sample by adding a convolution layer with 1x1 filters and
            # 2x2 stride
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # build all the residual blocks in this layer
        layers = []
        
        # the first residual block in this layer will use downsampling 
        # _unless_ its the very first residual layer
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # return the layered residual blocks
        return nn.Sequential(*layers)

    # this where all the layers are put together
    def forward(self, x):
        
        # first convolution layer
        x = self.conv1(x)

        # residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # ending batch norm and relu before average pooling and the fully 
        # connected layer
        x = self.bn_end(x)
        x = self.relu_end(x)

        # global average pooling
        x = self.avgpool(x)
        
        # confusingly - view reshapes the tensor ...
        x = x.view(x.size(0), -1)
        
        # our fully connected final layer
        x = self.fc(x)

        return x
    

# construct a resnet 110 model (with preactivations) for cifar 10
def resnet110(**kwargs):
    model = ResNetCifar10(PreactivationBasicBlock, 110, **kwargs)
    return model