"""
cifar10_resnet_v2.py

this uses pytorch to implement deep residual neural networks from:

"Deep residual learning for image recognition" (He, et al., 2016)
"Identity mappings in deep residual neural networks" (He, et al., 2016)

this version uses the improved residual blocks with preactivation from the 
second paper which have been shown to improve results (especially in even 
deeper networks such as resnet 1001 with over a thousand layers) 

this module allows for actually building the network(s)

author:     alex shenfield
date:       22/11/2017

"""

from .resnet_blocks_v2 import PreactivationBasicBlock
from .resnet_blocks_v2 import PreactivationBottleneckBlock

import torch.nn as nn
import math

#
# the key differences between the cifar10 resnet below and the one used for 
# training on imagenet are:
#
# 1) the use of maxpooling in the initial block, 
# 2) the number of filters used in the convolutional blocks, 
# 3) the number of layers,
# 4) the 7x7 average pooling before the final fully connected layer,
# 5) the fact that the imagenet resnet architecture uses bottleneck blocks 
#    throughout to reduce computation time (though the resnet 1001 
#    architecture also uses bottleneck blocks)
#

# build a version of the deep residual neural network architecture from 
# "Identity mappings in deep residual neural networks"suitable for training 
# on cifar 10
class Cifar10ResNet(nn.Module):

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
        
        # describe the model we're building
        print('preactivation based resnet for Cifar10 (using basic blocks)')
        print('depth: {}'.format(depth))
        print('layers for each block: {}'.format(layer_blocks))
        
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
        
        # final batch norm and relu (according to the torch version)
        # -> check the paper!
        self.bn_end   = nn.BatchNorm2d(64)        
        self.relu_end = nn.ReLU(inplace=True)
        
        # add global average pooling (on the output size of 8) and the fully 
        # connected linear layer
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # initialise all the weights and biases
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


# build a version of the deep residual neural network architecture from 
# "Identity mappings in deep residual neural networks"suitable for training 
# on cifar 10
#
# this class implements the architecture for the ridiculously deep 
# resnet-1001 on cifar 10
class Cifar10ResNet1001(nn.Module):

    def __init__(self, block, depth, num_classes=10):
        
        # what this does is make sure we call all the methods in the right 
        # order
        # see https://stackoverflow.com/a/27134600
        
        # python 3 allows use of super like this ...                
        super().__init__()
        
        # the depth of the cifar 10 model using the basic blocks should be 
        # 2 (because of the initial layer and the final classification layer)
        # + a mulitple of 9 (because we have 3 separate blocks of residual 
        # units with differing numbers of filters each with 3n convolutional 
        # units because of the bottleneck architecture)
        assert (depth - 2) % 9 == 0, 'depth should be 9n + 2'
        layer_blocks = (depth - 2) // 9
        
        # describe the model we're building
        print('preactivation based resnet for Cifar10 ' +
              '(using bottleneck blocks)')
        print('depth: {}'.format(depth))
        print('layers for each block: {}'.format(layer_blocks))
        
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
        
        # final batch norm and relu (according to the torch version)
        # -> check the paper!
        self.bn_end   = nn.BatchNorm2d(256)        
        self.relu_end = nn.ReLU(inplace=True)
        
        # add global average pooling (on the output size of 8) and the fully 
        # connected linear layer
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(256, num_classes)

        # initialise all the weights and biases
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

#
# builder methods for specific architectures
#
        
# construct a resnet 110 model (with preactivations) for cifar 10
def resnet110(**kwargs):
    model = Cifar10ResNet(PreactivationBasicBlock, 110, **kwargs)
    return model

# construct a resnet 1001 model (with preactivations) for cifar 10
def resnet1001(**kwargs):
    model = Cifar10ResNet1001(PreactivationBottleneckBlock, 1001, **kwargs)
    return model