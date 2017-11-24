"""
resnet_original.py

this uses pytorch to implement deep residual neural networks from:

"Deep residual learning for image recognition" (He, et al., 2016)

this version uses the original residual blocks (both bottleneck blocks and
basic blocks)

author:     alex shenfield
date:       20/11/2017

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
    
# original bottleneck residual block
class OriginalBottleneckBlock(nn.Module):
    
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
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # relu activation function
        self.relu = nn.ReLU(inplace=True)
        
        # 3x3 convolution
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)

        # bottleneck expand
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    # this where all the layers are put together
    def forward(self, x):
        
        # x is our input 
        residual = x

        # build the convolutional path (using old resnet style ...)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if we are down sampling in this block, we include the "downsample"
        # block in the shortcut path - this is the 1x1 convolution with stride
        # of two (followed by batch norm)
        if self.downsample is not None:
            residual = self.downsample(x)

        # so we add the residual (shortcut path) and apply relu to the result
        out += residual
        out = self.relu(out)

        return out
    

# basic original residual block
class OriginalBasicBlock(nn.Module):
    
    # in the basic block, we don't need to reexpand the planes
    expansion = 1

    # this is where all the layers are defined
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        
        # what this does is make sure we call all the methods in the right 
        # order
        # see https://stackoverflow.com/a/27134600
        
        # python 3 allows use of super like this ...
        super().__init__()
        
        # define our layers
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    # this where all the layers are put together
    def forward(self, x):
        
        # x is our input 
        residual = x

        # build the convolutional path (using old resnet style ...)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if we are down sampling in this block, we include the "downsample"
        # block in the shortcut path - this is the 1x1 convolution with stride
        # of two (followed by batch norm)
        if self.downsample is not None:
            residual = self.downsample(x)

        # so we add the residual (shortcut path) and apply relu to the result
        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
        
    # what this does is make sure we call all the methods in the right 
    # order
    # see https://stackoverflow.com/a/27134600
      
    # python 3 allows use of super like this ...
    def __init__(self, block, depth, num_classes=10):
                
        super().__init__()
        
        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
        
        # first convolutional layer + batchnorm + relu
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # layer up the residual blocks
        self.inplanes = 16
        self.layer1 = self.layer(block, 16, layer_blocks, stride=1)
        self.layer2 = self.layer(block, 32, layer_blocks, stride=2)
        self.layer3 = self.layer(block, 64, layer_blocks, stride=2)
        
        # add global average pooling 
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
        
        # first convolution layer + batch norm +relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # global average pooling
        x = self.avgpool(x)
        
        # confusingly - view reshapes the tensor ...
        x = x.view(x.size(0), -1)
        
        # our fully connected final layer
        x = self.fc(x)

        return x
    
    
def resnet110(**kwargs):
    """Constructs a ResNet-110 model.
    """
    model = ResNetCifar10(BasicBlock, 110, **kwargs)
    return model