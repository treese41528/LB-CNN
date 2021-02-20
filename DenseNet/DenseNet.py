# -*- coding: utf-8 -*-
# Implementation of Densely Connected Convolutional Networks (CVPR 2017)
# https://arxiv.org/abs/1608.06993


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os

class DenseCompositionFunction(nn.Module):
    def __init__(self,in_channels,growth_rate,bottle_neck = True):
        super(DenseCompositionFunction,self).__init__()
        bn_size = 4
        self.bottle_neck = bottle_neck
        
        if self.bottle_neck:
            self.bn_1 = nn.BatchNorm2d(in_channels)
            self.conv_1 = nn.Conv2d(in_channels=in_channels,kernel_size=[1,1],
                                  out_channels=bn_size*growth_rate,stride=[1,1],padding=0,bias = False)
            self.bn_2 = nn.BatchNorm2d(bn_size*growth_rate)
            self.conv_2 = nn.Conv2d(in_channels=bn_size*growth_rate,kernel_size=[3,3],
                              out_channels=growth_rate,stride=[1,1],padding=1,bias = False)
        else:
            self.bn_1 = nn.BatchNorm2d(in_channels)
            self.conv_1 = nn.Conv2d(in_channels=in_channels,kernel_size=[3,3],
                                  out_channels=growth_rate,stride=[1,1],padding=1,bias = False)
            
    def forward(self,x_input):
        out = self.conv_1(F.relu(self.bn_1(x_input)))
        if self.bottle_neck == True:
            out = self.conv_2(F.relu(self.bn_2(out)))
        out = torch.cat((x_input,out),1)
        return out

class DenseBlock(nn.Module):
    def __init__(self,num_layers,in_channels,growth_rate,bottle_neck = True):
        super(DenseBlock,self).__init__()
        
        self.layers = nn.ModuleDict()
        for l in range(num_layers):
            layer = DenseCompositionFunction(in_channels + l * growth_rate,growth_rate,bottle_neck)
            self.layers['denselayer%d' % (l + 1)] = layer
       
    def forward(self, init_features):
        feature_maps = init_features
        for name, layer in self.layers.items():
            new_features = layer(feature_maps)
            feature_maps = new_features
        return feature_maps 
    
class Transition_Layer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(Transition_Layer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1,padding=0, bias=False)
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2) 
        return out
    
class DenseNet(nn.Module):
    def __init__(self,growth_rate,layers,num_classes,channels = 3):
        super(DenseNet, self).__init__()
        self.k = growth_rate  #Growth Rate 
        self.L = layers  #Number of Layers in each Dense block
        
        self.compression = 2 #Set to 2 for compression
        self.bottle_neck = True
        
        #Image Dimensions
        self.channels= channels
        self.num_classes = num_classes
        
        self.num_init_features = int(2*self.k)
        #First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(self.channels, self.num_init_features, kernel_size=3, stride=1,
                                padding=1, bias=False))]))
        
        num_features = self.num_init_features #Note should be 2 times growth rate
        for i, num_layers in enumerate(self.L):
            dBlock = DenseBlock(num_layers,num_features,growth_rate,bottle_neck = self.bottle_neck)
            self.features.add_module('denseblock%d' % (i + 1), dBlock)
            num_features = num_features + num_layers * growth_rate
            
            if i < (len(self.L) - 1):
                trans = Transition_Layer(num_input_features=num_features,
                                    num_output_features= int(num_features // self.compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features // self.compression)
        self.num_features = num_features
        # Final batch norm
        self.features.add_module('norm_Final',nn.BatchNorm2d(self.num_features))
        
        # Linear layer
        self.classifier = nn.Linear(self.num_features, self.num_classes)
        
    def forward(self, x):
        out = self.features(x)
        out = F.relu(out)
        out = torch.squeeze(F.adaptive_avg_pool2d(out, (1, 1)))
        out_classifier = self.classifier(out)
        return out,out_classifier
    
    def save_models(self,epoch,optimizer_dict,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if optimizer_dict is not None:
            state = {
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'optimizer': optimizer_dict,
                }
        else:
            state = {
                'epoch': epoch,
                'state_dict': self.state_dict(),
                }
        torch.save(state,save_dir +'/Model' +str(2*np.sum(self.L)+4) + "_{}.model".format(epoch))
        print("Check Point Saved") 