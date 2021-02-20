# -*- coding: utf-8 -*-
# SimpleNetV1(2016)
# Implementation of https://arxiv.org/abs/1608.06037
# Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os


class _ConvLayer(nn.Module):
    def __init__(self,num_input_features,num_output_features):
        super(_ConvLayer,self).__init__()
        #self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size = 3, stride = 1, padding = 1, bias=False))
        #self.add_module('norm', nn.BatchNorm2d(num_output_features, eps=1e-05, momentum=0.05, affine=True,track_running_stats = True))
        #self.add_module('relu', nn.ReLU(inplace=True))
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size = 3, stride = 1, padding = 1, bias=False)
        self.norm =  nn.BatchNorm2d(num_output_features, eps=1e-05, momentum=0.05, affine=True,track_running_stats = True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x_in):
        x_out = self.relu(self.norm(self.conv(x_in)))
        return x_out

class _ConvBlock(nn.Module):
    def __init__(self,num_layers,num_input_features,num_output_features):
        super(_ConvBlock,self).__init__()
        self.convBlock_features = nn.ModuleDict()
        for l in range(num_layers):
            conv_layer = _ConvLayer(num_input_features[l],num_output_features[l])
            self.convBlock_features['convlayer%d' % (l + 1)] =  conv_layer
            
    def forward(self,x_in):
        x_out = x_in
        for name, layer in self.convBlock_features.items():
            x_out = layer(x_out)
        return x_out

class SimpleNet(nn.Module):
    def __init__(self,num_classes,num_input_channels):
        super(SimpleNet,self).__init__()
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        
        self.features = nn.Sequential(OrderedDict([]))
        self._makeLayers()
        self.dropOutFinal = nn.Dropout(0.1)
        
        self.classifier = nn.Linear(256, num_classes)
        
        
        
        
    def _makeLayers(self):
        num_input_features = [self.num_input_channels,64,128,128]
        num_output_features = [64,128,128,128]
        num_layers = 4
        conv_block = _ConvBlock(num_layers,num_input_features,num_output_features)
        self.features.add_module('convBlock1',conv_block)
        self.features.add_module('pool1', nn.MaxPool2d(kernel_size= 2, stride= 2, dilation= 1, ceil_mode = False))
        self.features.add_module('dropOut1',nn.Dropout2d(p=0.1))
        
        
        num_input_features = [128,128,128]
        num_output_features = [128,128,256]
        num_layers = 3
        conv_block2 = _ConvBlock(num_layers,num_input_features,num_output_features)
        self.features.add_module('convBlock2',conv_block2)
        self.features.add_module('pool2', nn.MaxPool2d(kernel_size = 2, stride = 2, dilation = 1, ceil_mode = False))
        self.features.add_module('dropOut2',nn.Dropout2d(p=0.1))
        
        
        num_input_features = [256,256]
        num_output_features = [256,256]
        num_layers = 2
        conv_block3 = _ConvBlock(num_layers,num_input_features,num_output_features)
        self.features.add_module('convBlock3',conv_block3)
        self.features.add_module('pool3', nn.MaxPool2d(kernel_size = 2, stride = 2, dilation = 1, ceil_mode = False))
        self.features.add_module('dropOut3',nn.Dropout2d(p=0.1))
        
        
        num_input_features = [256]
        num_output_features = [512]
        num_layers = 1
        conv_block4 = _ConvBlock(num_layers,num_input_features,num_output_features)
        self.features.add_module('convBlock4',conv_block4)
        self.features.add_module('pool4', nn.MaxPool2d(kernel_size = 2, stride = 2, dilation = 1, ceil_mode = False))
        self.features.add_module('dropOut4',nn.Dropout2d(p=0.1))
        
        
        num_input_features = [512,2048]
        num_output_features = [2048,256]
        num_layers = 2
        conv_block5 = _ConvBlock(num_layers,num_input_features,num_output_features)
        self.features.add_module('convBlock5',conv_block5)
        self.features.add_module('pool5', nn.MaxPool2d(kernel_size = 2, stride = 2, dilation = 1, ceil_mode = False))
        self.features.add_module('dropOut5',nn.Dropout2d(p=0.1))
        
        
        num_input_features = [256]
        num_output_features = [256]
        num_layers = 1
        conv_block6 = _ConvBlock(num_layers,num_input_features,num_output_features)
        self.features.add_module('convBlock6',conv_block6)
        
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
              nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    
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
        torch.save(state,save_dir +'/SimpleNetModel' + "_{}.model".format(epoch))
        print("Check Point Saved") 
        
    def forward(self,x_in):
        x_out = self.features(x_in)
        #Global Max Pooling
        x_out = F.max_pool2d(x_out, kernel_size= x_out.size()[2:]) 
        x_out = self.dropOutFinal(x_out)

        x_out = x_out.view(x_out.size(0), -1)
        class_out = self.classifier(x_out)
        return x_out, class_out
        
        
        
        
       