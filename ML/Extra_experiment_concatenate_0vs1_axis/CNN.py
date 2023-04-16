# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:17:45 2023

@author: Haiyang
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

    
"""
    CNN: A class of the CNN model.
    
    input Attributes
    ----------
    drop_out_rate : int
        the drop out rate in last layer of CNN.
    lpe: boolean
        whether or not add leanable positional embedding to CNN inputs.
    num_layer : int
        number of layers of CNN.
    num_output: int
        number of output
    in_channels: int
        number of input channels
    input_row : int 
        number of input row
    input_column: int
        number of input column
    
    Methods
    -------
    forward(x):
        calculate the outputs.
    
"""
    
class spectra_cnn(nn.Module):
    def __init__(self, num_output=2, in_channels=1, input_row = 2, input_column=414, num_layer = 8, drop_out_rate=0.30, lpe=False):
        super(spectra_cnn, self).__init__()
        self.in_channels = in_channels
        self.num_layer = num_layer
        self.num_features = in_channels*input_row*input_column
        self.drop_rate=drop_out_rate
        self.lpe=lpe
        self.out_channels_1 = num_layer*8+8
        self.dropout = nn.Dropout(drop_out_rate)
        self.pos_embedding = nn.Parameter(torch.randn(in_channels, input_row, input_column))
        # head of CNN
        if(input_row>=2):
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=self.out_channels_1, 
                               kernel_size=(2,6), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
        else:
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=self.out_channels_1, 
                               kernel_size=(1,6), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(self.out_channels_1)
        
        self.conv2 = nn.Conv2d(in_channels= self.out_channels_1, out_channels=self.out_channels_1-8,
                               kernel_size=(1,40), stride=1, padding=0, bias=True,
                               padding_mode='zeros')
        
        self.bn2 = nn.BatchNorm2d(self.out_channels_1-8)
        # add layers 
        self.layers = nn.ModuleList()
        self.layers.append(self.conv1)
        self.layers.append(self.bn1)
        self.layers.append(nn.ReLU())
        
        self.layers.append(self.conv2)
        self.layers.append(self.bn2)
        self.layers.append(nn.ReLU())
        count = self.out_channels_1-8
        if(num_layer>=4):
            for i in range(0, int((num_layer-4)/2)):
                #1
                self.layers.append(nn.Conv2d(in_channels= count, out_channels=count-8,
                               kernel_size=(1,6), stride=1, padding=0, bias=True,
                               padding_mode='zeros'))
                self.layers.append(nn.BatchNorm2d(count-8))
                self.layers.append(nn.ReLU())
                # 2
                self.layers.append(nn.Conv2d(in_channels= count-8, out_channels=count-16,
                               kernel_size=(1,40), stride=1, padding=0, bias=True,
                               padding_mode='zeros'))
                self.layers.append(nn.BatchNorm2d(count-16))
                self.layers.append(nn.ReLU())
                count = count-16
        ### count = 32
        if(num_layer>=4):
            self.layers.append(nn.Conv2d(in_channels = count, out_channels=16, 
                               kernel_size=(1,6), stride=1, padding=0, 
                               bias=True, padding_mode='zeros'))
        
            self.layers.append(nn.BatchNorm2d(16))
            self.layers.append(nn.ReLU())
        
            self.layers.append(nn.Conv2d(in_channels= 16, out_channels=8, 
                               kernel_size=(1,40), stride=1, 
                               padding=0, bias=True, padding_mode='zeros'))
            self.layers.append(nn.BatchNorm2d(8))
            self.layers.append(nn.ReLU())
        
        ###
        self.layers.append(nn.Flatten())
        self.layers.append(self.dropout)
        
        if(num_layer==2):
            self.linear = nn.Linear(5920, num_output)
        else:
            input_channels = 2608 - 352 * int((num_layer-4)/2)
            self.linear = nn.Linear(input_channels, num_output)
        
        # init parameter
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if(self.lpe==True):
            x = x + self.pos_embedding
        for layer in self.layers[:-1]:
            #print('layer=', layer)
            x = layer(x)
        x = self.linear(x)
        return x
    
    
    
    
    
