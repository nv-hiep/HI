# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:17:45 2023

@author: Haiyang
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#Transformer Positional Encoder

    
""" 
    PositionalEncoding: positional encoding method in original natural language processing.
    
    Methods
    -------
    forward(x):
        add positional encoding to x and return x.
"""


class PositionalEncoding(nn.Module):
    # custom code
    def __init__(self,num_features, sequence_len=6, d_model=9):
        super(PositionalEncoding, self).__init__()
        if torch.cuda.is_available():
            self.device='cuda:0'
        else:
            self.device='cpu'
        pe = torch.zeros((1, sequence_len, d_model), dtype=torch.float32).to(self.device)
        factor = -math.log(10000.0) / d_model  # outs loop
        for index in range(0, sequence_len):  # position of word in seq
            for i in range(0, d_model, 2):
                #print("i==",i)
                div_term = math.exp(i * factor)
                pe[0, index, i] = math.sin(index * div_term)
                if(i+1<d_model):
                    pe[0, index, i+1] = math.cos(index * div_term)
                
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x has shape [seq_len, bat_size, embed_dim]
        #print("self.pe[:x.size(0), :]=",self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return x

    
"""
    cnn_transformer: A class of the CNN_transformer model.
    
    input Attributes
    ----------
    num_features : int
        The number of features.
    drop_rate : the drop out rate in transformer.
        input channel number.
    pos_encoder : PositionalEncoding.
        original positional encoder in NLP.
    lpe: boolean
        whether or not add leanable positional embedding to transformer inputs.
    pos_embedding: nn.Parameter(.)
        learnable parameters for positional embedding.
        
    Methods
    -------
    forward(x):
        calculate the outputs.
    
"""
    
class cnn_transformer_small(nn.Module):
    def __init__(self, num_output=2, in_channels=1, input_row = 2, input_column=414, drop_out_rate=0, lpe=False):
        super(cnn_transformer_small, self).__init__()
        
        self.num_features = 54
        self.drop_rate=drop_out_rate
        self.pos_encoder = PositionalEncoding(num_features=self.num_features, sequence_len=6, d_model=9)
        self.lpe=lpe
        self.pos_embedding = nn.Parameter(torch.randn(6, 9))
        
        # CNN layers
        if(input_row>=2):
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=72, 
                               kernel_size=(2,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
        else:
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=72, 
                               kernel_size=(1,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
            
        self.bn1 = nn.BatchNorm2d(72)
        
        self.conv2 = nn.Conv2d(in_channels= 72, out_channels=64,
                               kernel_size=(1,10), stride=1, padding=0, bias=True,
                               padding_mode='zeros')
        
        self.bn2 = nn.BatchNorm2d(64)
            
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels=56,
                               kernel_size=(1,3), stride=1, padding=0, bias=True, 
                               padding_mode='zeros')
        
        self.bn3 = nn.BatchNorm2d(56)
        
        self.conv4 = nn.Conv2d(in_channels= 56, out_channels=48, 
                               kernel_size=(1,10), stride=1, padding=0, bias=True,
                               padding_mode='zeros')
        
        self.bn4 = nn.BatchNorm2d(48)
        
        self.conv5 = nn.Conv2d(in_channels = 48, out_channels=40, 
                               kernel_size=(1,3), stride=1, padding=0,
                               bias=True, padding_mode='zeros')
        
        self.bn5 = nn.BatchNorm2d(40)
        
        self.conv6 = nn.Conv2d(in_channels= 40, out_channels=32, 
                               kernel_size=(1,10), stride=1,
                               padding=0, bias=True, padding_mode='zeros')
        
        self.bn6 = nn.BatchNorm2d(32)
        
        self.conv7 = nn.Conv2d(in_channels = 32, out_channels=16, 
                               kernel_size=(1,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
        
        self.bn7 = nn.BatchNorm2d(16)
        
        self.conv8 = nn.Conv2d(in_channels= 16, out_channels=8, 
                               kernel_size=(1,10), stride=1, 
                               padding=0, bias=True, padding_mode='zeros')
        
        self.bn8 = nn.BatchNorm2d(8)
        if(input_row<=2):
            self.linear = nn.Linear(456, 54)
        else:
            self.linear = nn.Linear(int(1904*(input_row-1)), 54)
            
        self.flatten = nn.Flatten()
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=9,
                nhead=3,
                dim_feedforward=36,
                dropout=self.drop_rate,
                batch_first=True,
            ),
            num_layers=4
        )
        
        self.decoder = nn.Linear(54, num_output)

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
        #print(1, x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        #print(2, x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        #print(3, x.size())
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        #
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        #
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        #
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        #
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        #
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        #x = torch.squeeze(x)
        #x = x.reshape(-1, x.shape[2], x.shape[1])
        #
        #print(3, x.size()) #= (20,8,1,238)
        x = self.flatten(x) #1904
        x = self.linear(x)
        #print(3, x.size())
        x = x.reshape(x.shape[0], -1, 9)
        #print(3, x.size())
        if(self.lpe==True):
            x = x + self.pos_embedding
        # Transformer MODEL
        x = self.transformer(x)
        x = self.flatten(x)
        #print(5, x.size())
    
        x = self.decoder(x)
        return x