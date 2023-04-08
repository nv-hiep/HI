# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 23:07:39 2023

@author: Administrator-1
"""
import torch 
import torch.nn as nn
import math

    
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
                div_term = math.exp(i * factor)
                pe[0, index, i] = math.sin(index * div_term)
                if(i+1<d_model):
                    pe[0, index, i+1] = math.cos(index * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

    
""" 
    Transformer: Transformer class.

    Attributes
    ----------
    num_features : int
        number of features.
    d_m : int
        the embedding size of each token.
    sequence_len : int
        Length of the sequence.
    pos_encoder : PositionalEncoding
        The original positional encoding methods in natural language processing.
    ipe: boolean
        whether or not add the positional encoding.
    transformer: transformer.
        transformer that conatins few attention layers.

    Methods
    -------
    forward(x):
        calculate the outputs.
"""


class Transformer(nn.Module):
    def __init__(self, num_output=2, in_channels=1, input_row=1, input_column=414, ipe=False):
        super(Transformer, self).__init__()
        self.num_features = in_channels*input_row*input_column
        self.d_m = 9
        self.sequence_len = int ((in_channels*input_row*input_column)/ self.d_m)
        self.pos_encoder = PositionalEncoding(num_features=self.num_features, sequence_len=self.sequence_len, d_model=self.d_m)
        self.ipe=ipe
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_m,
                nhead=3,
                dim_feedforward=40,
                dropout=0,
                batch_first=True,
            ),
            num_layers=4
        )
        
        self.flatten = nn.Flatten()
        self.decoder = nn.Linear(self.num_features, num_output)
    def forward(self, x):
        #print("1=",x.shape)
        
        x = torch.reshape(x, (x.shape[0],-1,self.d_m))
        if(self.ipe==True):
            x = self.pos_encoder(x)
        #print("2=",x.shape)
        x = self.transformer(x)
        #print("3=",x.shape)
        x = self.flatten(x) #1904
        #print("4=",x.shape)
        x = self.decoder(x)
        return x