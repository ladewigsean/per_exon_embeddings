from scripts.alibi import ALiBiConfig, ALiBiTransformer
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder


#should move these into TransformerClassifier if they work
#should add earlier
class PaddingMask(nn.Module):
    """
        Generate padding mask based on lengths tensor
        expects max_length as input
    """
    def __init__(self,max_length):
        super().__init__()
        self.max_length = max_length
        self.register_buffer("max_idx",torch.arange(self.max_length,)[None, :])
    def forward(self,lengths,seq_length=None):
        idx = self.max_idx
        if seq_length is not None:
            idx = idx[:,:seq_length]
        
        return idx >= lengths[:, None]   # bool [B,S], True at pad  
class CovariancePooling(nn.Module):
    """
        CovariancePooling
        
    """
    def __init__(self,dc,embed_size = 1024,):
        super().__init__()
        self.L = nn.Parameter(torch.rand((embed_size,dc))) # rand init with loss backpropegation 
        self.R = nn.Parameter(torch.rand((embed_size,dc)))# rand init with loss backpropegation 
        self.output_dim = dc * dc
    #dont know 
    def forward(self,x,padding_mask):
       
       #x = [batch_size, max_length, embed_size]
       left = torch.matmul(x,self.L)# [batch_size, embed_size, dc]
       left = torch.transpose(left, 1,2)# [batch_size, dc, embed_size]
       right = torch.matmul(x,self.R)# [batch_size, embed_size, dc]
       output = torch.matmul(left,right)# [batch_size, dc, dc]
       
       output = output.flatten(1) # [batch_size, dc^2]
       #print(output.shape)
       valid = (~padding_mask).unsqueeze(-1) #[batch_size, L] 
       denom = valid.sum(dim=1).clamp(min=1) #[batch_size,1]
       output = output / denom # [batch_size, dc^2]
       return output
#masked mean
class MaskedMeanPooling(nn.Module):
    def __init__(self,):
        super().__init__()
        self.output_dim = -1
    def forward(self,x,padding_mask):
        valid = (~padding_mask).unsqueeze(-1)          # [B,S,1]
        x_sum = (x * valid).sum(dim=1)                # [B,E]
        denom = valid.sum(dim=1).clamp(min=1)         # [B,1]
    
        x = x_sum / denom
        return x

    





#this is a simple neural network to test
class NominalClassifier(nn.Module):
    def __init__(self,num_classes, embed_size=1024, hidden_dim1=512,  dropout_rate=0.4,
                    max_length = 5000):
        """
        Simple neural network for Nominal classification. 

        Args:
            embed_size: Dimension of protein embeddings
            hidden_dim1: Dimension of hidden layer
            dropout: Dropout probability
            num_classes: output_dim
        """
        super().__init__()
        print(dropout_rate)
        self.network = nn.Sequential(
            #nn.Dropout(dropout_rate),
            nn.Linear(embed_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim1),

            nn.Linear(hidden_dim1, num_classes),
            
        )

    def forward(self, x,lengths):
        #is expecting [batch,1,embed_size] for other models, needs [batch,embed_size]
        x = x[:,-1,:]
        return self.network(x)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017).

    Computed and stored in float32 to preserve precision when added to
    embeddings. The `factor` parameter scales the encoding magnitude.
    """

    def __init__(self, d_model, max_length=5000, dropout=0.1, factor=1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.factor = factor

        # Always compute in float32 for numerical stability
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_length, d_model]
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :] * self.factor
        return self.dropout(x)
#learned pos encodings from here
#https://github.com/johnrobinsn/blog_notebooks/blob/main/02_learned_embeddings.ipynb
#very simple just uses an nn.embedding 
class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings as used in BERT and GPT-2.
    
    Each position (0, 1, 2, ..., max_len-1) has its own trainable 
    embedding vector of dimension d_model.
    
    Args:
        d_model: Dimension of the embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # The key component: a learnable embedding table
        # Shape: (max_len, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Register position indices as a buffer (not a parameter)
        # This avoids creating new tensors on every forward pass
        self.register_buffer(
            'position_ids', 
            torch.arange(max_len).unsqueeze(0)  # Shape: (1, max_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional embeddings added
        """
        seq_len = x.size(1)
        # Get position IDs 
        position_ids = self.position_ids[:, :seq_len] # Shape: (1, seq_len)
        
        # Look up position embeddings
        position_embeds = self.position_embeddings(position_ids)  # (1, seq_len, d_model)
        
        # Add to input (broadcasts across batch dimension)
        x = x + position_embeds
        
        return self.dropout(x)






#added transformer elements following this as base
#https://www.youtube.com/watch?v=9V4xgt3Vs8A
#dont nesc understand what nhead really means 
#removed option for multiple hidden layers, thought it too much
#has gone threw a lot of testing figuring out why PE has such negative effects
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, embed_size=1024, hidden_dim1=512,  dropout_rate=0.4,
                max_length = 5000, dim_feedforward = 2048 ,nhead=4,num_layers_transformer = 1,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),use_alibi = False,pe_factor=1,pe_mode = "pe"):
        super().__init__()
        print(dropout_rate)
        self.max_len = max_length
        self.device = device
        self.embed_size = embed_size
        self.pooling = MaskedMeanPooling()
        self.padding = PaddingMask(max_length=max_length)
        if pe_mode == "pe":
            #self.position_encoder = PositionalEncoding(d_model=embed_size,dropout=dropout_rate,
            #                                       max_length=self.max_len,factor=pe_factor)
            self.position_encoder = PositionalEncoding(d_model=embed_size,dropout=dropout_rate,
                                                   max_length=self.max_len,factor=pe_factor)
        if pe_mode =="learned_pe":
            self.position_encoder = LearnedPositionalEmbedding(embed_size,max_len=max_length,dropout=dropout_rate)
        
        if use_alibi:
            #https://github.com/jaketae/alibi
            config = ALiBiConfig(num_layers=num_layers_transformer,d_model = embed_size,
                                 num_heads = nhead,max_len=max_length,dropout=dropout_rate,causal = False)
            self.transformer_encoder = ALiBiTransformer(config,device=self.device)
        else:
            transformer_layer = TransformerEncoderLayer(
                d_model=embed_size,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout_rate,batch_first=True)
            self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=num_layers_transformer)
        self.use_alibi = use_alibi
        
        self.network = nn.Sequential(
            nn.Linear(embed_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, num_classes),
        )#
        
    def forward(self, x,lengths):
        #truncate padding per batch, testing if this helps
        
        seq_len = x.size(1)
        #padding mask[batch,max_length] of 0s at exons and -inf at padding
        
        #bool padding mask
        padding_mask = self.padding(lengths, seq_len)
        #print(padding_mask.shape) --> torch.Size([16, 28])
        
        
        
        #"""
        #position encoder +[1,max_length,embed_size]
        x = self.position_encoder(x)

        #[batch,max_length,embed_size]
        
        if self.use_alibi:
            x = self.transformer_encoder(x, padding_mask=padding_mask)
        else: 
            x = self.transformer_encoder(x,src_key_padding_mask=padding_mask)
        #"""
        #output from [batch,max_length,embed_size] --> [batch,embed_size]
        x = self.pooling(x, padding_mask)
        
        #[batch,embed_size] --> [batch,number_classes]
        return self.network(x) 

class PoolingClassifier(nn.Module):
    def __init__(self, num_classes, embed_size=1024, hidden_dim1=512,  dropout_rate=0.4,
                max_length = 5000,dc= 4):
        super().__init__()
        self.max_len = max_length
        self.padding = PaddingMask(max_length=max_length)
        self.pooling = CovariancePooling(dc,embed_size=embed_size)
        pooling_output_dim = self.pooling.output_dim if self.pooling.output_dim != -1 else embed_size
        self.network = nn.Sequential(
            nn.Linear(pooling_output_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, num_classes),
        )
    def forward(self,x,lengths):
        #input x [batch_size,max_length,embed_size]

        seq_len = x.size(1)
        
        padding_mask = self.padding(lengths,seq_len)
        #[batch_size, embed_size]
        x = self.pooling(x,padding_mask)
        return self.network(x)