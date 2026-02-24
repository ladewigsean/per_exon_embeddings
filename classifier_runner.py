#!/usr/bin/env python3
import sys
import os

from alibi import ALiBiConfig, ALiBiTransformer

import argparse


import math
import pandas as pd
from pathlib import Path
import h5py
import numpy as np
from collections import Counter
import yaml
#from positional_encodings.torch_encodings import PositionalEncoding1D,PositionalEncodingPermute1D, Summer
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from lion_pytorch import Lion
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#import seaborn as sns
#import matplotlib.pyplot as plt
#import inspect
import wandb
import optuna
from tqdm import tqdm

from torch.amp import GradScaler, autocast



class MultiClassDataset(Dataset):
    def __init__(self, embeddings_path: str, csv_path: str, label_encoder: LabelEncoder = None):
        super().__init__()
        self.embeddings_file_path = embeddings_path
        self.metadata_df = pd.read_csv(csv_path, dtype={'identifier': str})
        self.id_column = 'identifier'
        self.gene_column = 'gene'
        self.h5f = None
        self.max_length = None

        if self.id_column not in self.metadata_df.columns or self.gene_column not in self.metadata_df.columns:
            raise ValueError(f"Required columns 'identifier' or 'gene' not found in {csv_path}.")
        
        if label_encoder is None:
            self.label_encoder = OneHotEncoder().fit(np.array(self.metadata_df[self.gene_column]).reshape(-1, 1))
            self.encodings = self.label_encoder.transform(np.array(self.metadata_df[self.gene_column]).reshape(-1, 1)).toarray()
            self.metadata_df['label'] = self.encodings.tolist()
            print(f"INFO: Fitted new LabelEncoder. Found {len(self.label_encoder.categories_[0])} classes.")
        else:
            self.label_encoder = label_encoder
            known_genes = self.label_encoder.categories_[0]
            self.metadata_df = self.metadata_df[self.metadata_df[self.gene_column].isin(known_genes)]
            self.encodings = self.label_encoder.transform(np.array(self.metadata_df[self.gene_column]).reshape(-1, 1)).toarray()
            self.metadata_df['label'] = self.encodings.tolist()
        self.num_classes = len(self.label_encoder.categories_[0])
        try:
            with h5py.File(self.embeddings_file_path, 'r') as h5f:
                embedding_ids = set(h5f.keys())
            original_len = len(self.metadata_df)
            self.metadata_df = self.metadata_df[self.metadata_df[self.id_column].isin(embedding_ids)].reset_index(drop=True)
            if original_len != len(self.metadata_df):
                print(f"INFO: Filtered metadata for HDF5 keys. Kept {len(self.metadata_df)}/{original_len} entries.")
        except Exception as e:
            print(f"ERROR: Could not filter dataset for HDF5: {e}")
            self.metadata_df = pd.DataFrame()
        
        with h5py.File(self.embeddings_file_path, 'r') as h5f:
            sample_key = list(h5f.keys())[0]
            self.embedding_dim = h5f[sample_key].shape[-1]
        self.get_max_length()
    
        

    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.embeddings_file_path, 'r')

        row = self.metadata_df.iloc[index]
        embedding = torch.tensor(self.h5f[str(row[self.id_column])][:], dtype=torch.float16)
        
        if self.max_length >1:
            og_size = embedding.shape[0]
            data_pad_len = self.max_length - og_size
            embedding = torch.cat([embedding,torch.zeros(data_pad_len,1024,dtype = torch.float16)])
        else:
            og_size = 1
            embedding = embedding.unsqueeze(0)
        return embedding, torch.tensor(row['label']), str(row[self.id_column]),og_size

    def __len__(self):
        return len(self.metadata_df)
    def get_max_length(self):
        if self.max_length == None:
            if self.h5f is None:
                self.h5f = h5py.File(self.embeddings_file_path, 'r')
            self.max_length = 1
            for key in self.h5f.keys():
                shape = self.h5f[key][:].shape
                if len(shape) == 1:
                    self.max_length == 1
                    break
                else:
                    if shape[0] > self.max_length:
                        self.max_length = shape[0]
        return self.max_length 

    def get_labels(self):
        return self.metadata_df['label'].values
    def get_data(self):
        return self.metadata_df
#omega scuffed but whatever, keeps it easier
class MultiClassSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.data = dataset.get_data().iloc[indices]
        self.num_classes = dataset.num_classes
        self.label_encoder = dataset.label_encoder
        self.embeddings_file_path = dataset.embeddings_file_path
        self.id_column = dataset.id_column
        self.h5f = None
    """
    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.embeddings_file_path, 'r')

        row = self.data.iloc[index]
        embedding = torch.tensor(self.h5f[str(row[self.id_column])][:], dtype=torch.float16)
        return embedding, int(row['label']), str(row[self.id_column])
    def __len__(self):
        return len(self.data)
    """
    def get_labels(self):
        return self.data['label'].values
    def get_data(self):
        return self.data
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

        self.network = nn.Sequential(
            nn.Linear(embed_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim1),

            nn.Linear(hidden_dim1, num_classes),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        #is expecting [batch,1,embed_size] for other models, needs [batch,embed_size]
        x = x[:,-1,:]
        return self.network(x)
"""
# postional came mostly from here, most things I read reccomended one directly from pytorch tutorial, but link never worked 
#https://colab.research.google.com/github/maqboolkhan/Transformer_classifier_pytorch/blob/master/Transformer_Classification_Pytorch.ipynb#scrollTo=26Mbt2jzpK7y
#tbh still dont understand 100% everything yet
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_length = 5000):
        super(PositionalEncoding, self).__init__()
        
        
        # Shape (pos) --> [max len, 1]
        pos = torch.arange(0, max_length).unsqueeze(1)
        pos_encoding = torch.zeros((max_length, d_model))

        sin_den = 10000 ** (torch.arange(0, d_model, 2)/d_model) # sin for even item of position's dimension
        cos_den = 10000 ** (torch.arange(1, d_model, 2)/d_model) # cos for odd 

        pos_encoding[:, 0::2] = torch.sin(pos / sin_den) 
        pos_encoding[:, 1::2] = torch.cos(pos / cos_den)

        # Shape (pos_embedding) --> [max len, d_model]
        # Adding one more dimension in-between
        pos_encoding = pos_encoding.unsqueeze(-2)
        # Shape (pos_embedding) --> [max len, 1, d_model]
        #batch size first so switch. prob can change the unqsqueeze but not 100% sure how(prob just -1 tbh, TODO)
        pos_encoding = torch.swapaxes(pos_encoding,0,1)
        self.dropout = nn.Dropout(dropout)
        # We want pos_encoding be saved and restored in the `state_dict`, but not trained by the optimizer
        # hence registering it!
        # Source & credits: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/2
        self.register_buffer('pos_encoding', pos_encoding)
    def forward(self, token_embedding):
        # shape (token_embedding) --> [batch size,sentence len, d_model]

        # Concatenating embeddings with positional encodings
        # Note: As we made positional encoding with the size max length of sentence in our dataset 
        #       hence here we are picking till the sentence length in a batch
        #       Another thing to notice is in the Transformer's paper they used FIXED positional encoding, 
        #       there are methods where we can also learn them
        return self.dropout(token_embedding + self.pos_encoding[:,:token_embedding.size(1)])
"""
class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, max_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, d_model,dtype=torch.float16)
        position = torch.arange(0, max_length, dtype=torch.float16).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2,dtype=torch.float16)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        #print(pe.shape) --> torch.Size([1, 28, 1024])
        
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
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
        # Get position IDs 
        position_ids = self.position_ids # Shape: (1, seq_len)
        
        # Look up position embeddings
        position_embeds = self.position_embeddings(position_ids)  # (1, seq_len, d_model)
        
        # Add to input (broadcasts across batch dimension)
        x = x + position_embeds
        
        return self.dropout(x)







#should move these into TransformerClassifier if they work
def constrained_mean(x,lengths,device,dtype= torch.float16):
    #def probably exists one line solution but whatever
    embed_size = x.shape[2]
    #batch_size = x.shape[0]
    output = torch.empty((0,embed_size),dtype=dtype,device=device)
    for y in range(len(lengths)):
        length = lengths[y]
        mean_embed = x[y,:length,:].mean(dim=0)
        output = torch.cat([output,mean_embed.unsqueeze(0)])
    return output
#does same as above but actually looks like code lol , from ivan
def masked_mean(x,padding_mask):
    valid = (~padding_mask).unsqueeze(-1)          # [B,S,1]
    x_sum = (x * valid).sum(dim=1)                # [B,E]
    denom = valid.sum(dim=1).clamp(min=1)         # [B,1]
    
    x = x_sum / denom
    return x

def gen_pad_mask(max_length,lengths,device,dtype= torch.float16):
    #ok there has to be a better way to do this, this is so scuffed
    padding_mask = torch.empty((0,max_length),dtype=dtype,device=device)
    for y in lengths:
        row = torch.full([max_length],float("-inf"),dtype=dtype,device=device)
        row[:y] = 0
        padding_mask = torch.cat([padding_mask,row.unsqueeze(0)])
    return padding_mask
def gen_pad_mask_bool(max_length,lengths,device):

    """
    padding_mask = torch.empty((0,max_length),dtype=bool,device=device)
    for y in lengths:
        row = torch.full([max_length],True,dtype=bool,device=device)
        row[:y] = False
        padding_mask = torch.cat([padding_mask,row.unsqueeze(0)])
    """
    #above doesnt work but this works, also looks like code, from ivan
    idx = torch.arange(max_length, device=device)[None, :]
    padding_mask = idx >= lengths[:, None]   # bool [B,S], True at pad
    return padding_mask



#added transformer elements following this as base
#https://www.youtube.com/watch?v=9V4xgt3Vs8A
#dont nesc understand what nhead really means 
#removed option for multiple hidden layers, thought it too much
#has gone threw a lot of testing figuring out why PE has such negative effects
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, embed_size=1024, hidden_dim1=512,  dropout_rate=0.4,
                    max_length = 5000, dim_feedforward = 2048 ,nhead=4,num_layers_transformer = 1,device = "cuda",use_alibi = False):
        super().__init__()
        self.max_len = max_length
        self.device = device
        self.embed_size = embed_size
        layers = []
        self.position_encoder = PositionalEncoding(d_model=embed_size,dropout=dropout_rate,max_length=self.max_len)
        #self.position_encoder = LearnedPositionalEmbedding(embed_size,max_len=max_length,dropout=dropout_rate)
        #self.position_encoder = RotaryPositionalEmbeddings(embed_size,max_len=max_length,dropout=dropout_rate)
        if use_alibi:
            #https://github.com/jaketae/alibi
            config = ALiBiConfig(num_layers=num_layers_transformer,d_model = embed_size, num_heads = nhead,max_len=max_length,dropout=dropout_rate,causal = False)
            self.transformer_encoder = ALiBiTransformer(config)
        else:
            transformer_layer = TransformerEncoderLayer(
                d_model=embed_size,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout_rate,batch_first=True)
            self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=num_layers_transformer)
        
        self.conv_layer =nn.Conv1d(self.max_len,1, kernel_size=3, padding = 1)
        layers.append(nn.Linear(embed_size, hidden_dim1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))    
        layers.append(nn.Linear(hidden_dim1, num_classes))
        #layers.append(nn.Softmax(dim=1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x,lengths):
        #padding mask[batch,max_length] of 0s at exons and -inf at padding
        
        #bool padding mask
        padding_mask = gen_pad_mask_bool(self.max_len,lengths,self.device)
        #print(padding_mask.shape) --> torch.Size([16, 28])
        
        #forward mask [max_length,max_length] 
        #src_mask = nn.Transformer.generate_square_subsequent_mask(self.max_len, device=self.device,dtype=torch.float16)

        #position encoder +[1,max_length,embed_size]
        x = self.position_encoder(x)

        #[batch,max_length,embed_size]
        #x = self.transformer_encoder(x,src_mask,src_key_padding_mask=padding_mask)
        x = self.transformer_encoder(x,src_key_padding_mask=padding_mask)
        #output from [batch,max_length,embed_size] --> [batch,embed_size]
        #x = x.mean(dim=1)
        #x = self.conv_layer(x)[:,-1,:]
        #x= constrained_mean(x,lengths,self.device)
        x = masked_mean(x,padding_mask)
        #[batch,embed_size] --> [batch,number_classes]
        return self.network(x)   
#reference from here
#https://www.geeksforgeeks.org/deep-learning/implementing-recurrent-neural-networks-in-pytorch/
class RNNClassifier(nn.Module):    
    def __init__(self, num_classes, embed_size=1024, hidden_dim1=512,  dropout_rate=0.4,
                  activation_function="tanh",  max_length = 5000, device = "cuda"):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim1 = hidden_dim1
        self.device = device
        self.rnn = nn.RNN(embed_size, hidden_dim1, num_layers=max_length,batch_first=True,nonlinearity = activation_function,dropout=dropout_rate)
        self.hiddenlayer = nn.Linear(hidden_dim1,num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        #best guess is that for first run there needs to be initial values
        h0 = torch.zeros(self.max_length, x.size(0), self.hidden_dim1).to(x.device)
        x,_ = self.rnn(x,h0)
        x = self.hiddenlayer(x[:, -1, :])
        x = self.softmax(x)
        return x     
#used prev trainers as outline 
class MultiClassTrainer:
    def __init__(self, model_config, learning_rate, weight_decay, class_weights_tensor, model = "RNN",criterion = "CEL",optimizer = "Adam",scheduler = "Plateau"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"INFO: Using device: {self.device}")
        
        self.model_config = model_config
        #added criterion as hyperparameter cause desperate
        if criterion == "CEL":
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights_tensor.to(self.device) if class_weights_tensor is not None else None
            )
        elif criterion == "CEL_weightless":
            self.criterion = nn.CrossEntropyLoss()
        #wierdly works best despite not being meant for 
        elif criterion == "MSE":
            self.criterion = nn.MSELoss()
        if model == "RNN":
            self.model = RNNClassifier(**self.model_config).to(self.device)
        elif model == "Transformer":
            self.model = TransformerClassifier(**self.model_config).to(self.device)
        elif model == "Basic":
            self.model = NominalClassifier(**self.model_config).to(self.device)
        #leads to some form of tritonmissing error, which isnt windows compatable
        """
        if self.device.type == 'cuda':
            self.model = torch.compile(self.model)
            self.model = self.model.to(self.device.type)
            print("INFO: Model compiled with torch.compile() for speedup.")
        """
        #optimizer
        #added optimizer as hyperparameter cause desperate
        #default
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #seperate weight decay, improvement over Adam
        elif optimizer == "AdamW": 
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #Apperently better AdamW, github here
        #https://github.com/lucidrains/lion-pytorch
        elif optimizer == "Lion": 
            self.optimizer = Lion(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #modify lr during epochs
        self.schedule_type = scheduler
        #lower lr when after no longer learning 
        if scheduler == "Plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.2)
        #lower lr each step by factor of gamma
        elif scheduler == "Exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.8)
        #cyclic cosine warm resets 
        elif scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0 = 8,T_mult = 2, eta_min = learning_rate/10)
        #also warm resets. exp_range is like curved triangler where it converges onto base_lr each cycle
        elif scheduler == "Cyclic":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,base_lr = learning_rate, max_lr = learning_rate*10,step_size_up = 8,mode="exp_range" )
        #no scheduler basicaly, just need a scheduler object, constant has other use
        elif scheduler == "None":
             self.scheduler = torch.optim.lr_scheduler.ConstatnLR(self.optimizer,factor = 1, total_iters=1)
        #needed because float16, i think 
        self.use_amp = (self.device.type == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        

    def train_and_validate(self, train_loader, val_loader, num_epochs, patience, checkpoint_path, 
                      label_encoder, log_to_wandb=False, trial=None, step_offset=0):
        #stop is decided by val macro f1, might change back to val accuracy
        best_val_f1 = 0.0 
        epochs_without_improvement = 0
        last_epoch = 0

        for epoch in range(num_epochs):
            last_epoch = epoch
            #train
            self.model.train()
            train_loss, train_acc = self._run_epoch(train_loader, training=True)

            # Validation phase: get loss and full report
            self.model.eval()
            with torch.no_grad():
                val_report, _, _, _ = self.evaluate_on_loader(val_loader, label_encoder)
            
            # extract the F1 score
            current_val_f1 = val_report['macro avg']['f1-score'] 
            current_val_acc = val_report['accuracy'] * 100  #for logging
            
            if self.schedule_type == "Plateau":
                self.scheduler.step(1 - current_val_f1) # Schedule on 1 - F1 score, this is just used to see whether model has stalled
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val macro avg F1: {current_val_f1:.4f}, Val Acc: {current_val_acc:.2f}% | LR: {current_lr:.2e}")
            #report to wandb
            if log_to_wandb:
                wandb.log({
                    "epoch": epoch, 
                    "train_loss": train_loss, 
                    "train_accuracy": train_acc,
                    "val_f1_macro": current_val_f1, 
                    "val_accuracy": current_val_acc,
                    "learning_rate": current_lr
                })

            # checkpointing based on best F1 score, might change
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'val_f1_macro': current_val_f1,
                    'label_encoder_classes': list(label_encoder.categories_[0])
                }, checkpoint_path)
                print(f"  ✓ Saved best model (val_f1: {current_val_f1:.4f})")
                epochs_without_improvement = 0
            #early stop
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'INFO: Early stopping at epoch {epoch+1}. Best val_f1: {best_val_f1:.4f}')
                    break
            
            if trial:
                trial.report(current_val_f1, epoch + step_offset)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()   
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"INFO: Loaded best model from checkpoint (val_f1: {checkpoint.get('val_f1_weighted', 0):.4f})")

        val_metrics, _, _,_ = self.evaluate_on_loader(val_loader, label_encoder)
        
        return val_metrics, last_epoch + 1
    #default _run_epoch function, still includes training = false as option but isnt used
    def _run_epoch(self, dataloader, training=False):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for embeddings, labels, _,lengths in tqdm(dataloader):
            embeddings = embeddings.to(self.device)
            labels = labels.float().to(self.device)
            lengths = lengths.to(self.device)
            if training:
                self.optimizer.zero_grad()
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(embeddings,lengths)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else: # This block is no longer used for validation loss calculation in train_and_validate
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(embeddings,lengths)
                    loss = self.criterion(outputs, labels)
            
            predictions = (torch.argmax(outputs, axis=1)).float()
            actual = (torch.argmax(labels, axis=1)).float()
            total_correct += (predictions == actual).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = (100 * total_correct / total_samples) if total_samples > 0 else 0
        return avg_loss, accuracy
    #idk seperate function for validation to make report and works as easy function for eval in other cases
    def evaluate_on_loader(self, data_loader, label_encoder):
        self.model.eval()
        all_labels = []
        all_preds = []
        all_ids = []
        
        with torch.no_grad():
            for embeddings, labels, ids_batch,lengths in data_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.float().to(self.device)
                lengths = lengths.to(self.device)                
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(embeddings,lengths)
                predictions = (torch.argmax(outputs, axis=1)).float()
                actual = (torch.argmax(labels, axis=1)).float()
                all_labels.extend(actual.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())
                all_ids.extend(list(ids_batch))

        report = classification_report(
            all_labels, all_preds, 
            target_names=label_encoder.categories_[0], 
            output_dict=True, 
            zero_division=0
        )
        #
        return report,  all_preds, all_ids, all_labels
def run_hpo_mode(train_dataset,wandb_project,wandb_entity,hpo_metric="weighted avg",n_trials = 50,num_epochs = 100 ,wandb_disable=False,k_folds = 5,max_length = 5000,nn_model = "RNN",random_seed = 42,embed_size=1024,patience = 10):
    #to give wandb params, maybe there is easier way
    args = {
        "hpo_metric":hpo_metric,
        "embed_size":embed_size,
        "n_trials":  n_trials,
        "k_folds":  k_folds,
        "max_length":  max_length,
        "nn_model":  nn_model,
        "random_seed":  random_seed,
        "num_epochs": num_epochs,
        "patience": patience,
        "dropout_rate": 0.2,
        'learning_rate': 1e-4,
        'weight_decay':1e-4,
        'hidden_dim1': 512,
        "criterion": "MSE",
        "optimizer":"Lion",
        'batch_size': 16
    } 
    print(f"\n{'='*60}")
    print(f"Starting HPO: {n_trials} trials, optimizing '{hpo_metric}'")
    print(f"{'='*60}\n")

    def objective(trial):
        run = wandb.init(
            project=wandb_project, 
            entity=wandb_entity, 
            config=args, 
            reinit='return_previous', 
            mode="disabled" if wandb_disable else "online",
            name=f"trial_{trial.number}"
        )
        
        #"""
        trial_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.6),
            'hidden_dim1': trial.suggest_categorical('hidden_dim1', [256, 512, 768]),
            #'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            #"optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Lion"]),
            "optimizer": trial.suggest_categorical("optimizer", [ "AdamW", "Lion"]),
            "criterion": trial.suggest_categorical("criterion", ["CEL","MSE", "CEL_weightless"]),#
            "scheduler": trial.suggest_categorical("scheduler", ["Plateau","Exponential", "CosineAnnealingWarmRestarts","Cyclic","None"])
            
        }
        if nn_model == "Transformer":
            trial_params["nhead"] = trial.suggest_categorical("nhead", [2, 4, 8])#,8
            trial_params["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [1024,1536, 2048 ])#, 4096
            trial_params["num_layers_transformer"] = trial.suggest_categorical("num_layers_transformer", [1,2,3])#, 4

        wandb.config.update(trial_params, allow_val_change=True)
        #"""
        
        cfg = wandb.config

        # Prepare stratified k-fold validation
        labels = np.argmax(train_dataset.encodings, axis=1)
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        fold_metrics = []
        
        # Calculate class weights
        class_counts = Counter(labels)
        weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(train_dataset.num_classes)], dtype=torch.float)
        weights = weights / weights.sum() * len(weights)  # Normalize weights
        
        model_config = {
            'num_classes': train_dataset.num_classes, 
            'embed_size': cfg.embed_size, 
            'hidden_dim1': cfg.hidden_dim1, 
            'dropout_rate': cfg.dropout_rate, 
            "max_length" : max_length
            
        }
        if nn_model == "Transformer":
            model_config["nhead"] = cfg.nhead
            model_config["dim_feedforward"] = cfg.dim_feedforward
            model_config["num_layers_transformer"] = cfg.num_layers_transformer
        global_step = 0
        print(f"Optimizer: {cfg.optimizer}\nCriterion: {cfg.criterion}")
        #as is now, does kfold k times for each tuning step, dont know if this is right or if it should cycle through kfold once each step
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)),labels )):
            print(f"\n--- Trial {trial.number}, Fold {fold+1}/{k_folds} ---")

            # Create data loaders
            train_loader = DataLoader(
                Subset(train_dataset, train_idx),
                batch_size=cfg.batch_size,
                
                shuffle=True 
            )
            val_loader = DataLoader(
                Subset(train_dataset, val_idx),
                batch_size=cfg.batch_size,
                
                shuffle=False
            )
            #adding this to loaders causes pickle issue with h5py object(not 100% sure why its trying to pickle h5) will try to fix later when i have bigger dataset but for now training time is managable
            """
                num_workers=8,
                pin_memory=True,
                persistent_workers=False
            """
            #
            # Train model
            trainer = MultiClassTrainer(model_config, cfg.learning_rate, cfg.weight_decay, weights,model=nn_model,optimizer=cfg.optimizer,criterion=cfg.criterion)
            checkpoint = f"temp_trial_{trial.number}_{wandb_project}_fold_{fold}.pt"

            # Pass step_offset and update it
            val_metrics, epochs_ran = trainer.train_and_validate(
                train_loader, val_loader,
                cfg.num_epochs, cfg.patience,
                checkpoint, train_dataset.label_encoder,
                log_to_wandb=True,
                trial=trial,
                step_offset=global_step # Pass the current global step
            )
            global_step += epochs_ran # Update the counter for the next fold
            
            metric_key = hpo_metric
            if metric_key in val_metrics and isinstance(val_metrics[metric_key], dict):
                # handles 'weighted avg', 'macro avg', or a specific class name by getting its f1-score
                metric_value = val_metrics[metric_key].get('f1-score', 0)
            elif metric_key in val_metrics:
                # handles metrics that are already floats like 'accuracy'
                metric_value = val_metrics[metric_key]
            else:
                # fallback if the specified metric doesn't exist
                print(f"WARNING: HPO metric '{metric_key}' not found. Defaulting to weighted avg f1-score.")
                metric_value = val_metrics['weighted avg']['f1-score']

            fold_metrics.append(metric_value)

            # Clean up checkpoint
            if os.path.exists(checkpoint):
                os.remove(checkpoint)
            
        # Calculate average metric across folds
        avg_metric = np.mean(fold_metrics)
        std_metric = np.std(fold_metrics)
        
        wandb.log({
            "avg_cv_metric": avg_metric,
            "std_cv_metric": std_metric,
            "fold_metrics": fold_metrics
        })
        
        print(f"\nTrial {trial.number} CV Result: {avg_metric:.4f} ± {std_metric:.4f}")
        run.finish()
        
        return avg_metric

    # Create and run study
    #not 100% percent sure how this hpo works but it works
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=30)
    )
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n{'='*60}")
    print("✅ HPO Complete")
    print(f"{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best {hpo_metric}: {study.best_value:.4f}")
    
    # Save best hyperparameters
    best_params_path = wandb_project+".yaml"
    with open(best_params_path, 'w') as f:
        yaml.dump(study.best_trial.params, f, default_flow_style=False)
    
    print(f"\n💾 Best hyperparameters saved to '{best_params_path}':")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
def split_dataset_into_subsets(dataset):
    df = dataset.get_data()
    max_length = dataset.get_max_length()
    train_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==0))
    val_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==1))
    test_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==2))
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_dataset, val_dataset, test_dataset, max_length
def run(h5,metadata,wandb_project,wandb_entity,nn_model = "Transformer",num_epochs = 100,patience = 10, kfolds = 5,n_trials = 50, random_seed =42,wandb_disable = False ):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.benchmark = True
        
        
    dataset = MultiClassDataset(embeddings_path=h5,csv_path=metadata)
    max_length = dataset.get_max_length()

    run_hpo_mode(train_dataset=dataset ,wandb_project=wandb_project,wandb_entity=wandb_entity,nn_model =nn_model,max_length=max_length,n_trials=n_trials,num_epochs=num_epochs,patience=patience,wandb_disable=wandb_disable )
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


#personal key lol
#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
if __name__ == '__main__':
    #nn_model = "RNN"
    nn_model = "Transformer"
    #nn_model = "Basic"
    h5,csv = "splits\\per_exon_train.h5","splits\\per_exon_train.csv"
    #h5,csv = "splits\\per_prot_train.h5","splits\\per_prot_train.csv"
    #entity = "v6_transformer_per_exon_pe"
    entity = "testing"
    run(h5,csv,entity,"per-exon-testing",nn_model=nn_model,n_trials=50,num_epochs=150,wandb_disable=True)