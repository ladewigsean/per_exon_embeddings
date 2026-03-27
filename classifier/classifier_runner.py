#!/usr/bin/env python3
import sys
import random
import os
from alibi import ALiBiConfig, ALiBiTransformer
import argparse
import math
import pandas as pd
import h5py
import numpy as np
from collections import Counter
import yaml
#from positional_encodings.torch_encodings import PositionalEncoding1D,PositionalEncodingPermute1D, Summer
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import Dataset, DataLoader, Subset
from lion_pytorch import Lion
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

import wandb
import optuna
from tqdm import tqdm

from torch.amp import GradScaler, autocast



class MultiClassDataset(Dataset):
    def __init__(self, embeddings_path: str, csv_path: str, label_encoder = None):
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
            self.label_encoder = OneHotEncoder(sparse_output=False).fit(
                self.metadata_df[[self.gene_column]].values
            )
        else:
            self.label_encoder = label_encoder
            known_genes = self.label_encoder.categories_[0]
            self.metadata_df = self.metadata_df[self.metadata_df[self.gene_column].isin(known_genes)]

        self.encodings = self.label_encoder.transform(
            self.metadata_df[[self.gene_column]].values
        )
        self.metadata_df = self.metadata_df.copy()
        self.metadata_df["label"] = list(self.encodings)
        self.num_classes = len(self.label_encoder.categories_[0])
        print(f"INFO: Label encoder has {self.num_classes} classes.")
        with h5py.File(self.embeddings_file_path, "r") as h5f:
            embedding_ids = set(h5f.keys())
        original_len = len(self.metadata_df)
        self.metadata_df = self.metadata_df[
            self.metadata_df[self.id_column].isin(embedding_ids)
        ].reset_index(drop=True)
        # Recompute encodings after filtering
        self.encodings = self.label_encoder.transform(
            self.metadata_df[[self.gene_column]].values
        )
        if original_len != len(self.metadata_df):
            print(f"INFO: Filtered metadata for HDF5 keys. Kept {len(self.metadata_df)}/{original_len} entries.")
        
        with h5py.File(self.embeddings_file_path, 'r') as h5f:
            sample_key = list(h5f.keys())[0]
            self.embedding_dim = h5f[sample_key].shape[-1]
        self._compute_max_length()
    
        

    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.embeddings_file_path, 'r')

        row = self.metadata_df.iloc[index]
        # Load as float32 — mixed precision autocast handles the rest
        embedding = torch.tensor(self.h5f[str(row[self.id_column])][:], dtype=torch.float32)
        
        if self.max_length > 1:
            seq_len = embedding.shape[0]
            pad_len = self.max_length - seq_len
            embedding = torch.cat([
                embedding,
                torch.zeros(pad_len, self.embedding_dim, dtype=torch.float32),
            ])
        else:
            seq_len = 1
            embedding = embedding.unsqueeze(0)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return embedding, label, str(row[self.id_column]), seq_len

    def __len__(self):
        return len(self.metadata_df)

    def _compute_max_length(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.embeddings_file_path, 'r')
        self.max_length = 1

        for key in self.h5f.keys():
            shape = self.h5f[key].shape
            if len(shape) == 1:
                self.max_length = 1
                continue
            if shape[0] > self.max_length:
                self.max_length = shape[0]
    def get_max_length(self):
        if self.max_length == None:
            self._compute_max_length()
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
        self.encodings = dataset.encodings[indices]
        self.num_classes = dataset.num_classes
        self.label_encoder = dataset.label_encoder
        self.embeddings_file_path = dataset.embeddings_file_path
        self.id_column = dataset.id_column
        self.h5f = None
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


#should move these into TransformerClassifier if they work

def masked_mean(x,padding_mask):
    valid = (~padding_mask).unsqueeze(-1)          # [B,S,1]
    x_sum = (x * valid).sum(dim=1)                # [B,E]
    denom = valid.sum(dim=1).clamp(min=1)         # [B,1]
    
    x = x_sum / denom
    return x


def gen_pad_mask_bool(max_length,lengths,device):
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
                max_length = 5000, dim_feedforward = 2048 ,nhead=4,num_layers_transformer = 1,
                device = "cuda",use_alibi = False,pe_factor=0.01):
        super().__init__()
        self.max_len = max_length
        self.device = device
        self.embed_size = embed_size
       
        self.position_encoder = PositionalEncoding(d_model=embed_size,dropout=dropout_rate,
                                                   max_length=self.max_len,factor=pe_factor)
        #self.position_encoder = LearnedPositionalEmbedding(embed_size,max_len=max_length,dropout=dropout_rate)
        
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
        #self.conv_layer =nn.Conv1d(self.max_len,1, kernel_size=3, padding = 1)
        self.network = nn.Sequential(
            nn.Linear(embed_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, num_classes),
        )
        
    def forward(self, x,lengths):
        #truncate padding per batch, testing if this helps
        
        seq_len = x.size(1)
        #padding mask[batch,max_length] of 0s at exons and -inf at padding
        
        #bool padding mask
        padding_mask = gen_pad_mask_bool(seq_len, lengths, x.device)
        #print(padding_mask.shape) --> torch.Size([16, 28])
        
        #forward mask [max_length,max_length] 
        #src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=self.device,dtype=torch.float32)

        #position encoder +[1,max_length,embed_size]
        x = self.position_encoder(x)

        #[batch,max_length,embed_size]
        #x = self.transformer_encoder(x,src_mask,src_key_padding_mask=padding_mask)
        if self.use_alibi:
            x = self.transformer_encoder(x)
        else: 
            x = self.transformer_encoder(x,src_key_padding_mask=padding_mask)
        #output from [batch,max_length,embed_size] --> [batch,embed_size]
        #x = x.mean(dim=1)
        #x = self.conv_layer(x)[:,-1,:]
        #x= constrained_mean(x,lengths,self.device)
        x = masked_mean(x, padding_mask)
        #[batch,embed_size] --> [batch,number_classes]
        return self.network(x)   
#used prev trainers as outline 
class MultiClassTrainer:
    def __init__(self, model_config, learning_rate, weight_decay,
                 class_weights_tensor, model="Transformer",
                 criterion="CEL", optimizer="Adam", scheduler="Plateau"):
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
        if model == "Transformer":
            self.model = TransformerClassifier(**self.model_config).to(self.device)
        elif model == "Basic":
            self.model = NominalClassifier(**self.model_config).to(self.device)
        #leads to some form of tritonmissing error, which isnt windows compatable, works in linux but has too many errors and performance doesnt seem to be much much better 
        """
        if self.device.type == 'cuda':
            self.model = torch.compile(self.model,mode= "reduce-overhead")
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
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, factor=0.2)
        #lower lr each step by factor of gamma
        elif scheduler == "Exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.8)
        #cyclic cosine warm resets 
        elif scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0 = 8,T_mult = 2, eta_min = learning_rate/10)
        #also warm resets. exp_range is like curved triangler where it converges onto base_lr each cycle
        elif scheduler == "Cyclic":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,base_lr = learning_rate, max_lr = learning_rate*10,step_size_up = 16,mode="exp_range" )
        #no scheduler basicaly, just need a scheduler object, constant has other use
        elif scheduler == "None":
             self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer,factor = 1, total_iters=1)
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
                print(f"  -> Saved best model (val_f1: {current_val_f1:.4f})")
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
                    #need to clean up prune
                    raise optuna.exceptions.TrialPruned()   
        
        if os.path.exists(checkpoint_path):
            checkpoint= self.load_checkpoint(checkpoint_path)
            print(f"INFO: Loaded best model from checkpoint (val_f1: {checkpoint.get('val_f1_macro', 0):.4f})")

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
            # Trim padding to max actual length in this batch
            local_max = torch.max(lengths)
            embeddings = embeddings[:,:local_max,:]
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
            
            predictions = torch.argmax(outputs, dim=1)
            actual = torch.argmax(labels, dim=1)
            total_correct += (predictions == actual).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = (100 * total_correct / total_samples) if total_samples > 0 else 0
        return avg_loss, accuracy
    def load_checkpoint(self, checkpoint_path):
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    #idk seperate function for validation to make report and works as easy function for eval in other cases
    
    def evaluate_on_loader(self, data_loader, label_encoder):
        self.model.eval()
        all_labels = []
        all_preds = []
        all_ids = []
        
        with torch.no_grad():
            for embeddings, labels, ids_batch, lengths in data_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.float().to(self.device)
                lengths = lengths.to(self.device)
                local_max = torch.max(lengths)
                embeddings = embeddings[:, :local_max, :]                
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(embeddings,lengths)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                actual = torch.argmax(labels, dim=1).cpu().numpy()
                all_labels.extend(actual)
                all_preds.extend(predictions)
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
    #to give wandb params, maybe there is easier way, also gives defualts
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
        "dim_feedforward": 2048,
        "use_alibi": False,
        "pe_factor": 1.0,
        "nhead": 4,
        "num_layers_transformer":2,
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
            reinit='finish_previous', 
            mode="disabled" if wandb_disable else "online",
            name=f"trial_{trial.number}"
        )
        
    
        trial_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.6),
            #'hidden_dim1': trial.suggest_categorical('hidden_dim1', [256, 512, 768]),
            #'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            #"optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Lion"]),
            "optimizer": trial.suggest_categorical("optimizer", [ "AdamW", "Lion"]),
            "criterion": trial.suggest_categorical("criterion", ["MSE", "CEL_weightless"]),#"CEL",
            "scheduler": trial.suggest_categorical("scheduler", ["Plateau", "CosineAnnealingWarmRestarts","None"])#"Exponential","Cyclic",
            
        }
       
        if nn_model == "Transformer":
            trial_params["nhead"] = trial.suggest_categorical("nhead", [2, 4,8])#,8
            #trial_params["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [1024,1536, 2048 ])#, 4096
            trial_params["num_layers_transformer"] = trial.suggest_categorical("num_layers_transformer", [1,2,4])#, 4
            trial_params["pe_factor"]= trial.suggest_float("pe_factor", 0.0001, 1.0, log=True)
        
        wandb.config.update(trial_params, allow_val_change=True)
       
        cfg = wandb.config

        # Prepare stratified k-fold validation
        labels = np.argmax(train_dataset.encodings, axis=1)
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        fold_metrics = []
        val_accuracy_metrics = []
        
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
            model_config["use_alibi"] = cfg.use_alibi
            model_config["pe_factor"] = cfg.pe_factor
        global_step = 0
        folds_todo=3
        
        print(f"Optimizer: {cfg.optimizer}\nCriterion: {cfg.criterion}\nScheduler: {cfg.scheduler}\nDropout: {cfg.dropout_rate}\nPe_Factor: {cfg.pe_factor}")
        #as is now, does kfold k times for each tuning step, dont know if this is right or if it should cycle through kfold once each step
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)),labels )):
            print(f"\n--- Trial {trial.number}, Fold {fold+1}/{k_folds} ---")
            folds_remaining = k_folds-fold
            odds = folds_todo/folds_remaining 
            if random.random() > odds:
                print(f"skipping fold {fold+1}, {folds_todo} remaining folds")
                continue
            folds_todo = folds_todo-1
            # Create data loaders
            train_loader = DataLoader(
                Subset(train_dataset, train_idx),
                batch_size=cfg.batch_size,
                num_workers=8,
                pin_memory=True,
                persistent_workers=False,
                shuffle=True 
            )
            val_loader = DataLoader(
                Subset(train_dataset, val_idx),
                batch_size=cfg.batch_size,
                num_workers=8,
                pin_memory=True,
                persistent_workers=False,
                shuffle=False
            )
            #adding this to loaders causes pickle issue with h5py object(not 100% sure why its trying to pickle h5) will try to fix later when i have bigger dataset but for now training time is managable
            """
                num_workers=8,
                pin_memory=True,
                persistent_workers=False,
            """
            #
            # Train model
            trainer = MultiClassTrainer(model_config, cfg.learning_rate, cfg.weight_decay, weights,model=nn_model,optimizer=cfg.optimizer,criterion=cfg.criterion,scheduler=cfg.scheduler)
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
            val_accuracy_metrics.append(val_metrics["accuracy"])
            # Clean up checkpoint
            if os.path.exists(checkpoint):
                os.remove(checkpoint)
            
        # Calculate average metric across folds
        avg_metric = np.mean(fold_metrics)
        val_avg_acc = np.mean(val_accuracy_metrics)
        std_metric = np.std(fold_metrics)
        
        wandb.log({
            "avg_cv_metric": avg_metric,
            "fold_avg_val_acc": val_avg_acc ,
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
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
    return best_params_path
def split_dataset_into_subsets(dataset):
    df = dataset.get_data()
    max_length = dataset.get_max_length()
    train_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==0)[0])
    val_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==1)[0])
    test_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==2)[0])
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_dataset, val_dataset, test_dataset, max_length
#this is kind of pointless now but whatever
def run(test_dataset,wandb_project,wandb_entity,nn_model = "Transformer",
        num_epochs = 100,patience = 10, kfolds = 5,n_trials = 50, random_seed =42,
        wandb_disable = False,max_length=5000 ):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.benchmark = True
        
        
    
    

    yaml_path = run_hpo_mode(
        train_dataset=test_dataset, wandb_project=wandb_project,
        wandb_entity=wandb_entity, nn_model=nn_model,
        max_length=max_length, n_trials=n_trials,
        num_epochs=num_epochs, patience=patience,
        wandb_disable=wandb_disable, k_folds=kfolds,
    )
    return yaml_path
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
def train_model(train_dataset,val_dataset, wandb_project,wandb_entity,yaml_file,wandb_disable=False,embed_size = 1024,max_length=500,num_epochs=200, nn_model="Transformer",patience = 20):
    random_seeds = [42,121,1023,4398,5000]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #defualt args
    args = {
        "embed_size":embed_size,
        "max_length":  max_length,
        "nn_model":  nn_model,
        "num_epochs": num_epochs,
        "patience": patience,
        "dropout_rate": 0.2,
        'learning_rate': 1e-4,
        'weight_decay':1e-4,
        'hidden_dim1': 512,
        "criterion": "MSE",
        "optimizer":"Lion",
        "dim_feedforward": 2048,
        "use_alibi": False,
        "pe_factor": 1.0,
        "nhead": 4,
        "num_layers_transformer":2,
        'batch_size': 16
    } 
    #update with hyperparameter values
    with open(yaml_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    
    args.update(data_loaded)
    
    model_config = {
        'num_classes': train_dataset.num_classes, 
        'embed_size': args["embed_size"], 
        'hidden_dim1': args["hidden_dim1"], 
        'dropout_rate': args["dropout_rate"], 
        "max_length" : max_length
        
    }
    if nn_model == "Transformer":
        model_config["nhead"] = args["nhead"]
        model_config["dim_feedforward"] = args["dim_feedforward"]
        model_config["num_layers_transformer"] = args["num_layers_transformer"]
        model_config["use_alibi"] = args["use_alibi"]
        model_config["pe_factor"] = args["pe_factor"]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],               
        shuffle=True 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"],               
        shuffle=False
    )
    labels = np.argmax(train_dataset.encodings, axis=1)
    class_counts = Counter(labels)
    weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(train_dataset.num_classes)], dtype=torch.float)
    weights = weights / weights.sum() * len(weights)  # Normalize weights
    best_acc = 0
    for random_seed in random_seeds:
        print(f"starting random seed: {random_seed}")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        trial_name = f"val_seed_{random_seed}"
        args["random_seed"] = random_seed
        run = wandb.init(
            project=wandb_project, 
            entity=wandb_entity, 
            config=args, 
            reinit='finish_previous', 
            mode="disabled" if wandb_disable else "online",
            name=trial_name
        )
        cfg = wandb.config
        trainer = MultiClassTrainer(model_config, cfg.learning_rate, cfg.weight_decay, weights,model=nn_model,optimizer=cfg.optimizer,criterion=cfg.criterion,scheduler=cfg.scheduler)
        checkpoint = f"{trial_name}_{wandb_project}.pt"
        val_metrics, epochs_ran = trainer.train_and_validate(
            train_loader, val_loader,
            cfg.num_epochs, cfg.patience,
            checkpoint, train_dataset.label_encoder,
            log_to_wandb=True,
            trial=None,
            step_offset=0 
        )
        print(f"✅ Model training complete. Final model saved to '{checkpoint}'")
        wandb.log({
            "report": val_metrics,  
        })
        current_acc = val_metrics["accuracy"]
        if current_acc > best_acc:
            best_acc = current_acc
            if os.path.exists(best_checkpoint):
                os.remove(best_checkpoint)
            best_checkpoint = checkpoint
        else:
            if os.path.exists(best_checkpoint):
                os.remove(best_checkpoint)
        run.finish()
    return best_checkpoint
def plot_multiclass_confusion_matrix(y_true, y_pred, class_names, save_path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Dynamic figure size based on number of classes
    fig_size = max(10, len(class_names) * 0.5)
    plt.figure(figsize=(fig_size, fig_size))
    
    # Use percentage if many classes
    if len(class_names) > 20:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        cbar_label = 'Percentage (%)'
    else:
        cm_normalized = cm
        fmt = 'd'
        cbar_label = 'Count'
    
    sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': cbar_label})
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"INFO: Confusion matrix saved to {save_path}")
#prob should add test to train_model function but i like them being callable seperate 
def test_model(test_dataset, wandb_project,wandb_entity,yaml_file,checkpoint_path,wandb_disable=False,embed_size = 1024,max_length=500,num_epochs=200, nn_model="Transformer",patience = 20):
    #default args
    args = {
        "embed_size":embed_size,
        "max_length":  max_length,
        "nn_model":  nn_model,
        "num_epochs": num_epochs,
        "patience": patience,
        "dropout_rate": 0.2,
        'learning_rate': 1e-4,
        'weight_decay':1e-4,
        'hidden_dim1': 512,
        "criterion": "MSE",
        "optimizer":"Lion",
        "dim_feedforward": 2048,
        "use_alibi": False,
        "pe_factor": 1.0,
        "nhead": 4,
        "num_layers_transformer":2,
        'batch_size': 16
    } 
    #update with hyperparameter values
    with open(yaml_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    
    args.update(data_loaded)
    #model args
    model_config = {
        'num_classes': test_dataset.num_classes, 
        'embed_size': args["embed_size"], 
        'hidden_dim1': args["hidden_dim1"], 
        'dropout_rate': args["dropout_rate"], 
        "max_length" : max_length
        
    }
    if nn_model == "Transformer":
        model_config["nhead"] = args["nhead"]
        model_config["dim_feedforward"] = args["dim_feedforward"]
        model_config["num_layers_transformer"] = args["num_layers_transformer"]
        model_config["use_alibi"] = args["use_alibi"]
        model_config["pe_factor"] = args["pe_factor"] 
    #make loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],               
        shuffle=False
    )
    #init wandb
    run = wandb.init(
        project=wandb_project, 
        entity=wandb_entity, 
        config=args, 
        reinit='finish_previous', 
        mode="disabled" if wandb_disable else "online",
        name="final_test"
    )
    cfg = wandb.config
    #make trainer object and load checkpoint
    trainer = MultiClassTrainer(model_config, cfg.learning_rate, cfg.weight_decay, None,model=nn_model,optimizer=cfg.optimizer,criterion=cfg.criterion,scheduler=cfg.scheduler)
    trainer.load_checkpoint(checkpoint_path)
    label_encoder = test_dataset.label_encoder
    report, pred_labels, all_ids, true_labels = trainer.evaluate_on_loader(test_loader,label_encoder)
    wandb.log({"report": report})
    #make confusion matrix
    cm_path = "final_model_confusion_matrix.png"
    plot_multiclass_confusion_matrix(true_labels, pred_labels, label_encoder.categories_[0], cm_path)
    wandb.log({"final_confusion_matrix": wandb.Image(cm_path)})
    run.finish()
#personal key lol
if __name__ == '__main__':
    #python classifier_runner.py --h5 CYP_PA_Attempt4_per_exon.h5 --csv CYP_PA_Attempt4_train_val_test.csv --entity per_exon --nn_model Transformer
    parser = argparse.ArgumentParser(description="Train or optimize a multi-class SCPP classifier.")
    parser.add_argument("--h5", required=True, help="Path to embeddings HDF5 file")
    parser.add_argument("--csv", required=True, help="Path to metadata CSV")
    parser.add_argument("--entity", required=True, help="entity name")
    parser.add_argument("--nn_model", required=True, help="nn_model type",choices=["Basic","Transformer"])
    parser.add_argument("--project", default="per-exon-testing")
    parser.add_argument("--wandb_disable", action="store_true")
    parser.add_argument("--yaml_file", help="Yaml file with params, if given, will skip HPO step")
    parser.add_argument("--pt_file", help="path to pt file contianing model weights, will skip validation step, if a yaml_file was also given")
    args = parser.parse_args()
    train_dataset, val_dataset, test_dataset, max_length =split_dataset_into_subsets(MultiClassDataset(embeddings_path=args.h5,csv_path=args.csv))
    print(f"Max Length: {max_length}")
    if args.yaml_file:
        yaml_path = args.yaml_file
    else:
        yaml_path = run(train_dataset,args.entity+"_HPO",args.project,nn_model=args.nn_model,n_trials=80,num_epochs=30,patience=5,wandb_disable=False)
    if args.pt_file and args.yaml_file:
        best_model = args.pt_file
    else:
        best_model= train_model(train_dataset,val_dataset,args.entity+"_test",args.project,yaml_path,nn_model = args.nn_model,wandb_disable=False)
    test_model(test_dataset,args.entity+"_test",args.project,yaml_path,best_model,nn_model = args.nn_model)


    