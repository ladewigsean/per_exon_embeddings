#!/usr/bin/env python3
import sys
import random
import os


import numpy as np
from collections import Counter
import yaml

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from lion_pytorch import Lion
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,log_loss
import matplotlib.pyplot as plt

import wandb
import optuna
from tqdm import tqdm

from scripts.custom_loss import WeightedMSELoss,FocalLossCELoss
from scripts.custom_datasets import _h5_worker_init_fn
from scripts.modules import NominalClassifier,PoolingClassifier,TransformerClassifier

from torch.amp import GradScaler, autocast
default_args = {
    
    "dropout_rate": 0.20,
    'learning_rate': 1e-4,
    'weight_decay':1e-4,
    'hidden_dim1': 512,
    "criterion": "MSE",
    "optimizer":"AdamW",
    "scheduler":"CosineAnnealingWarmRestarts",
    "dim_feedforward": 2048,
    "use_alibi": False,
    "pe_factor": 0.0,
    "nhead": 4,
    "num_layers_transformer":2,
    'batch_size': 64,
    "pe_mode": "pe",#"pe","learned_pe"
    "dc": 8,
} 
class MultiClassTrainer:
    def __init__(self, model_config, learning_rate, weight_decay,
                 class_weights_tensor, model="Transformer",
                 criterion="CEL", optimizer="Adam", scheduler="Plateau",device = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"INFO: Using device: {self.device}")
        self.class_weights = class_weights_tensor
        self.model_config = model_config
        #added criterion as hyperparameter cause desperate
        if criterion == "CEL":
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights_tensor.to(self.device) if class_weights_tensor is not None else None,
                ##label_smoothing=0.1
            )
        elif criterion == "CEL_focal":
            self.criterion = FocalLossCELoss(gamma=5)
        elif criterion == "CEL_weightless":
            self.criterion = nn.CrossEntropyLoss()
        #wierdly works best despite not being meant for 
        elif criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif criterion == "MSE_weighted":
            self.criterion = WeightedMSELoss(
                weight=class_weights_tensor.to(self.device) if class_weights_tensor is not None else None
            )
        if model == "Transformer":
            self.model = TransformerClassifier(**self.model_config).to(self.device)
        elif model == "Basic":
            self.model = NominalClassifier(**self.model_config).to(self.device)
        elif model == "Pooling":
            self.model = PoolingClassifier(**self.model_config).to(self.device)
        #leads to some form of tritonmissing error, which isnt windows compatable, works in linux but has too many errors and performance doesnt seem to be much much better 
        #todo: get this to work   
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
        self.softmax = nn.Softmax(dim=1)

    def train_and_validate(self, train_loader, val_loader, num_epochs, patience, checkpoint_path, 
                      label_encoder, log_to_wandb=False, step_offset=0):
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
            parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
            if len(parameters) == 0:
                total_norm = 0.0
            else:
                device = parameters[0].grad.device
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()
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
                    "learning_rate": current_lr,
                    "val_entropy": val_report["entropy"],
                    "total_norm": total_norm
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
        all_lengths = []
        all_preds_raw = torch.empty((0,len(label_encoder.categories_[0])),dtype=torch.float32)

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
                all_preds_raw = torch.vstack([all_preds_raw,outputs.cpu()])
                all_lengths.extend(lengths.cpu().numpy())

        report = classification_report(
            all_labels, all_preds, 
            target_names=label_encoder.categories_[0], 
            output_dict=True, 
            zero_division=0
        ) 
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_lengths = np.array(all_lengths)
        bucket1_ind = (all_lengths <=4).nonzero()[0]
        bucket2_ind = np.logical_and(4<all_lengths,all_lengths<=8).nonzero()[0]
        bucket3_ind = (all_lengths >= 9).nonzero()[0]
        
        report["bucket_1_val_acc"] = float(np.sum(all_labels[bucket1_ind]==all_preds[bucket1_ind])/len(bucket1_ind)) if len(bucket1_ind)!=0 else np.nan
        report["bucket_2_val_acc"] = float(np.sum(all_labels[bucket2_ind]==all_preds[bucket2_ind])/len(bucket2_ind)) if len(bucket2_ind)!=0 else np.nan
        report["bucket_3_val_acc"] = float(np.sum(all_labels[bucket3_ind]==all_preds[bucket3_ind])/len(bucket3_ind)) if len(bucket3_ind)!=0 else np.nan
        try:
            report["entropy"] = log_loss(
                all_labels, self.softmax(all_preds_raw).numpy(),
                #labels=label_encoder.categories_[0],
                #sample_weight=self.class_weights
            )
        except Exception:
            report["entropy"]= None
        #
        return report,  all_preds, all_ids, all_labels

def run_hpo_mode(train_dataset,wandb_project,wandb_entity,hpo_metric="macro avg",n_trials = 50,
                 num_epochs = 100 ,wandb_disable=False,k_folds = 5,max_length = 5000,nn_model = "Transformer",random_seed = 42,embed_size=1024,
                 patience = 10,yaml_folder = "yaml",checkpoint_folder="model_weights"):

    #to give wandb params, maybe there is easier way, also gives defualts
    args = default_args.copy()
    args.update({
        "hpo_metric":hpo_metric,
        "embed_size":embed_size,
        "n_trials":  n_trials,
        "k_folds":  k_folds,
        "max_length":  max_length,
        "nn_model":  nn_model,
        "random_seed":  random_seed,
        "num_epochs": num_epochs,
        "patience": patience,
    })
    
    
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
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.8),
            #'hidden_dim1': trial.suggest_categorical('hidden_dim1', [256, 512, 768]),
            #'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            #"optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Lion"]),
            #"optimizer": trial.suggest_categorical("optimizer", [ "AdamW", "Lion"]),
            #"criterion": trial.suggest_categorical("criterion", ["MSE", "CEL_weightless"]),#"CEL",
            #"scheduler": trial.suggest_categorical("scheduler", ["Plateau", "CosineAnnealingWarmRestarts","None"])#"Exponential","Cyclic",
            
        }
       
        if nn_model == "Transformer":
            #trial_params["nhead"] = trial.suggest_categorical("nhead", [2, 4,8])#,8
            #trial_params["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [1024,1536, 2048 ])#, 4096
            #trial_params["num_layers_transformer"] = trial.suggest_categorical("num_layers_transformer", [1,2,4])#, 4
            if ("pe_mode" in trial_params and trial_params["pe_mode"]=="pe") or  ("pe_mode" not in trial_params and args["pe_mode"]=="pe"):
                trial_params["pe_factor"]= trial.suggest_float("pe_factor", 1e-4, 5, log=True)
        elif nn_model == "Pooling":
            trial_params["dc"] = trial.suggest_categorical("dc",[4,8,16,32,64,96])
        
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
            model_config["pe_mode"] = cfg.pe_mode
        elif nn_model == "Pooling":
            model_config["dc"] = cfg.dc
        global_step = 0
        
        
        print(f"Optimizer: {cfg.optimizer}\nCriterion: {cfg.criterion}\nScheduler: {cfg.scheduler}\nDropout: {cfg.dropout_rate}\nPe_Factor: {cfg.pe_factor}")
        #as is now, does kfold k times for each tuning step, dont know if this is right or if it should cycle through kfold once each step
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)),labels )):
            print(f"\n--- Trial {trial.number}, Fold {fold+1}/{k_folds} ---")
            
            # Create data loaders
            train_loader = DataLoader(
                Subset(train_dataset, train_idx),
                batch_size=cfg.batch_size,
                num_workers=8,
                worker_init_fn=_h5_worker_init_fn, 
                persistent_workers=True,
                shuffle=True 
            )
            val_loader = DataLoader(
                Subset(train_dataset, val_idx),
                batch_size=cfg.batch_size,
                num_workers=8,
                worker_init_fn=_h5_worker_init_fn, 
                persistent_workers=True,
                shuffle=False
            )
            
            #
            # Train model
            trainer = MultiClassTrainer(model_config, cfg.learning_rate, cfg.weight_decay, weights,model=nn_model,optimizer=cfg.optimizer,criterion=cfg.criterion,scheduler=cfg.scheduler)
            checkpoint = os.path.join(checkpoint_folder, f"temp_trial_{trial.number}_{wandb_project}_fold_{fold}.pt")

            # Pass step_offset and update it
            val_metrics, epochs_ran = trainer.train_and_validate(
                train_loader, val_loader,
                cfg.num_epochs, cfg.patience,
                checkpoint, train_dataset.label_encoder,
                log_to_wandb=True,
                
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
            trial.report(metric_value, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n{'='*60}")
    print("✅ HPO Complete")
    print(f"{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best {hpo_metric}: {study.best_value:.4f}")
    
    # Save best hyperparameters
    best_params_path = os.path.join(yaml_folder, wandb_project+".yaml")
    with open(best_params_path, 'w') as f:
        params = default_args.copy()
        params.update(study.best_trial.params)
        
        
        yaml.dump(params, f, default_flow_style=False)
    
    print(f"\n💾 Best hyperparameters saved to '{best_params_path}':")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    return best_params_path

#this is kind of pointless now but whatever
def run(test_dataset,wandb_project,wandb_entity,nn_model = "Transformer",
        num_epochs = 100,patience = 10, kfolds = 5,n_trials = 50, random_seed =42, embed_size = 1024,
        wandb_disable = False,max_length=5000 ,yaml_folder = "yaml",checkpoint_folder = "model_weights"):
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
        embed_size=embed_size,yaml_folder=yaml_folder,
        checkpoint_folder=checkpoint_folder
    )
    return yaml_path
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
def train_model(train_dataset,val_dataset, wandb_project,wandb_entity,yaml_file,wandb_disable=False,embed_size = 1024,max_length=500,num_epochs=100, nn_model="Transformer",patience = 10,checkpoints_folder="model_weights"):
    random_seeds = [42,121,1023,4398,5000]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #defualt args
    args = default_args.copy()
    args.update({
        "embed_size":embed_size,
        "max_length":  max_length,
        "nn_model":  nn_model,
        "num_epochs": num_epochs,
        "patience": patience,
    } )
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
        model_config["pe_mode"] = args["pe_mode"]
    elif nn_model == "Pooling":
        model_config["dc"]  = args["dc"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=8,
        worker_init_fn=_h5_worker_init_fn, 
        persistent_workers=True,               
        shuffle=True 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"],    
        num_workers=8,
        worker_init_fn=_h5_worker_init_fn, 
        persistent_workers=True,              
        shuffle=False
    )
    labels = np.argmax(train_dataset.encodings, axis=1)
    class_counts = Counter(labels)
    weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(train_dataset.num_classes)], dtype=torch.float)
    weights = weights / weights.sum() * len(weights)  # Normalize weights
    best_acc = 0
    best_checkpoint = None
    seed_accs = []
    seed_macro_f1 = []
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
        checkpoint = os.path.join(checkpoints_folder,f"{trial_name}_{wandb_project}.pt")
        val_metrics, epochs_ran = trainer.train_and_validate(
            train_loader, val_loader,
            cfg.num_epochs, cfg.patience,
            checkpoint, train_dataset.label_encoder,
            log_to_wandb=True,
            
            step_offset=0 
        )
        print(f"✅ Model training complete. Final model saved to '{checkpoint}'")
        wandb.log({
            "report": val_metrics,  
        })
        current_acc = val_metrics["accuracy"]
        seed_accs.append(current_acc)
        seed_macro_f1.append(val_metrics["macro avg"]["f1-score"])
        if current_acc > best_acc:
            best_acc = current_acc
            if best_checkpoint:
                if os.path.exists(best_checkpoint):
                    os.remove(best_checkpoint)
            best_checkpoint = checkpoint
        else:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)
        run.finish()
    seed_accs = np.array(seed_accs)
    seed_macro_f1 = np.array(seed_macro_f1)
    print(f"average acc: {seed_accs.mean():.2f} ± {1.96 * seed_accs.std():.2f}")
    print(f"average macro f1: {seed_macro_f1.mean():.2f} ± {1.96 * seed_macro_f1.std():.2f}")
    return best_checkpoint, [seed_accs.mean(),seed_accs.std(),seed_macro_f1.mean(),seed_macro_f1.std()]
def plot_multiclass_confusion_matrix(y_true, y_pred, class_names, save_path):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    
    # Dynamic figure size based on number of classes
    fig_size = max(10, len(class_names) * 0.5)
    plt.figure(figsize=(fig_size, fig_size))
    
    # Use percentage if many classes
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fmt = '.1f'
    cbar_label = 'Percentage (%)'
    
    
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
    args = default_args.copy()
    args.update({
        "embed_size":embed_size,
        "max_length":  max_length,
        "nn_model":  nn_model,
        "num_epochs": num_epochs,
        "patience": patience,
    } )
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
        model_config["pe_mode"] = args["pe_mode"]
    elif nn_model == "Pooling":
            model_config["dc"] = args["dc"]
    #make loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"], 
        num_workers=8,
        worker_init_fn=_h5_worker_init_fn, 
        persistent_workers=True,                 
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