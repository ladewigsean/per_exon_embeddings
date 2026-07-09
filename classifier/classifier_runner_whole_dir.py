#!/usr/bin/env python3

import argparse
import pandas as pd
import pathlib
import os 
import sys
from scripts.custom_datasets import split_dataset_into_subsets, MultiClassDataset
from scripts.runners_eval import run, test_model, train_model
YAML_FOLDER = "yaml"
MODEL_WEIGHTS_FOLDER = "model_weights"
OUTPUT_CSVS_FOLDER = "output_csvs"

if __name__ == '__main__':
    #python classifier_runner.py --entity SPE_per_exon --nn_model Transformer --h5 input_data/SPE/SerProtEuk_per_exon.h5 --csv input_data/SPE/SerProtEuk.csv
    parser = argparse.ArgumentParser(description="Train or optimize a multi-class SCPP classifier.")
    parser.add_argument("--dir", required=True, help="Path to dataset directory")
    
    parser.add_argument("--project", default="per-exon-testing")
    parser.add_argument("--wandb_disable", action="store_true")
    
    parser.add_argument("--hpo_trials",type=int, default=30,help="number of hpo trials. default 30")
    parser.add_argument("--skip_per_res", action="store_true", help="if flag given, skips over per_res.h5")
    args = parser.parse_args()
    dir_name = pathlib.Path(args.dir).stem
    print(dir_name)
    csv_file = None
    csvlist = [path for path in pathlib.Path(args.dir).rglob('*.csv')]
    if len(csvlist) == 0:
        raise FileNotFoundError("No CSV found in given dir")
    elif len(csvlist) > 1:
        raise FileNotFoundError("multiple CSV found in given dir")
    else:
        csv_file = str(csvlist[0])
    h5s = pathlib.Path(args.dir).rglob('*.h5')
    results = {}
    output = os.path.join(OUTPUT_CSVS_FOLDER,f"{dir_name}.csv")
    old_data = None
    if os.path.isfile(output):
        old_data = pd.read_csv(output)
    if old_data is None:
        with open(output,"w") as file:
            file.write(",".join(["method","mean_acc","std_acc","mean_macro_f1","std_macro_f1"]))
            file.write("\n")
    for h5 in h5s:
        entity = h5.stem
        
        if not old_data is None and str(entity) in list(old_data["method"]):
            continue
        h5 = str(h5)
        if "per_res" in str(entity) and args.skip_per_res:
            print(f"Skip_per_res flag given, Skipping {h5}\n")
            continue
        train_dataset, val_dataset, test_dataset, max_length =split_dataset_into_subsets(MultiClassDataset(embeddings_path=h5,csv_path=csv_file))
        embed_size = train_dataset.embedding_dim
        print(f"Max Length: {max_length}")
        if max_length>1:
            nn_model = "Transformer"
        else: 
            nn_model = "Basic"
        yaml_path = run(train_dataset,entity+"_HPO",args.project,nn_model=nn_model,n_trials=args.hpo_trials,num_epochs=35,patience=5,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,yaml_folder=YAML_FOLDER, checkpoint_folder=MODEL_WEIGHTS_FOLDER,)
        best_model, data= train_model(train_dataset,val_dataset,entity+"_test",args.project,yaml_path,nn_model = nn_model,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,checkpoints_folder=MODEL_WEIGHTS_FOLDER)
        with open(output,"a") as file:
            data = [str(d) for d in data]
            file.write(",".join([entity]+data))
            file.write("\n")
        
    



    