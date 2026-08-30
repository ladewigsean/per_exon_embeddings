#!/usr/bin/env python3

import argparse
import pandas as pd
import pathlib
import os
import re
import sys
from scripts.custom_datasets import split_dataset_into_subsets, MultiClassDataset
from scripts.runners_eval import run, test_model, train_model
YAML_FOLDER = "yaml"
MODEL_WEIGHTS_FOLDER = "model_weights"
OUTPUT_CSVS_FOLDER = "output_csvs"
PREDICTIONS_FOLDER = "predictions"
# mean_*/std_* are VALIDATION metrics over 5 seeds; test_* is a single held-out
# evaluation of the val-best checkpoint (see the note next to the test_model call).
COLUMNS = ["method", "mean_acc", "std_acc", "mean_macro_f1", "std_macro_f1",
           "test_acc", "test_macro_f1", "test_status"]


def find_surviving_checkpoint(entity):
    """The one checkpoint train_model kept for `entity`, or None if that is ambiguous.

    train_model names them f"val_seed_{seed}_{wandb_project}.pt" and deletes all but the
    val-best, so a COMPLETED arm leaves exactly one file behind. But --force_test exists
    to recover runs that did not complete, which is exactly when several seeds survive
    and there is no record of which was best. Refuse to guess rather than silently score
    an arbitrary seed and report it as the result.

    The regex matters too: a bare glob lets `*` swallow a family prefix, so entity
    "HOX_per_prot" would match "val_seed_1_NCBIHOX_per_prot_test.pt" from a different
    dataset (model_weights/ is one shared folder and the repo has both HOX and NCBI_HOX).
    """
    pattern = re.compile(rf"val_seed_\d+_{re.escape(entity)}_test\.pt$")
    matches = sorted(p for p in pathlib.Path(MODEL_WEIGHTS_FOLDER).glob("val_seed_*.pt")
                     if pattern.match(p.name))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        print(f"WARNING: {len(matches)} surviving checkpoints for {entity} "
              f"({', '.join(p.name for p in matches)}); cannot tell which seed was "
              f"val-best, so refusing to guess. Retrain this arm.", file=sys.stderr)
    return None


if __name__ == '__main__':
    #python classifier_runner.py --entity SPE_per_exon --nn_model Transformer --h5 input_data/SPE/SerProtEuk_per_exon.h5 --csv input_data/SPE/SerProtEuk.csv
    parser = argparse.ArgumentParser(description="Train or optimize a multi-class SCPP classifier.")
    parser.add_argument("--dir", required=True, help="Path to dataset directory")

    parser.add_argument("--project", default="per-exon-testing")
    parser.add_argument("--wandb_disable", action="store_true")

    parser.add_argument("--hpo_trials",type=int, default=30,help="number of hpo trials. default 30")
    parser.add_argument("--skip_per_res", action="store_true", help="if flag given, skips over per_res.h5")
    parser.add_argument("--skip_test", action="store_true",
                        help="skip held-out test evaluation (the run then reports VALIDATION "
                             "numbers only, which is what this script used to do)")
    parser.add_argument("--force_test", action="store_true",
                        help="for arms already in the output CSV but with no test_acc, re-run "
                             "ONLY the test stage on the surviving checkpoint. Skips HPO and "
                             "training, so it costs a forward pass rather than a whole sweep.")
    parser.add_argument("--n_bins", type= int,default=10, help="number of bins")
    parser.add_argument("--overwrite",action="store_true",help="overwrite old files")
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
    output_folder = os.path.join(OUTPUT_CSVS_FOLDER,dir_name)
    test_folder = os.path.join(output_folder,"test_values_folder")
    best_test_folder = os.path.join(output_folder,"best_predict_only_folder")
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        os.mkdir(test_folder)
        os.mkdir(best_test_folder)
    output_val = os.path.join(output_folder,f"{dir_name}_val.csv")
    output_test = os.path.join(output_folder,f"{dir_name}_test.csv")
    output_pident = os.path.join(output_folder,f"{dir_name}_pident_{args.n_bins}.csv")
    old_data = None
    if not os.path.isfile(output_val) or args.overwrite:
        with open(output_val,"w") as file:
            file.write(",".join(["method","mean_acc","std_acc","mean_macro_f1","std_macro_f1"]))
            file.write("\n")
    else:
        old_data = pd.read_csv(output_val)    
    
        
    if not os.path.isfile(output_test) or args.overwrite:
        with open(output_test,"w") as file:
            file.write(",".join(["method","mean_acc","std_acc","mean_macro_f1","std_macro_f1"]))
            file.write("\n")
    if not os.path.isfile(output_pident) or args.overwrite:
        with open(output_pident,"w") as file:
            size = 100// args.n_bins
            ranges = [(f"{str(x*size)}-{str((x+1)*size)}%") for x in range(args.n_bins)]
            file.write(",".join(["method"]+ranges))
            file.write("\n")
    for h5 in h5s:
        entity = str(h5.stem)
        
        h5 = str(h5)
        if "per_res" in str(entity) and args.skip_per_res:
            print(f"Skip_per_res flag given, Skipping {h5}\n")
            continue 
        # --force_test re-runs ONLY the cheap test stage for arms trained before the
        # runner evaluated split 2, so held-out numbers do not cost a repeat of the
        # HPO + 5-seed sweep.
        
        h5 = str(h5)
        
        # Check the artefacts --force_test depends on BEFORE opening the h5. Building the
        # dataset first means a per_res arm reads a multi-GB file and scans every key,
        # only to bail two lines later because there is no checkpoint to score.
        yaml_path = best_model = None
        if not (not old_data is None and str(entity) in list(old_data["method"])):
            

            train_dataset, val_dataset, test_dataset, max_length =split_dataset_into_subsets(MultiClassDataset(embeddings_path=h5,csv_path=csv_file))
            embed_size = train_dataset.embedding_dim
            print(f"Max Length: {max_length}")
            if max_length>1:
                nn_model = "Transformer"
            else:
                nn_model = "Basic"
            yaml_path = os.path.join(YAML_FOLDER, entity+"_final_HPO.yaml")
            if not os.path.isfile(yaml_path):  
                yaml_path = run(train_dataset,entity+"_final_HPO",args.project,nn_model=nn_model,n_trials=args.hpo_trials,num_epochs=25,patience=5,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,yaml_folder=YAML_FOLDER, checkpoint_folder=MODEL_WEIGHTS_FOLDER,)
            best_model, val_data,test_data,pident,test_out_df,best_test_df= train_model(train_dataset,val_dataset,test_dataset,entity+"_final",args.project,yaml_path,nn_model = nn_model,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,checkpoints_folder=MODEL_WEIGHTS_FOLDER,n_bins=args.n_bins)
            with open(output_val,"a") as file:
                data = [str(d) for d in val_data]
                file.write(",".join([entity]+data))
                file.write("\n")
            with open(output_test,"a") as file:
                data = [str(d) for d in test_data]
                file.write(",".join([entity]+data))
                file.write("\n")
            with open(output_pident,"a") as file:
                data = [str(d) for d in pident]
                file.write(",".join([entity]+data))
                file.write("\n")
            test_out_df.to_csv(os.path.join(test_folder,f"{entity}.csv"))
            best_test_df["identifier" ] = best_test_df.index
            best_test_df.to_csv(os.path.join(best_test_folder,f"{entity}.csv"),index = False)
            #per prot transformer, pretty scuffed but works
        if "per_prot" in str(entity) and not "meta" in str(entity):
            train_dataset, val_dataset, test_dataset, max_length =split_dataset_into_subsets(MultiClassDataset(embeddings_path=h5,csv_path=csv_file))
            entity = entity+"_transformer"
            if not old_data is None and str(entity) in list(old_data["method"]):
                continue
            nn_model = "Transformer"
            yaml_path = os.path.join(YAML_FOLDER, entity+"_final_HPO.yaml")
            if not os.path.isfile(yaml_path):  
                yaml_path = run(train_dataset,entity+"_final_HPO",args.project,nn_model=nn_model,n_trials=args.hpo_trials,num_epochs=25,patience=5,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,yaml_folder=YAML_FOLDER, checkpoint_folder=MODEL_WEIGHTS_FOLDER,)
            best_model, val_data,test_data,pident,test_out_df,best_test_df= train_model(train_dataset,val_dataset,test_dataset,entity+"_final",args.project,yaml_path,nn_model = nn_model,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,checkpoints_folder=MODEL_WEIGHTS_FOLDER,n_bins=args.n_bins)
            with open(output_val,"a") as file:
                data = [str(d) for d in val_data]
                file.write(",".join([entity]+data))
                file.write("\n")
            with open(output_test,"a") as file:
                data = [str(d) for d in test_data]
                file.write(",".join([entity]+data))
                file.write("\n")
            with open(output_pident,"a") as file:
                data = [str(d) for d in pident]
                file.write(",".join([entity]+data))
                file.write("\n")
            test_out_df.to_csv(os.path.join(test_folder,f"{entity}.csv"))
            best_test_df["identifier" ] = best_test_df.index
            best_test_df.to_csv(os.path.join(best_test_folder,f"{entity}.csv"),index = False)
        
        
    

