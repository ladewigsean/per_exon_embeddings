#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import pathlib
import os
import re
import sys
from scripts.custom_datasets import split_dataset_into_subsets, MultiClassDataset
from scripts.runners_eval import run, test_model, train_model,performance_test
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
    parser.add_argument("--default_avg",type=int, default=20,help="number of epochs to average over")
    parser.add_argument("--skip_per_res", action="store_true", help="if flag given, skips over per_res.h5")
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
    output_folder = os.path.join(OUTPUT_CSVS_FOLDER,f"performance_{dir_name}")
    batches_folder = os.path.join(output_folder,"batches")
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        os.mkdir(batches_folder)
        
    output_general = os.path.join(output_folder,f"{dir_name}_performance.csv")
    
    old_data = None
    if not os.path.isfile(output_general) or args.overwrite:
        with open(output_general,"w") as file:
            file.write(",".join(["method","mean_load","std_load","mean_general","std_general"]))
            file.write("\n")
    else:
        old_data = pd.read_csv(output_general)    
    
    yaml_path = os.path.join(YAML_FOLDER,"DEEPLOC_per_prot_final_HPO.yaml")   
    for h5 in h5s:
        entity = str(h5.stem)
        print(f"Starting {entity}")
        h5 = str(h5)
        if "per_res" in str(entity) and args.skip_per_res:
            print(f"Skip_per_res flag given, Skipping {h5}\n")
            continue 
        #this is now not needed 
        if "per_res" in str(entity):
            batch_size = 16
            average_of = 20
        else:
            batch_size = 16
            average_of = args.default_avg
        
        
        h5 = str(h5)
        
        
        
        if not (not old_data is None and str(entity) in list(old_data["method"])):
            

            train_dataset, val_dataset, test_dataset, max_length =split_dataset_into_subsets(MultiClassDataset(embeddings_path=h5,csv_path=csv_file))
            embed_size = train_dataset.embedding_dim
            print(f"Max Length: {max_length}")
            if max_length>1:
                nn_model = "Transformer"
            else:
                nn_model = "Basic"
            seq_load_times, general_time, batches_lengths,batches_forward, batches_backwards = performance_test(train_dataset,yaml_path,embed_size,max_length,average_of,nn_model,batch_size=batch_size)
            seq_load_times = np.array(seq_load_times)
            general_time = np.array(general_time)
            data = [str(seq_load_times.mean()),str(seq_load_times.std()), str(general_time.mean()),str(general_time.std())]
            with open(output_general,"a") as file:
                
                file.write(",".join([entity]+data))
                file.write("\n")
            df = pd.DataFrame.from_dict({"max_length":batches_lengths,"forward":batches_forward,"backward":batches_backwards})
            df.to_csv(os.path.join(batches_folder,f"{entity}.csv"))
            
        if "per_prot" in str(entity) and not "meta" in str(entity):
            train_dataset, val_dataset, test_dataset, max_length =split_dataset_into_subsets(MultiClassDataset(embeddings_path=h5,csv_path=csv_file))
            embed_size = train_dataset.embedding_dim
            entity = entity+"_transformer"
            if not old_data is None and str(entity) in list(old_data["method"]):
                continue
            nn_model = "Transformer"
            seq_load_times, general_time, batches_lengths,batches_forward, batches_backwards = performance_test(train_dataset,yaml_path,embed_size,max_length,args.default_avg,nn_model,batch_size=batch_size)
            seq_load_times = np.array(seq_load_times)
            general_time = np.array(general_time)
            data = [str(seq_load_times.mean()),str(seq_load_times.std()), str(general_time.mean()),str(general_time.std())]
            with open(output_general,"a") as file:
                
                file.write(",".join([entity]+data))
                file.write("\n")
            df = pd.DataFrame.from_dict({"max_length":batches_lengths,"forward":batches_forward,"backward":batches_backwards})
            df.to_csv(os.path.join(batches_folder,f"{entity}.csv"))
        
        
    

