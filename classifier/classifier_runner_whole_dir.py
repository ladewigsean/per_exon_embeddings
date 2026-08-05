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
PREDICTIONS_FOLDER = "predictions"
COLUMNS = ["method", "mean_acc", "std_acc", "mean_macro_f1", "std_macro_f1",
           "test_acc", "test_macro_f1"]

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
            file.write(",".join(COLUMNS))
            file.write("\n")
    # Resume by first column whatever it is called: some existing output_csvs were written
    # with an empty first header, and old_data["method"] raises KeyError on those.
    done = set()
    if old_data is not None and len(old_data.columns):
        done = set(old_data[old_data.columns[0]].astype(str))
    for h5 in h5s:
        entity = h5.stem

        if str(entity) in done:
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
        # `data` is VALIDATION metrics: HPO tuned on train, early stopping and best-seed
        # selection both used val, so those numbers are optimistically biased and are not
        # a held-out result. Split 2 has never been touched -- evaluate on it here, and
        # write the per-example rows the stratified analysis needs.
        test_acc = test_f1 = ""
        if not args.skip_test:
            try:
                if len(test_dataset) == 0:
                    raise ValueError("test split is empty -- check cluster_split's split_report")
                os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
                test_report = test_model(
                    test_dataset, entity+"_final_test", args.project, yaml_path, best_model,
                    nn_model=nn_model, wandb_disable=args.wandb_disable,
                    max_length=max_length, embed_size=embed_size,
                    pred_out=os.path.join(PREDICTIONS_FOLDER, f"{entity}.csv"),
                    cm_path=os.path.join(PREDICTIONS_FOLDER, f"{entity}_confusion_matrix.png"))
                test_acc = test_report["accuracy"]
                test_f1 = test_report["macro avg"]["f1-score"]
            except Exception as exc:
                # Never let one arm's test failure kill a multi-hour sweep, but say so
                # loudly -- a silently blank test column would read as "not run yet".
                print(f"WARNING: test evaluation FAILED for {entity}: {exc}", file=sys.stderr)
        with open(output,"a") as file:
            data = [str(d) for d in data] + [str(test_acc), str(test_f1)]
            file.write(",".join([entity]+data))
            file.write("\n")
        
    



    