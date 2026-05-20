#!/usr/bin/env python3

import argparse
from scripts.custom_datasets import split_dataset_into_subsets, MultiClassDataset
from scripts.runners_eval import run, test_model, train_model
YAML_FOLDER = "yaml"
MODEL_WEIGHTS_FOLDER = "model_weights"

if __name__ == '__main__':
    #python classifier_runner.py --h5 CYP_PA_Attempt5_per_exon.h5 --csv CYP_PA_Attempt5_train_val_test.csv --entity cel_weightless_e --nn_model Transformer
    parser = argparse.ArgumentParser(description="Train or optimize a multi-class SCPP classifier.")
    parser.add_argument("--h5", required=True, help="Path to embeddings HDF5 file")
    parser.add_argument("--csv", required=True, help="Path to metadata CSV")
    parser.add_argument("--entity", required=True, help="entity name")
    parser.add_argument("--nn_model", required=True, help="nn_model type",choices=["Basic","Transformer","Pooling"])
    parser.add_argument("--project", default="per-exon-testing")
    parser.add_argument("--wandb_disable", action="store_true")
    parser.add_argument("--yaml_file", help="Yaml file with params, if given, will skip HPO step")
    parser.add_argument("--pt_file", help="path to pt file contianing model weights, will skip validation step, if a yaml_file was also given")
    args = parser.parse_args()
    train_dataset, val_dataset, test_dataset, max_length =split_dataset_into_subsets(MultiClassDataset(embeddings_path=args.h5,csv_path=args.csv))
    embed_size = train_dataset.embedding_dim

    print(f"Max Length: {max_length}")
    if args.yaml_file:
        yaml_path = args.yaml_file
    else:
        yaml_path = run(train_dataset,args.entity+"_HPO",args.project,nn_model=args.nn_model,n_trials=30,num_epochs=35,patience=5,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,yaml_folder=YAML_FOLDER, checkpoint_folder=MODEL_WEIGHTS_FOLDER)
        
    if args.pt_file and args.yaml_file:
        best_model = args.pt_file
    else:
        best_model= train_model(train_dataset,val_dataset,args.entity+"_test",args.project,yaml_path,nn_model = args.nn_model,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,checkpoints_folder=MODEL_WEIGHTS_FOLDER)
    test_model(test_dataset,args.entity+"_test",args.project,yaml_path,best_model,nn_model = args.nn_model,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size)


    