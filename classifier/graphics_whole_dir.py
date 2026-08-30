import pandas as pd
import os 
import numpy as np
import sys
import argparse
import pathlib
import re
import scripts.stratified_analysis
#to make personal args namespaces for Ivan scripts
class Namespace:
    def __init__(self, kwargs):
        self.__dict__.update(**kwargs)
DEFAULT_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SUBSETS = {
    "global":
    {
        "baseline":"per_prot",
        "contains":
        [
            "per_prot_transformer","fixed_length_chunks","fixed_total_chunks","per_exon","per_exon_meta","per_res", 
            "per_prot","per_prot_meta","per_prot_meta_shuffle",
            "meta_only","meta_only_shuffle","length_only","phase_only"
        ]
    },
    "transformer":
    {
        "baseline":"per_prot_transformer",
        "contains":
        [
            "per_prot_transformer","fixed_length_chunks","fixed_total_chunks","per_exon","per_exon_meta", "per_res", 
                    
        ]
    },
    "per_prot":
    {
        "baseline":"per_prot",
        "contains":
        [
            "per_prot","per_prot_meta","per_prot_meta_shuffle",
        ]
    },
    "exon_aware":
        {
            "baseline":"per_prot",
            "contains":
            [
                "per_prot","per_prot_meta","per_prot_meta_shuffle",
                "per_prot_transformer","fixed_length_chunks","fixed_total_chunks","per_exon","per_exon_meta", "per_res", 
            ]
        },
    "meta":
    {
        "baseline":"per_prot",
        "contains":
        [
            "per_prot","meta_only","meta_only_shuffle","length_only","phase_only"
        ]
    },

}

def format_input(dir,prefix):
    type_to_file = {}
    dir_name = str(pathlib.Path(dir).stem)
    print(dir_name)
    input_folder = os.path.join(dir,"best_predict_only_folder")
    csvs = pathlib.Path(input_folder).rglob('*.csv')
    for csv in csvs:
        name = str(csv.stem)
        embedding_type = re.sub(f"{prefix}_","",name)
        type_to_file[embedding_type] = csv
    return type_to_file
def run_analysis_on_dir(dir,prefix):
    type_to_file = format_input(dir,prefix)
    print(type_to_file)
    dir_name = pathlib.Path(dir).stem
    output_folder = os.path.join(dir,"analysis")
    os.makedirs(output_folder,exist_ok=True)
    for key in SUBSETS.keys():
        output_subset_dir = os.path.join(output_folder,key)
        os.makedirs(output_subset_dir,exist_ok=True)
        
        subset_dict = SUBSETS[key]
        baseline = subset_dict["baseline"]
        csv = type_to_file[baseline]
        df = pd.read_csv(csv)
        p_bins = list(pd.qcut(np.array(list(df["train_max_pident"])), 8,retbins=True)[1])
        print(p_bins)
        f_bins = list(pd.qcut(np.array(df["train_max_fident_cov"]), 8,retbins=True)[1])
        predictions = []
        for embedding_type in subset_dict["contains"]:
            if embedding_type in type_to_file:
                predictions.append(f"{embedding_type}={type_to_file[embedding_type]}")
        prefix_out = str(os.path.join(output_subset_dir,f"{dir_name}_fident_cov"))
        print(prefix_out)
        analysis_args = Namespace({"identity":csv, "predictions":predictions,"baseline":baseline,"out_prefix":prefix_out,"id_col":"identifier","identity_col":"train_max_fident_cov","cluster_col":"cluster","score":"margin","score_normalise":"rank","slope_acc_tolerance":0.05,"bins":f_bins,"n_boot":2000,"seed":0,"title":f"{dir_name}_fident_cov"})
        scripts.stratified_analysis.main(analysis_args)
        prefix_out = str(os.path.join(output_subset_dir,f"{dir_name}_pident_cov"))
        print(prefix_out)
        analysis_args = Namespace({"identity":csv, "predictions":predictions,"baseline":baseline,"out_prefix":prefix_out,"id_col":"identifier","identity_col":"train_max_pident","cluster_col":"cluster","score":"margin","score_normalise":"rank","slope_acc_tolerance":0.05,"bins":p_bins,"n_boot":2000,"seed":0,"title":f"{dir_name}_pident_cov"})
        scripts.stratified_analysis.main(analysis_args)
if __name__ == '__main__':
    #python classifier_runner.py --entity SPE_per_exon --nn_model Transformer --h5 input_data/SPE/SerProtEuk_per_exon.h5 --csv input_data/SPE/SerProtEuk.csv
    parser = argparse.ArgumentParser(description="Train or optimize a multi-class SCPP classifier.")
    parser.add_argument("--dir", required=True, help="Path to dataset directory")
    parser.add_argument("--prefix_in", required=True, help="File Prefix")
    args = parser.parse_args()
    run_analysis_on_dir(args.dir,args.prefix_in)