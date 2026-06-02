import argparse
import csv
import sys
import os
from dotenv import load_dotenv
from collections import defaultdict
import time 
from Bio import SeqIO
import json
import subprocess
import zipfile
import os
import shutil
import re
import json
import pandas as pd 
import numpy as np
from Bio import Entrez
from UniProtMapper import ProtMapper
import math
import seaborn as sns
import matplotlib.pyplot as plt
import scripts.grab_data

DATASETS_CMD = ["/mnt/c/Users/ladew/Documents/datasets_path/datasets"]



"""
def thin_data_original(data, ram = "32G",p_identity="50",mode = "diamond"):
    print("Thinning data")
    print(f"length before: {len(data)}")
    temp_input = f"temp_{p_identity}_all_to_all.fasta"
    save_fasta(data,temp_input)
    temp_output = f"temp_{p_identity}_all_to_all"
    if mode == "diamond":
        temp_output = temp_output +".tsv"
        subprocess.run(["diamond","cluster","-d",temp_input,"-o",temp_output,"--approx-id",p_identity,"-M",ram])
        df = pd.read_csv(temp_output,sep="\t",names=["cluster","entry"],header=None)
        if os.path.isfile(temp_output):
            os.remove(temp_output)
    elif mode == "mmseqs":
        p_identity = str(int(p_identity)/100)
        subprocess.run(["wsl","--exec","mmseqs","easy-cluster",temp_input,temp_output,"temp/","--min-seq-id",p_identity,"-c","0.8","--cov-mode","0"])
        df = pd.read_csv(temp_output+"_cluster.tsv",sep="\t",names=["cluster","entry"],header=None)
        if os.path.isfile(temp_output+"_cluster.tsv"):
            os.remove(temp_output+"_cluster.tsv")
        #if os.path.isfile(temp_output+"_rep_seq.fasta"):
            #os.remove(temp_output+"_rep_seq.fasta")
        if os.path.isfile(temp_output+"_all_seqs.fasta"):
            os.remove(temp_output+"_all_seqs.fasta")
    
    keys_to_keep = set(df["cluster"]) 
    filtered_data = {key: data[key] for key in keys_to_keep}
    print(f"length after: {len(filtered_data)}")
    if os.path.isfile(temp_input):
        os.remove(temp_input)
    if os.path.isfile(temp_output):
        os.remove(temp_output)
    return filtered_data
"""
def thin_data(data, ram = "32G",p_identity="50",mode = "diamond"):
    print("Thinning data")
    print(f"length before: {len(data)}")
    #so scuffed
    temp_input = f"temp_{p_identity}_splits.fasta"
    temp_fasta = f"temp_{p_identity}.fasta"
    temp_csv = f"temp_splits.csv"
    scripts.grab_data.save_fasta(data,temp_fasta)
    scripts.grab_data.save_csv_splits(data,temp_csv)
    
    scripts.grab_data.generate_fasta_with_splits(temp_fasta,temp_csv,temp_input)
    temp_output = f"temp_{p_identity}"
    if mode == "diamond":
        temp_output = temp_output +".tsv"
        subprocess.run(["diamond","cluster","-d",temp_input,"-o",temp_output,"--approx-id",p_identity,"-M",ram])
        df = pd.read_csv(temp_output,sep="\t",names=["cluster","entry"],header=None)
        if os.path.isfile(temp_output):
            os.remove(temp_output)
    elif mode == "mmseqs":
        p_identity = str(int(p_identity)/100)
        subprocess.run(["mmseqs","easy-cluster",temp_input,temp_output,"temp/","--min-seq-id",p_identity,"-c","0.8","--cov-mode","0"])
        """
        with open(temp_output+"_cluster.tsv", "r") as file:
            tsv = file.read().replace("\n\t","\t")
        with open(temp_output+"_cluster.tsv", "w") as file:
            file.write(tsv)
        """
        df = pd.read_csv(temp_output+"_cluster.tsv",sep="\t",names=["cluster","entry"],header=None)
        if os.path.isfile(temp_output+"_cluster.tsv"):
            os.remove(temp_output+"_cluster.tsv")
        if os.path.isfile(temp_output+"_rep_seq.fasta"):
            os.remove(temp_output+"_rep_seq.fasta")
        if os.path.isfile(temp_output+"_all_seqs.fasta"):
            os.remove(temp_output+"_all_seqs.fasta")
    
    keys_to_keep = set(df["cluster"]) 
    filtered_data = {key: data[key] for key in keys_to_keep}
    print(f"length after: {len(filtered_data)}")
    if os.path.isfile(temp_input):
        os.remove(temp_input)
    if os.path.isfile(temp_output):
        os.remove(temp_output)
    return filtered_data


def flatten_dict(input_dict):
    output_dict ={}
    for key in input_dict.keys():
        output_dict.update(input_dict[key]) 
    return output_dict
        
def repair_data(csv, dict_, output_csv):
    original = pd.read_csv(csv)
    id_list =  list(original["identifier"])
    species_list = [dict_[key]["meta"][2] for key in id_list]
    original["species"] = species_list
    original.to_csv(output_csv, index=False)



def length_df(flat_dict):
    dict_for_df = {}
    for key, item in flat_dict.items():
        length_seq = len(item["seq"])
        length_exon = len(scripts.grab_data.split_string_to_list(item["meta"][-1]))+1
        gene = item["meta"][1]
        dict_for_df[key] = [gene ,length_seq,length_exon]
    df = pd.DataFrame.from_dict(dict_for_df, orient="index", columns=["Gene", "length_seq", "length_exon"])
    return df
def filter_outliers_all_graphic(parsed_dict,folder,prefix = "boxplot",whisk = 2):
    parsed_dict = flatten_dict(parsed_dict)
    
    df = length_df(parsed_dict)

    plt.close()
    sns.set_theme(rc={'figure.figsize':(40,8)})
    sns.set_style('whitegrid')
    plt.xticks(rotation=30)
    ax= sns.boxplot(x='Gene',y='length_seq',data=df,whis=whisk,medianprops={"color": "r", "linewidth": 2})
    #ax = sns.stripplot(x="Gene", y="length_seq",data=df)
    plt.savefig(os.path.join(folder,prefix + "_seq_length.png"))
    plt.close()
    #just in case i need to reformat
    sns.set_style('whitegrid')
    plt.xticks(rotation=30)
    ax= sns.boxplot(x='Gene',y='length_exon',data=df,whis=whisk,medianprops={"color": "r", "linewidth": 2})
    #ax = sns.stripplot(x="Gene", y="length_exon",data=df)
    plt.savefig(os.path.join(folder,prefix + "_exon_length.png"))
    plt.close()
    sns.set_style('whitegrid')
    plt.xticks(rotation=30)
    ax = sns.countplot(x='Gene',data=df)
    
    ax.bar_label(ax.containers[0])
    plt.savefig(os.path.join(folder,prefix + "_counts.png"))
def filter_outliers_length_data(flat_dict,whisk = 2):
    df = length_df(flat_dict)
    q75_s, q25_s = np.percentile(df["length_seq"], [75 ,25])
    iqr_s = max(q75_s - q25_s,0.5)
    bottom_bound_s = q25_s-(iqr_s*whisk)
    top_bound_s = q75_s+(iqr_s*whisk)
    q75_e, q25_e = np.percentile(df["length_exon"], [75 ,25])
    iqr_e = max(q75_e - q25_e,.5)
    bottom_bound_e = q25_e-(iqr_e*whisk)
    top_bound_e = q75_e+(iqr_e*whisk)
    df = df[(df["length_seq"] >= bottom_bound_s)&(df["length_seq"] <= top_bound_s) & (df["length_exon"] >= bottom_bound_e)&(df["length_exon"] <= top_bound_e)]
    filtered_data = {key: flat_dict[key] for key in df.index}
    
    return filtered_data
def splice_aware_filter_attmept1(dict1,prefix,folder, ident_threshold="80",mode="diamond",whisk = 2,min_nummer = 60):
    log=["\n\n**********************\nComplete Positive Dataset Report\n**********************\n"]
    short_report=["**********\nShort Report\n**********\n"]
    combined_all ={}
    combined_all_filtered = {}
    midway_filter={}
    sorted_symbols = sorted(list(dict1.keys()))
    unique_num = []
    outliers_num= []
    clustering_num = []
    for symbol in sorted_symbols:
        print(f"filtering {symbol}")
        log.append(f"********\n{symbol}\n********")
        unique_count = len(dict1[symbol])
        unique_num.append(unique_count )
        log.append(f"ncbi efetch proteins: {len(dict1[symbol])}")
        combined_all.update(dict1[symbol])
        
        filtered_combined = filter_outliers_length_data(dict1[symbol],whisk=whisk)
        outliers_num.append(len(filtered_combined))
        log.append(f"Proteins after filtering outliers: {len(filtered_combined)}")
        filtered_combined= thin_data(filtered_combined, p_identity=ident_threshold,mode = mode)
        clustering_num.append(len(filtered_combined))
        log.append(f"Proteins after clustering: {len(filtered_combined)}")
        included_symbol = "X"
        
        if len(filtered_combined) >= min_nummer:
            combined_all_filtered.update(filtered_combined)
            included_symbol = ""
            #adds emtpy line
            log.append("")
        else:
            log.append(f"Not enough data in class, the entire class will be removed from final data({len(filtered_combined)} < {min_nummer})\n")
        short_report.append("\t".join([symbol, (f"{unique_count}-->{len(filtered_combined)}".ljust(11)), f"{ident_threshold}%",included_symbol]))
        
    df_counts = pd.DataFrame.from_dict({"Gene":sorted_symbols,'Original':unique_num, 'Post_Outlier_removal':outliers_num, 'Post_clustering':clustering_num})
    #prob could have made original data melted but this is simpler
    tidy = df_counts.melt(id_vars='Gene')
    plt.close()
    sns.set_theme(rc={'figure.figsize':(40,8)})
    sns.set_style('whitegrid')
    plt.xticks(rotation=30)
    ax = sns.barplot(x="Gene",y = "value",hue="variable",data=tidy)
    for container in ax.containers:
        ax.bar_label(container,fontsize="x-small",rotation=40 )
    ax.set(ylabel="Count")
    plt.savefig(os.path.join(folder,prefix + "_counts_after_filtering.png"))
    with open(os.path.join(folder,prefix+"_report.txt"),"w") as file:
        file.write("\n".join(short_report))
        file.write("\n".join(log))
    return combined_all_filtered 
"""
with open('CYP_input.json',"r") as f:
    main_family_dict = json.load(f)
prefix = "CYP_PA_Attempt5"
download = False

if download:
    parsed_data = download_entrez(main_family_dict)
    
    #parsed_data =download_entrez_rna(main_family_dict)
    no_filter = flatten_dict(parsed_data)
    
    scripts.grab_data.save_fasta(no_filter,f"{prefix}_no_filter.fasta")
    scripts.grab_data.save_csv(no_filter,f"{prefix}_no_filter.csv")
    scripts.grab_data.save_acc(no_filter,f"{prefix}_acc_no_filter.txt")
    scripts.grab_data.add_cds_ranges(f"{prefix}_no_filter.csv",f"{prefix}_no_filter_cuts.csv","json_folder")

else:
    parsed_data = read_metadata_fasta(f"{prefix}_no_filter.fasta",f"{prefix}_no_filter.csv")
    parsed_data = flatten_dict(parsed_data)

#convert_to_accession(f"{prefix}_no_filter.csv",f"{prefix}_no_filter_entrez.csv")

parsed_data = scripts.grab_data.read_metadata_fasta(f"{prefix}_no_filter.fasta",f"{prefix}_no_filter_cuts.csv")
prefix = "CYP_PA_Attempt5"
whisk = 3
filter_outliers_all_graphic(parsed_data,prefix=prefix,whisk=whisk)

combined_all_filtered = splice_aware_filter_attmept1(parsed_data,prefix, ident_threshold="85",mode="mmseqs",whisk=whisk)
scripts.grab_data.save_fasta(combined_all_filtered,f"{prefix}.fasta")
scripts.grab_data.save_csv_splits(combined_all_filtered,f"{prefix}.csv")
"""