import argparse
import csv
import sys
import os
from collections import defaultdict
import time 
from Bio import SeqIO
from rapidfuzz.distance import Levenshtein
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
ENABLE_DEBUG_PRINTS = False
RAPIDFUZZ_AVAILABLE = True
DATASETS_CMD = ["wsl", "--exec", "/mnt/c/Users/ladew/Documents/datasets_path/datasets"]
def debug_print(*args, **kwargs):
    if ENABLE_DEBUG_PRINTS:
        print("  DEBUG:", *args, **kwargs, file=sys.stderr)
def save_fasta(dict_to_save,file_name):
    with open(file_name, "wb") as file:
        for key, item in dict_to_save.items():
            seq = item["seq"]
            file.write(bytes(f">{key}\n{seq}\n", "UTF-8"))
def save_csv(dict_to_save,file_name):
    with open(file_name, "w") as file:
        file.write("identifier,gene,species")
        for key, item in dict_to_save.items():
            file.write("\n"+",".join(item["meta"]))
def save_csv_splits(dict_to_save,file_name):
    with open(file_name, "w") as file:
        file.write("identifier,gene,species,cut_pos")
        for key, item in dict_to_save.items():
            file.write("\n"+",".join(item["meta"][:-1])+f",\"{item["meta"][-1]}\"")
def save_acc(dict_to_save,file_name):
    with open(file_name, "w") as file:
        for key, item in dict_to_save.items():
            file.write(item["meta"][0]+"\n")
def download_entrez(main_family_dict,output_folder="output_entrez",taxon = "animals",batch_size=32 ):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    Entrez.email = "ladewigsean@gmail.com"
    Entrez.api_key = "22c1495b8546b4ba67e46ca57de28aba5b09"
    parsed_data = {}
    main_family = main_family_dict["main_family"]
    prefix = main_family_dict["prefix"]
    families = main_family_dict["families"]
    for family in families:
        family_name = family["family"]
        subfamilies = family["subfamily"]
        for subfamily in subfamilies:
            symbol = prefix + family_name + subfamily
            protein_file = os.path.join(output_folder,(symbol+".fasta"))
            print("Starting: " + symbol )
            if not os.path.isfile(protein_file):
                print("downloading Entrez: " + symbol )
                
                handle = Entrez.esearch(db="gene",retmax = 1000, term=f"(({main_family} {family_name}{subfamily}*[Gene/Protein Name]) OR (({main_family}[Gene/Protein Name]) AND (subfamily {subfamily}[Gene/Protein Name] AND family {family_name}[Gene/Protein Name]) ) ) AND animals[porgn]",usehistory="y", idtype="acc")
                search_results = Entrez.read(handle)
                handle.close()
                count = int(search_results["Count"])
                print(f"Genes found: {count}")
                if count == 0:
                    print(f"No {symbol} genes found, skipping...")
                    continue
                links = set({})
                for start in range(0, count, batch_size):
                    end = min(count, start + batch_size)
                    attempts = 3
                    success = False
                    while attempts >0 and not success:
                        try:
                            handle = Entrez.elink(
                                dbfrom="gene",
                                retstart=start,
                                retmax=batch_size,
                                webenv=search_results["WebEnv"],
                                query_key=search_results["QueryKey"],
                                linkname="gene_protein")
                        
                            linked = Entrez.read(handle)
                            handle.close()
                            links.update([link["Id"] for link in linked[0]["LinkSetDb"][0]["Link"]])
                            success = True
                        except:
                            attempts = attempts -1
                            print(f"failed batch {start}-{end}\nattempts remaing: {attempts}")
                    if attempts == 0:
                        print (f"fully failed batch {start}-{end}, Skipping batch...")
                    elif attempts < 3:
                        print(f"batch {start}-{end} completed after {4-attempts} attempts" )
                links = list(links)
                search_results = Entrez.read(Entrez.epost("protein", id=",".join(links)))
                count = len(links)
                print(f"Protein IDs found: {count}")
                with open(protein_file, "w") as file:
                    for start in range(0, count, batch_size):
                        end = min(count, start + batch_size)
                        attempts = 3
                        success = False
                        while attempts >0 and not success:
                            try:
                                handle = Entrez.efetch(
                                    db="protein",  
                                    rettype="fasta", 
                                    retmode="text",
                                    retstart=start,
                                    retmax=batch_size,
                                    webenv=search_results["WebEnv"],
                                    query_key=search_results["QueryKey"])
                                fasta = handle.read()
                                handle.close()
                                file.write(fasta)
                                success = True
                            except:
                                attempts = attempts -1
                                print(f"failed batch {start}-{end}\nattempts remaing: {attempts}")
                        if attempts == 0:
                            print (f"fully failed batch {start}-{end}, Skipping batch...")
                        elif attempts < 3:
                            print(f"batch {start}-{end} completed after {4-attempts} attempts" )
            current_data = parse_symbol_grab(protein_file,symbol,check_symbol=False)
            parsed_data[symbol] = (current_data)
        
    return parsed_data
def download_entrez_rna(main_family_dict,output_folder="output_entrez_rna",taxon = "animals",batch_size=32 ):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    Entrez.email = "ladewigsean@gmail.com"
    Entrez.api_key = "22c1495b8546b4ba67e46ca57de28aba5b09"
    parsed_data = {}
    main_family = main_family_dict["main_family"]
    prefix = main_family_dict["prefix"]
    families = main_family_dict["families"]
    for family in families:
        family_name = family["family"]
        subfamilies = family["subfamily"]
        for subfamily in subfamilies:
            symbol = prefix + family_name + subfamily
            protein_file = os.path.join(output_folder,(symbol+".fasta"))
            print("Starting: " + symbol )
            if not os.path.isfile(protein_file):
                print("downloading Entrez: " + symbol )
                
                handle = Entrez.esearch(db="gene",retmax = 1000, term=f"(({main_family} {family_name}{subfamily}*[Gene/Protein Name]) OR (({main_family}[Gene/Protein Name]) AND (subfamily {subfamily}[Gene/Protein Name] AND family {family_name}[Gene/Protein Name]) ) ) AND animals[porgn]",usehistory="y", idtype="acc")
                search_results = Entrez.read(handle)
                handle.close()
                count = int(search_results["Count"])
                print(f"Genes found: {count}")
                if count == 0:
                    print(f"No {symbol} genes found, skipping...")
                    continue
                links = set({})
                for start in range(0, count, batch_size):
                    end = min(count, start + batch_size)
                    attempts = 3
                    success = False
                    while attempts >0 and not success:
                        try:
                            handle = Entrez.elink(
                                dbfrom="gene",
                                retstart=start,
                                retmax=batch_size,
                                webenv=search_results["WebEnv"],
                                query_key=search_results["QueryKey"],
                                linkname="gene_nuccore_refseqrna")
                        
                            linked = Entrez.read(handle)
                            handle.close()
                            links.update([link["Id"] for link in linked[0]["LinkSetDb"][0]["Link"]])
                            success = True
                        except:
                            attempts = attempts -1
                            print(f"failed batch {start}-{end}\nattempts remaing: {attempts}")
                    if attempts == 0:
                        print (f"fully failed batch {start}-{end}, Skipping batch...")
                    elif attempts < 3:
                        print(f"batch {start}-{end} completed after {4-attempts} attempts" )
                links = list(links)
                search_results = Entrez.read(Entrez.epost("nuccore", id=",".join(links)))
                count = len(links)
                print(f"RNA IDs found: {count}")
                with open(protein_file, "w") as file:
                    for start in range(0, count, batch_size):
                        end = min(count, start + batch_size)
                        attempts = 3
                        success = False
                        while attempts >0 and not success:
                            try:
                                handle = Entrez.efetch(
                                    db="nuccore",  
                                    rettype="fasta", 
                                    retmode="text",
                                    retstart=start,
                                    retmax=batch_size,
                                    webenv=search_results["WebEnv"],
                                    query_key=search_results["QueryKey"])
                                fasta = handle.read()
                                handle.close()
                                file.write(fasta)
                                success = True
                            except:
                                attempts = attempts -1
                                print(f"failed batch {start}-{end}\nattempts remaing: {attempts}")
                        if attempts == 0:
                            print (f"fully failed batch {start}-{end}, Skipping batch...")
                        elif attempts < 3:
                            print(f"batch {start}-{end} completed after {4-attempts} attempts" )
            current_data = parse_symbol_grab(protein_file,symbol,check_symbol=False)
            parsed_data[symbol] = (current_data)
        
    return parsed_data
def parse_symbol_grab(file,gene, file_format="fasta",check_symbol= True):
    parsed_data = {}
    for seq_record in SeqIO.parse(file, file_format):
        
        if (check_symbol and not re.search(gene, seq_record.description, re.IGNORECASE))or str(seq_record.seq)[0]!="M":
            continue
        regex = re.search(r"organism=([^\]]*)",seq_record.description)
        species = "unknown"
        if regex:
            species = regex.group(1)      
        else:
            #sometimes they I added some uniprot when less than 50 o_o
            regex = re.search(r"OS=([^ ]* [^ ]*)",seq_record.description)
            if regex:
                species = regex.group(1)
            else:
                #the nested if never fails
                regex = re.search(r"\[([^\]]*)\]",seq_record.description)
                if regex:
                    species = regex.group(1)
                else:
                    regex = re.search(r"Predicted: ([^ ] [^ ]) ",seq_record.description)
                    if regex:
                        species = regex.group(1)
        if "|" in seq_record.id:
            seq_record.id = re.search(r"\|([^\|]*)\|",seq_record.id).group(1)
        if "_" not in seq_record.id:
            continue
        if len(str(seq_record.seq)) <20:
            print(seq_record.id)
            continue 
        parsed_data[seq_record.id] = {'meta': [seq_record.id, gene, species], 'seq': str(seq_record.seq), 'full_header': seq_record.description}
    return parsed_data 

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
    save_fasta(data,temp_fasta)
    save_csv_splits(data,temp_csv)
    
    generate_fasta_with_splits(temp_fasta,temp_csv,temp_input)
    temp_output = f"temp_{p_identity}"
    if mode == "diamond":
        temp_output = temp_output +".tsv"
        subprocess.run(["diamond","cluster","-d",temp_input,"-o",temp_output,"--approx-id",p_identity,"-M",ram])
        df = pd.read_csv(temp_output,sep="\t",names=["cluster","entry"],header=None)
        if os.path.isfile(temp_output):
            os.remove(temp_output)
    elif mode == "mmseqs":
        p_identity = str(int(p_identity)/100)
        subprocess.run(["wsl","--exec","mmseqs","easy-cluster",temp_input,temp_output,"temp/","--min-seq-id",p_identity,"-c","0.8","--cov-mode","0"])
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

def read_metadata_fasta(fasta,metadata,filter_keep = None):
    #really shit, need to remake open fasta like 50 times. should make dict for ID-> symbol and go through once
    #like omega ass, just lazy
    df = pd.read_csv(metadata)
    parsed_data = {}
    symbols = set(df["gene"])
    if filter_keep:
        symbols = symbols.intersection(set(filter_keep))
    for symbol in symbols:
        parsed_data[symbol]= {}
    
    for seq_record in SeqIO.parse(fasta, "fasta"):
        row = df[df["identifier"]==seq_record.id]
        if row.shape[0] != 1:
            print(f"{seq_record.id} {row.size}")
            continue
        row = row.iloc[0]
        parsed_data[row["gene"]][seq_record.id] = {'meta': [seq_record.id, row["gene"], row["species"],row["cut_pos"]], 'seq': str(seq_record.seq), 'full_header': seq_record.description}
    return parsed_data
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


def run_datasets_for_accession(accession, json_folder):
    output_json = os.path.join(json_folder, f"{accession}.json")
    
    if os.path.exists(output_json):
        print(f"JSON file {output_json} already exists. Skipping.")
        return accession, output_json

    cmd = DATASETS_CMD+["summary", "gene", "accession", accession, "--report", "product"]
    print(f"Running command for {accession}: {' '.join(cmd)}")
    with open(output_json, "w") as outfh:
        subprocess.run(cmd, stdout=outfh)
    return accession, output_json
def process_json(json_file,accession):
    with open(json_file) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"    -> WARNING: Could not parse JSON, file may be empty: {json_file}")
            return []

    reports = data.get("reports", [])
    results = []
    if not reports: 
        print(f"JSON file {json_file} has zero reports")
        return []
    elif len(reports) > 1:
        print(f"JSON file {json_file} has multiple reports")
    report =reports[0]
    product_info = report.get("product", {})        
    taxid = product_info.get("tax_id")        
    gene_symbol = product_info.get("symbol", "unknown").lower()
    transcripts = product_info.get("transcripts", [])
    if len(transcripts) > 1:
        print(f"JSON file {json_file} has multiple transcripts")
    elif len(transcripts)==0:
        print(f"JSON file {json_file} has zero transcripts")
        return []
    cut_pos_cds = []
    for transcript in transcripts:
        acc_version = transcript.get("accession_version")
        prot_accession = transcript.get("protein", {}).get("accession_version")
        if prot_accession != accession:
            print(f"The current protein({prot_accession}) accession does not match searched protein accession({accession}),skipping")
            continue
        glocs = transcript.get("genomic_locations", [])
        if not glocs or not glocs[0].get("exons"):
            print(f"JSON file {json_file} failed to find exons")
            return []
        exons_sorted = sorted(glocs[0]["exons"], key=lambda e: int(e.get("order", 0)))
        cut_pos = []
        current_length = 0
        for exon in exons_sorted:
            length = abs(int(exon["begin"])-int(exon["end"]))+1
            current_length = length +current_length
            cut_pos.append(current_length)
        cds_info = transcript.get("cds", {}).get("range", [{}])[0]
        cds_start = int(cds_info["begin"])-1
        cds_end_relative = int(cds_info["end"])-2-cds_start
        cut_pos_cds = []
        for pos in cut_pos:
            relative_pos = pos-cds_start 
            if relative_pos >= cds_end_relative:
                break
            if relative_pos > 0:
                cut_pos_cds.append(str(relative_pos))
        break
    if cut_pos_cds == []:
        print(f"could not find correct protein accesion in {json_file}")
    return cut_pos_cds
            
            
            
def add_cds_ranges(metadata_csv,output_csv,json_folder):
    df = pd.read_csv(metadata_csv)
    cut_pos_column = []
    accessions = list(df["identifier"])
    for accession in accessions:
        _,json_file = run_datasets_for_accession(accession,json_folder)
        cut_pos_cds = process_json(json_file,accession)
        cut_pos_column.append(f"[{",".join(cut_pos_cds)}]")
    df["cut_pos"] = cut_pos_column
    df.to_csv(output_csv,index=False)
def convert_to_accession(metadata_csv, output_csv):
    df = pd.read_csv(metadata_csv)
    mapper = ProtMapper()
    #print(ProtMapper()._supported_dbs)     
    accession_list = df["identifier"]
    to_map_lst = []
    for identifier in accession_list:
        if "_" not in identifier:
            #to_map_lst.append(identifier[:-2]) 
            to_map_lst.append(identifier)
        
    dbs = ["UniProtKB_AC-ID"]
    for db in dbs:
        print("DB: " + db)
        print("IDs to Map: " + str(len(to_map_lst)) )
        result,to_map_lst = mapper.get(to_map_lst,from_db =db, to_db="RefSeq_Protein")
        print(result)
        ##print(to_map_lst)
        result = result.drop_duplicates(["From"])
        wide_dict = result.to_dict("index")
        transformation_dict = {}
        for key in wide_dict.keys():
            row = wide_dict[key]
            transformation_dict[row["From"]] = row["To"]
        df = df.replace({"replace":transformation_dict})
    df.to_csv(output_csv,index=False) 
def split_string_to_list(splits_string):
    return json.loads(splits_string)
def generate_splits_for_seq(sequence, splits_string):
    splits_list = split_string_to_list(splits_string)
    new_sequence = ""
    last_split = 0
    last_split_prot = 0
    current_offset = 0

    for split in splits_list:
        length = split-last_split
        split_prot = math.ceil(split/3)
        ending_phase = split %3
        segment = sequence[last_split_prot:split_prot-1]
        last_residue = sequence[split_prot-1]
        segment_end = ending_phase*"*"+last_residue+(3-ending_phase)*"*"
        last_split = split
        last_split_prot = split_prot
        new_sequence = new_sequence+segment+segment_end
    new_sequence = new_sequence+sequence[last_split_prot:]
    return new_sequence
       
        
def generate_fasta_with_splits(fasta,metadata_csv,output_fasta):
    df = pd.read_csv(metadata_csv)
    with open(output_fasta, "wb") as file:
        for seq_record in SeqIO.parse(fasta, "fasta"):
            row = df[df["identifier"]==seq_record.id]
            if row.size == 0:
                continue
            row = row.iloc[0]
            seq = generate_splits_for_seq(seq_record.seq,row["cut_pos"])
            file.write(bytes(f">{seq_record.id}\n{seq}\n", "UTF-8"))
def length_df(flat_dict):
    dict_for_df = {}
    for key, item in flat_dict.items():
        length_seq = len(item["seq"])
        length_exon = len(split_string_to_list(item["meta"][-1]))+1
        gene = item["meta"][1]
        dict_for_df[key] = [gene ,length_seq,length_exon]
    df = pd.DataFrame.from_dict(dict_for_df, orient="index", columns=["Gene", "length_seq", "length_exon"])
    return df
def filter_outliers_all_graphic(parsed_dict,prefix = "boxplot",whisk = 2):
    parsed_dict = flatten_dict(parsed_dict)
    
    df = length_df(parsed_dict)

    plt.close()
    sns.set_theme(rc={'figure.figsize':(40,8)})
    sns.set_style('whitegrid')
    plt.xticks(rotation=30)
    ax= sns.boxplot(x='Gene',y='length_seq',data=df,whis=whisk,medianprops={"color": "r", "linewidth": 2})
    #ax = sns.stripplot(x="Gene", y="length_seq",data=df)
    plt.savefig(prefix + "_seq_length.png")
    plt.close()
    #just in case i need to reformat
    sns.set_style('whitegrid')
    plt.xticks(rotation=30)
    ax= sns.boxplot(x='Gene',y='length_exon',data=df,whis=whisk,medianprops={"color": "r", "linewidth": 2})
    #ax = sns.stripplot(x="Gene", y="length_exon",data=df)
    plt.savefig(prefix + "_exon_length.png")
    plt.close()
    sns.set_style('whitegrid')
    plt.xticks(rotation=30)
    ax = sns.countplot(x='Gene',data=df)
    
    ax.bar_label(ax.containers[0])
    plt.savefig(prefix + "_counts.png")
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
def splice_aware_filter_attmept1(dict1,prefix, ident_threshold="80",mode="diamond",whisk = 2,min_nummer = 60):
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
    plt.savefig(prefix + "_counts_after_filtering.png")
    with open(prefix+"_report.txt","w") as file:
        file.write("\n".join(short_report))
        file.write("\n".join(log))
    return combined_all_filtered 

with open('CYP_input.json',"r") as f:
    main_family_dict = json.load(f)
prefix = "CYP_PA_Attempt5"
download = False

if download:
    parsed_data = download_entrez(main_family_dict)
    
    #parsed_data =download_entrez_rna(main_family_dict)
    midway_filter = flatten_dict(parsed_data)
    
    save_fasta(midway_filter,f"{prefix}_no_filter.fasta")
    save_csv(midway_filter,f"{prefix}_no_filter.csv")
    save_acc(midway_filter,f"{prefix}_acc_no_filter.txt")
    add_cds_ranges(f"{prefix}_no_filter.csv",f"{prefix}_no_filter_cuts.csv","json_folder")
"""
else:
    parsed_data = read_metadata_fasta(f"{prefix}_no_filter.fasta",f"{prefix}_no_filter.csv")
    parsed_data = flatten_dict(parsed_data)
"""
#convert_to_accession(f"{prefix}_no_filter.csv",f"{prefix}_no_filter_entrez.csv")
prefix = "CYP_PA_Attempt4"
parsed_data = read_metadata_fasta(f"{prefix}_no_filter.fasta",f"{prefix}_no_filter_cuts.csv")
prefix = "CYP_PA_Attempt5"
whisk = 3
filter_outliers_all_graphic(parsed_data,prefix=prefix,whisk=whisk)

combined_all_filtered = splice_aware_filter_attmept1(parsed_data,prefix, ident_threshold="85",mode="mmseqs",whisk=whisk)
save_fasta(combined_all_filtered,f"{prefix}.fasta")
save_csv_splits(combined_all_filtered,f"{prefix}.csv")
