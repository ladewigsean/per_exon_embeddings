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
def filter_self(parsed_data, max_levenshtein_distance=4):
    if not parsed_data:
        return set(), 0, 0, 0
    initial_count = len(parsed_data) # Count after parser's duplicate accession removal
    print(f"\nStep 2: Filtering {initial_count} unique sequences from parser...")

    # Filter identical
    print("  [Filter 1/3] Identifying identical sequences...")
    start_time = time.time()
    sequence_to_accessions = defaultdict(list)
    for acc, data in parsed_data.items():
        if data.get('seq'):
            sequence_to_accessions[data['seq']].append(acc)
        else:
             debug_print(f"WARN: Skipping accession '{acc}' in identity check - missing sequence.")

    debug_print(f"Identical sequence check: {len(sequence_to_accessions)} unique sequences found.")

    accessions_after_identity_check = set()
    num_removed_identical = 0
    for sequence, accessions in sequence_to_accessions.items():
        if len(accessions) > 1:
            debug_print(f"Found {len(accessions)} accessions for sequence (len {len(sequence)}): {accessions}")
            accessions.sort()
            kept_acc = accessions[0]
            accessions_after_identity_check.add(kept_acc)
            num_removed_identical += (len(accessions) - 1)
            debug_print(f" -> Keeping '{kept_acc}', removing {len(accessions)-1}.")
        else:
            accessions_after_identity_check.add(accessions[0])

    count_after_identity = len(accessions_after_identity_check)
    print(f"  - Removed {num_removed_identical} identical sequences ({initial_count} -> {count_after_identity}). Time: {time.time() - start_time:.2f}s")
    # num_removed_identical > 0 means different accessions had the same sequence; a redundant check 

    # Filter by Levenshtein distance
    num_removed_near_identical = 0
    accessions_after_near_identity_check = accessions_after_identity_check # Default if skipping

    if max_levenshtein_distance and max_levenshtein_distance > 0:
        if not RAPIDFUZZ_AVAILABLE:
            print(f"  [Filter 2/3] Skipping near-identical check: 'rapidfuzz' library not found.")
        else:
            print(f"  [Filter 2/3] Identifying near-identical sequences (Levenshtein <= {max_levenshtein_distance}) among remaining {count_after_identity}...")
            start_time = time.time()
            unique_seq_data = []
            for acc in accessions_after_identity_check:
                if acc in parsed_data and parsed_data[acc].get('seq'):
                    unique_seq_data.append((acc, parsed_data[acc]['seq']))
                else:
                     debug_print(f"WARN: Skipping accession '{acc}' in near-identity check - missing sequence data.")
            unique_seq_data.sort(key=lambda x: len(x[1]), reverse=True)
            representatives = []
            accessions_to_remove_near_identity = set()
            num_compared = 0
            for acc, seq in unique_seq_data:
                if acc in accessions_to_remove_near_identity: continue
                is_near_duplicate = False
                for rep_acc, rep_seq in representatives:
                    num_compared += 1
                    if abs(len(seq) - len(rep_seq)) > max_levenshtein_distance: continue
                    distance = Levenshtein.distance(seq, rep_seq, score_cutoff=max_levenshtein_distance)
                    if distance <= max_levenshtein_distance:
                        accessions_to_remove_near_identity.add(acc)
                        is_near_duplicate = True
                        debug_print(f"Near-Identical: Removing '{acc}' (dist {distance} to '{rep_acc}')")
                        break
                if not is_near_duplicate:
                    representatives.append((acc, seq))
            accessions_after_near_identity_check = accessions_after_identity_check - accessions_to_remove_near_identity
            num_removed_near_identical = len(accessions_to_remove_near_identity)
            count_after_near_identity = len(accessions_after_near_identity_check)
            print(f"  - Removed {num_removed_near_identical} near-identical sequences ({count_after_identity} -> {count_after_near_identity}). Compared pairs vs reps: {num_compared}. Time: {time.time() - start_time:.2f}s")
    else:
         print(f"  [Filter 2/3] Skipping near-identical sequence check.")


    # Filter fragments
    print(f"  [Filter 3/3] Identifying fragment sequences among the remaining {len(accessions_after_near_identity_check)}...")
    start_time = time.time()
    accessions_to_remove_substrings = set()
    candidate_accessions = list(accessions_after_near_identity_check)
    n_candidates = len(candidate_accessions)

    
    for i in range(n_candidates):
        acc1 = candidate_accessions[i]
        if acc1 in accessions_to_remove_substrings: continue
        if not (acc1 in parsed_data and parsed_data[acc1].get('seq')):
            debug_print(f"WARN: Skipping accession '{acc1}' in substring check - missing sequence data.")
            continue
        seq1 = parsed_data[acc1]['seq']
        len1 = len(seq1)
        for j in range(i + 1, n_candidates):
            acc2 = candidate_accessions[j]
            if acc2 in accessions_to_remove_substrings: continue
            if not (acc2 in parsed_data and parsed_data[acc2].get('seq')):
                 debug_print(f"WARN: Skipping comparison with '{acc2}' in substring check - missing sequence data.")
                 continue
            seq2 = parsed_data[acc2]['seq']
            len2 = len(seq2)
            if acc1 == acc2 or seq1 == seq2: continue
            try:
                if len1 < len2 and seq1 in seq2:
                    accessions_to_remove_substrings.add(acc1)
                    debug_print(f"Fragment: Removing '{acc1}' (substring of '{acc2}')")
                    break
                elif len2 < len1 and seq2 in seq1:
                    accessions_to_remove_substrings.add(acc2)
                    debug_print(f"Fragment: Removing '{acc2}' (substring of '{acc1}')")
            except Exception as e:
                print(f"  ERROR comparing substrings for {acc1} and {acc2}: {e}", file=sys.stderr)

    final_accessions_to_keep = accessions_after_near_identity_check - accessions_to_remove_substrings
    num_removed_substrings = len(accessions_to_remove_substrings)
    final_count = len(final_accessions_to_keep)
    count_before_substring = len(accessions_after_near_identity_check)
    print(f"  - Removed {num_removed_substrings} fragment sequences ({count_before_substring} -> {final_count}). Time: {time.time() - start_time:.2f}s")

    print(f"Step 2: Finished filtering.")
    return final_accessions_to_keep, num_removed_identical, num_removed_near_identical, num_removed_substrings
def filter_from(data_to_remove_from, data_to_remove, ram = "32G",p_identity="80", mode = "diamond"):
    print("Remove positive proteins")
    print(f"length before: {len(data_to_remove_from)}")
    remove_by_accessions=data_to_remove_from.copy()
    for key in data_to_remove.keys():
        if key in remove_by_accessions:
            del remove_by_accessions[key]
    to_be_clustered = remove_by_accessions | data_to_remove
    
    temp_input = f"temp_{p_identity}_all_to_all.fasta"
    save_fasta(to_be_clustered,temp_input)
    temp_output = f"temp_{p_identity}_all_to_all.tsv"
    if mode == "diamond":
        subprocess.run(["diamond","cluster","-d",temp_input,"-o",temp_output,"--approx-id",p_identity,"-M",ram])
        df = pd.read_csv(temp_output,sep="\t",names=["cluster","entry"],header=None)
    elif mode == "mmseqs":
        p_identity = str(int(p_identity)/100)
        subprocess.run(["wsl","--exec","mmseqs","easy-cluster",temp_input,temp_output,"temp/","--min-seq-id",p_identity,"-c","0.8","--cov-mode","0"])
        with open(temp_output+"_cluster.tsv", "r") as file:
            tsv = file.read().replace("\n\t","\t")
        with open(temp_output+"_cluster.tsv", "w") as file:
            file.write(tsv)
        df = pd.read_csv(temp_output+"_cluster.tsv",sep="\t",names=["cluster","entry"],header=None)
        if os.path.isfile(temp_output+"_cluster.tsv"):
            os.remove(temp_output+"_cluster.tsv")
        #if os.path.isfile(temp_output+"_rep_seq.fasta"):
            #os.remove(temp_output+"_rep_seq.fasta")
        if os.path.isfile(temp_output+"_all_seqs.fasta"):
            os.remove(temp_output+"_all_seqs.fasta")
    #this seems really messy because it is
    
    df = df[~df["cluster"].isin(set(data_to_remove.keys()))] 
    to_be_purged= set(df[df["entry"].isin(set(data_to_remove.keys()))]["cluster"])
    df = df[~df["cluster"].isin(to_be_purged)] 
    # keeping just clusters would make sense as they will be filtered out later anyways, but want to this function to only remove overlaps not thin  
    keys_to_keep = set(df["cluster"])#.union(set(df["entry"]))
    filtered_data = {key: remove_by_accessions[key] for key in keys_to_keep}
    print(f"length after: {len(filtered_data)}")
    if os.path.isfile(temp_input):
        os.remove(temp_input)
    if os.path.isfile(temp_output):
        os.remove(temp_output)
    return filtered_data
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
        """
        with open(temp_output+"_cluster.tsv", "r") as file:
            tsv = file.read().replace("\n\t","\t")
        with open(temp_output+"_cluster.tsv", "w") as file:
            file.write(tsv)
        """
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
        df_symbol = df[df["gene"]==symbol]
        for seq_record in SeqIO.parse(fasta, "fasta"):
            row = df_symbol[df_symbol["identifier"]==seq_record.id]
            if row.size == 0:
                continue
            row = row.iloc[0]
            parsed_data[symbol][seq_record.id] = {'meta': [seq_record.id, row["gene"], row["species"],row["cut_pos"]], 'seq': str(seq_record.seq), 'full_header': seq_record.description}
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
def merge_positive(dict1,report_file, ident_threshold="80",mode="diamond"):
    log=["\n\n**********************\nComplete Positive Dataset Report\n**********************\n"]
    short_report=["**********\nShort Report\n**********\n"]
    combined_all ={}
    combined_all_filtered = {}
    midway_filter={}
    sorted_symbols = sorted(set(dict1.keys()))
    for symbol in sorted_symbols:
        print(f"filtering {symbol}")
        log.append(f"********\n{symbol}\n********")
        
        log.append(f"ncbi efetch proteins: {len(dict1[symbol])}")
        
        
        combined_all.update(dict1[symbol])
        unique_accessions = filter_self(dict1[symbol],max_levenshtein_distance=0)[0]
        filtered_combined={key: dict1[symbol][key] for key in unique_accessions}
        midway_filter.update(filtered_combined)
        log.append(f"proteins after removing duplicate sequences/substrings: {len(filtered_combined)}")
        unique_count = len(filtered_combined)
        filtered_combined= thin_data(filtered_combined, p_identity=ident_threshold,mode = mode)
        log.append(f"Proteins after clustering: {len(filtered_combined)}\n")
        short_report.append("\t".join([symbol, (f"{unique_count}-->{len(filtered_combined)}".ljust(11)), f"{ident_threshold}%"]))
        combined_all_filtered.update(filtered_combined)
    with open(report_file,"w") as file:
        file.write("\n".join(short_report))
        file.write("\n".join(log))
    return combined_all,midway_filter,combined_all_filtered 

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
def splice_aware_filter_attmept1(dict1,report_file, ident_threshold="80",mode="diamond"):
    log=["\n\n**********************\nComplete Positive Dataset Report\n**********************\n"]
    short_report=["**********\nShort Report\n**********\n"]
    combined_all ={}
    combined_all_filtered = {}
    midway_filter={}
    sorted_symbols = sorted(set(dict1.keys()))
    for symbol in sorted_symbols:
        print(f"filtering {symbol}")
        log.append(f"********\n{symbol}\n********")
        unique_count = len(dict1[symbol])
        log.append(f"ncbi efetch proteins: {len(dict1[symbol])}")
        
        
        combined_all.update(dict1[symbol])
        
        filtered_combined= thin_data_original(dict1[symbol], p_identity=ident_threshold,mode = mode)
        log.append(f"Proteins after clustering: {len(filtered_combined)}\n")
        short_report.append("\t".join([symbol, (f"{unique_count}-->{len(filtered_combined)}".ljust(11)), f"{ident_threshold}%"]))
        combined_all_filtered.update(filtered_combined)
    with open(report_file,"w") as file:
        file.write("\n".join(short_report))
        file.write("\n".join(log))
    return combined_all_filtered 
with open('CYP_input.json',"r") as f:
    main_family_dict = json.load(f)
prefix = "CYP_PA_Attempt4"
download = False

if download:
    parsed_data = download_entrez(main_family_dict)
    #combined_all,midway_filter,combined_all_filtered= merge_positive(parsed_data,f"{prefix}_report.txt", ident_threshold="85",mode="mmseqs")
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
parsed_data = read_metadata_fasta(f"{prefix}_no_filter.fasta",f"{prefix}_no_filter_cuts.csv")
print("test_1")
combined_all_filtered = splice_aware_filter_attmept1(parsed_data,f"{prefix}_report_no_splice.txt", ident_threshold="85",mode="mmseqs")
save_fasta(combined_all_filtered,f"{prefix}_no_splice.fasta")
save_csv_splits(combined_all_filtered,f"{prefix}_no_splice.csv")
