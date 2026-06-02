
import os
from dotenv import load_dotenv
from Bio import SeqIO
import sys
import json
import subprocess

import os
from io import StringIO
import re
import json
import pandas as pd 
import numpy as np
from Bio import Entrez
from Bio import UniProt
from UniProtMapper import ProtMapper
import math
import seaborn as sns
import matplotlib.pyplot as plt
DATASETS_CMD = ["/mnt/c/Users/ladew/Documents/datasets_path/datasets"]

def download_entrez_from_fam_dict_uniprot(main_family_dict,output_folder="output_entrez",batch_size=32,log_file = "log.txt" ):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    load_dotenv(".env")
    Entrez.email = os.environ['entrez_email']
    Entrez.api_key = os.environ['entrez_key']
    mapper = ProtMapper()
    parsed_data = {}
    query_format = main_family_dict["query"]
    main_family = main_family_dict["main_family"]
    prefix = main_family_dict["prefix"]
    additional = main_family_dict["additional"]
    families = main_family_dict["families"]
    log = []
    for family in families:
        family_name = family["family"]
        subfamilies = family["subfamily"]
        for subfamily,custom_query in subfamilies.items():
            
            symbol = prefix + family_name + subfamily
            
            protein_file = os.path.join(output_folder,(symbol+".fasta"))
            map_csv = os.path.join(output_folder,(symbol+".csv"))
            acc_txt = os.path.join(output_folder,(symbol+".txt"))
            print("Starting: " + symbol )
            log.append("Starting: " + symbol)
            if not os.path.isfile(protein_file):
                print("downloading : " + symbol )
                log.append("downloading : " + symbol )
                if not custom_query is None:
                    query = custom_query
                else:
                    query = eval(query_format)#:)
                if additional:
                    query = f"{query}{additional}"
                uniprot_search = UniProt.search(query,["accession"])
                print(f"{len(uniprot_search)} search results ")
                log.append(f"{len(uniprot_search)} search results ")
                
                uniprot_ids = [uniprot_dict["primaryAccession"] for uniprot_dict in uniprot_search]
                print(f"have {len(uniprot_ids)} ids ")
                log.append(f"have {len(uniprot_ids)} uniprot ids ")
                if len(uniprot_ids)==0:
                    continue
                with open(acc_txt,"w") as f:
                    f.write("\n".join(uniprot_ids))
                
                result,leftover = mapper.get(uniprot_ids,from_db ="UniProtKB_AC-ID", to_db="RefSeq_Protein")
                protein_list = list(set(result["To"]))
                count = len(protein_list)
                result.to_csv(map_csv)
                print(f"Entrez Protein IDs found: {count}")
                log.append(f"Entrez Protein IDs found: {count}")
                
                num_fails = 0
                links = set({})
                for start_post in range(0,count, 500):
                    current_proteins = protein_list[start_post:start_post+500]
                    search_results = Entrez.read(Entrez.epost("protein", id=",".join(current_proteins)))
                    for start in range(0, len(current_proteins), batch_size):
                        end = min(len(current_proteins), start + batch_size)
                        attempts = 3
                        success = False
                        while attempts >0 and not success:
                            try:
                                handle = Entrez.elink(
                                    dbfrom="protein",
                                    retstart=start,
                                    retmax=batch_size,
                                    webenv=search_results["WebEnv"],
                                    query_key=search_results["QueryKey"],
                                    linkname="protein_gene")
                            
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
                    
                genes_list = list(links)
                print(f"Failed elink {num_fails} batches")
                log.append(f"Failed elink {num_fails} batches")
                print(f"proteins traced to {len(genes_list)} genes")
                log.append(f"proteins traced to {len(genes_list)} genes")
                count = len(links)
                links = set({})
                num_fails = 0
                for start_post in range(0,count, 500):
                    current_genes = genes_list[start_post:start_post+500]
                    search_results = Entrez.read(Entrez.epost("gene", id=",".join(current_genes)))
                    for start in range(0, len(current_genes), batch_size):
                        end = min(len(current_genes), start + batch_size)
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
                protein_list = list(links)
                print(f"Failed elink2 {num_fails} batches")
                log.append(f"Failed elink2 {num_fails} batches")
                print(f"proteins found {len(protein_list )} from genes")
                log.append(f"proteins found {len(protein_list )} from genes")
                count = len(links)
                num_fails = 0
                with open(protein_file, "w") as file:
                    for start_post in range(0,count, 500):
                        current_proteins = protein_list[start_post:start_post+500]
                        search_results = Entrez.read(Entrez.epost("protein", id=",".join(current_proteins)))
                        for start in range(0, len(current_proteins), batch_size):
                            end = min(len(current_proteins), start + batch_size)
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
                                    print(f"failed batch {start_post+start}-{start_post+end}\nattempts remaing: {attempts}")
                            if attempts == 0:
                                print (f"fully failed batch {start_post+start}-{start_post+end}, Skipping batch...")
                                num_fails +=1
                            elif attempts < 3:
                                print(f"batch {start_post+start}-{start_post+end} completed after {4-attempts} attempts" )
                print(f"Failed {num_fails} batches")
                log.append(f"Failed {num_fails} batches")

            current_data = parse_symbol_grab(protein_file,symbol,check_symbol=False)
            parsed_data[symbol] = (current_data)
    with open(log_file,"w") as f:
        f.write("\n".join(log))
    return parsed_data

def download_entrez_from_fam_dict(main_family_dict,output_folder="output_entrez",taxon = "animals",batch_size=32 ):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    load_dotenv(".env")
    Entrez.email = os.environ['entrez_email']
    Entrez.api_key = os.environ['entrez_key']
    
    parsed_data = {}
    query_format = main_family_dict["query"]
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
                if "custom_query" in family:
                    query = eval(family["custom_query"])#:)
                else:
                    query = eval(query_format)#:)
                handle = Entrez.esearch(db="gene",retmax = 1000, term=query,usehistory="y", idtype="acc")
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
def save_acc_from_csv(metadata_csv,file_name):
    df = pd.read_csv(metadata_csv)
    with open(file_name, "w") as file:
        file.write("\n".join(list(df["identifier"])))
        
            

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

def run_datasets_for_accession(accessions_file, json_file_output):
    

    cmd = DATASETS_CMD+["summary", "gene", "accession","--inputfile",accessions_file,  "--report", "product","--as-json-lines"]
    print(f"Running command for {accessions_file}: {' '.join(cmd)}")
    with open(json_file_output, "w") as outfh:
        subprocess.run(cmd, stdout=outfh)
    return json_file_output
def process_json(json_file,accessions):
    output_dict = {}
    with open(json_file) as f:
        try:
            raw_data = f.read()
        except json.JSONDecodeError:
            print(f"    -> WARNING: Could not parse JSON, file may be empty: {json_file}")
            return []
    input_stream = StringIO(raw_data)

    results = {}
    for line in input_stream:
        try:
            product_info  = json.loads(line)
        except json.JSONDecodeError:
            print(f"couldnt parse as json\n{line}")
          
        taxid = product_info.get("tax_id")        
        gene_symbol = product_info.get("symbol", "unknown").lower()
        transcripts = product_info.get("transcripts", [])
       
        if len(transcripts)==0:
            print(f"Report with Gene ID {product_info.get("gene_id","unkown")} has zero transcripts")
            
        
        for transcript in transcripts:
            acc_version = transcript.get("accession_version")
            prot_accession = transcript.get("protein", {}).get("accession_version")
            if prot_accession not in accessions:
                print(f"The current protein({prot_accession}) accession does not match a searched protein accessions,skipping")
                continue
            glocs = transcript.get("genomic_locations", [])
            if not glocs or not glocs[0].get("exons"):
                print(f" failed to find exons for {prot_accession}")
                continue
            exons_sorted = sorted(glocs[0]["exons"], key=lambda e: int(e.get("order", 0)))
            cut_pos = []
            current_length = 0
            for exon in exons_sorted:
                length = abs(int(exon["begin"])-int(exon["end"]))+1
                current_length = length +current_length
                cut_pos.append(current_length)
            cds_info = transcript.get("cds", {}).get("range", [{}])[0]
            if cds_info== {}:
                continue
            cds_start = int(cds_info["begin"])-1
            cds_end_relative = int(cds_info["end"])-2-cds_start
            cut_pos_cds = []
            for pos in cut_pos:
                relative_pos = pos-cds_start 
                if relative_pos >= cds_end_relative:
                    break
                if relative_pos > 0:
                    cut_pos_cds.append(str(relative_pos))
            results[prot_accession]=cut_pos_cds
        
    return results
            
            
            
def add_cds_ranges(metadata_csv,output_csv,json_file):
    df = pd.read_csv(metadata_csv)
    cut_pos_column = []
    cut_pos_cds_dict = process_json(json_file,set(df["identifier"]))
    print(f"protiens before exon search: {len(df["identifier"])}")
    df = df[df["identifier"].isin(cut_pos_cds_dict.keys())].reset_index()
    print(f"protiens after exon search: {len(df["identifier"])}")
    accessions = list(df["identifier"])
    for accession in accessions:
        cut_pos_column.append(f"[{",".join(cut_pos_cds_dict[accession])}]")
    df["cut_pos"] = cut_pos_column
    df.to_csv(output_csv,index=False)
def get_exon_data(metadata_csv,accession_file,output_csv,folder):
    json_file = os.path.join(folder,"ncbi_exon_data.json")
    if not os.path.isfile(json_file):
        run_datasets_for_accession(accession_file,json_file)
    add_cds_ranges(metadata_csv,output_csv,json_file)
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