from pyfaidx import Fasta
from dotenv import load_dotenv
import pandas as pd
import os
from Bio import Entrez
from Bio import UniProt
from Bio import SeqIO
import sys
from UniProtMapper import ProtMapper
from Levenshtein import distance as ldistance
from scripts.grab_data import parse_symbol_grab
def read_fasta_as_dict(file_path):
    prot_dict = {}
    fasta = Fasta(str(file_path))
    for seq in fasta:
        prot_dict[seq.name] =str(seq)
        
    return prot_dict
def compare_fasta(fasta,id_to_seq, to_from_df):
    
    to_keep_records = []
    df = to_from_df.set_index("To")
    count = 0
    for record in SeqIO.parse(fasta, "fasta"):
        count += 1
        #get uniprot id
        original_id = df.loc[record.id,"From"]
        print(original_id)
        if isinstance(original_id,pd.Series):
            
            original_id = original_id.iloc[0]
            print(original_id)
        original_seq = id_to_seq[original_id]
        new_seq = record.seq
        distance = ldistance(original_seq,new_seq,score_cutoff=15)
        if distance <= 15:
            to_keep_records.append(record)
        
           
        #make sure they are somewhat similair
        """
        if len(rows) == 1:
            to_keep_records.append(record)
            continue
        """
        
         
    SeqIO.write(to_keep_records,fasta,"fasta")
    print(f"Length Before compare: {count}\nLength after compare: {len(to_keep_records)}")
"""
def deeploc_grab(file_path):
    data = {}
    fasta = Fasta(str(file_path))
    for seq in fasta:
        split_header = seq.name.split(" ")
        if len(split_header ) !=3:
            split_header = split_header + ["train"]
        split_header = split_header + [str(seq)]
        data[split_header[0]] =split_header[1:] 
        
    
    return headers
"""
def deeploc_split_csv(csv):
    df = pd.read_csv(csv)
    df = df[["ACC","localization"]]
    df["localization"] = df["localization"].str.replace("/","_")
    df["localization"] = df["localization"].str.replace(".","_")
    #df = df.set_index("ACC")
    class_dict = df.groupby("localization")["ACC"].apply(list).to_dict()
    
    
    return class_dict

def read_cath20_csv(csv,id_to_seq,folder,groupby = "homologous_superfamily"):
    
    temp_output = os.path.join(folder,"cath20_uniprot.csv")
    if not os.path.isfile(temp_output):
        from scripts.cath20_id_to_uniprot import cath20_id_to_uniprot
        df = pd.read_csv(csv)
        df = df[["id","class","architecture","topology","homologous_superfamily"]]
        cath_20_ids, uniprot_ids = cath20_id_to_uniprot(list(df["id"]))
        df = df.set_index("id") 
        df.loc[cath_20_ids,"uniprot_id"] = uniprot_ids
        
        print(f"Length cath20 ids: {len(df)}")
        df = df.dropna(subset=["uniprot_id"])
        print(f"Length uniprot ids: {len(df)}")
        df.to_csv(temp_output)
    else:
        df = pd.read_csv(temp_output)
    df[groupby] = groupby +"_"+ df[groupby].astype(str)
    class_dict = df.groupby(groupby)["uniprot_id"].apply(list).to_dict()
    new_id_to_seq = { new:id_to_seq[old] for old,new in zip(list(df["id"]),list(df["uniprot_id"]))}
    return class_dict,new_id_to_seq


def grab_deeploc(file_path,original_fasta,batch_size=32,output_folder="output_entrez",log_file = "log.txt"):
    id_to_seq = read_fasta_as_dict(original_fasta)
    deeploc_classes = deeploc_split_csv(file_path)
    return grab_with_uniprot_dict(deeploc_classes,id_to_seq,batch_size,output_folder,log_file)
def grab_cath20(file_path,original_fasta,batch_size=32,output_folder="output_entrez",log_file = "log.txt",groupby = "homologous_superfamily"):
    id_to_seq = read_fasta_as_dict(original_fasta)
    superfamily_classes,id_to_seq = read_cath20_csv(file_path,id_to_seq,output_folder,groupby)
    return grab_with_uniprot_dict(superfamily_classes,id_to_seq,batch_size,output_folder,log_file,suffix="cath20")
def grab_with_uniprot_dict(deeploc_classes,id_to_seq,batch_size=32,output_folder="output_entrez",log_file = "log.txt",suffix= "deeploc"):
    
    load_dotenv(".env")
    Entrez.email = os.environ['entrez_email']
    Entrez.api_key = os.environ['entrez_key']
    parsed_data = {}
    mapper = ProtMapper()
    log = []
    for loc, uni_ids in deeploc_classes.items():     
        loc = str(loc)
        
        protein_file = os.path.join(output_folder,(f"{loc}_{suffix}.fasta"))
        print(f"Starting: {loc}"  )
        log.append(f"Starting: {loc}" )
        if not os.path.isfile(protein_file):
            print(f"downloading : {loc}"  )
            log.append(f"downloading : {loc}"  )
            
            
            print(f"have {len(uni_ids)} ids ")
            log.append(f"have {len(uni_ids)} uniprot ids ")
            if len(uni_ids)==0:
                continue
            
            
            result,leftover = mapper.get(uni_ids,from_db ="UniProtKB_AC-ID", to_db="RefSeq_Protein")
            if result is None:
                print(f"skipping: {loc}")
                continue
            protein_list = list(set(result["To"]))
            count = len(protein_list)
            
            print(f"Entrez Protein IDs found: {count}")
            log.append(f"Entrez Protein IDs found: {count}")
            
            num_fails = 0
            links = set({})
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
            #confirm correct fasta
            compare_fasta(protein_file,id_to_seq,result)
        current_data = parse_symbol_grab(protein_file,loc,check_symbol=False)
        print(f"test_{loc}")
        parsed_data[loc] = (current_data)
        print(f"length ={len(parsed_data[loc])}")
    
    
    with open(log_file,"w") as f:
        f.write("\n".join(log))
    return parsed_data