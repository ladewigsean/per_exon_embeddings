from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import json
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer
from contextlib import ExitStack
import os
import math
#base from tum 2.1_ClickThrough_GenerateEmbeddings.ipynb, but added the per exon function
def seq_preprocess(df, model_type="esm"):
    df["sequence"] = df["sequence"].str.replace("[UZO]", "X", regex=True)
    #? think this is right?
    df["cut_pos"] = df["cut_pos"].apply(lambda x: json.loads(x))
    if model_type == "pt":
        df["sequence"] = df.apply(lambda row: " ".join(row["sequence"]), axis=1)
    return df
def setup_model(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "esm" in checkpoint:
        mod_type = "esm"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = EsmModel.from_pretrained(checkpoint)
    elif "ankh" in checkpoint:
        mod_type = "ankh"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = T5EncoderModel.from_pretrained(checkpoint)
    else:
        mod_type = "pt"
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        model = T5EncoderModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        model = model.half()
    return model.to(device), tokenizer, mod_type
def read_fasta(file_path):
    headers = []
    sequences = []
    fasta = Fasta(str(file_path))
    for seq in fasta:
        headers.append(seq.name)
        sequences.append(str(seq))
    return headers, sequences
def generate_splits_for_embed(per_res_embed, splits_list ):
    new_embed = np.empty((0,1024),np.float16)
    last_split = 0

    for split in splits_list:
        #
        exon_embed = per_res_embed[:,math.floor(last_split/3):math.ceil(split/3), : ].mean(axis=1).flatten()
        last_split = split
        new_embed = np.vstack([new_embed,[exon_embed]])
    new_embed  = np.vstack([new_embed,[per_res_embed[:,math.floor(last_split/3):,:].mean(axis=1).flatten()]])
    return new_embed 
def generate_same_size_splits(per_res_embed, size=150 ):
    number_nuc = per_res_embed.shape[1]*3
    return generate_splits_for_embed(per_res_embed,list(range(size,number_nuc,size)))
    
def generate_same_total_splits(per_res_embed, total = 10 ):
    number_nuc = per_res_embed.shape[1]*3
    splits_list = list(map(int,np.linspace(0,number_nuc,11,dtype = np.int64)[1:-1]))
    return generate_splits_for_embed(per_res_embed,splits_list)
def create_embedding(checkpoint, df, emb_types=["per_prot"], output_files=["protein_embeddings.h5"]):
    print("Setting up model...")
    model, tokenizer, mod_type = setup_model(checkpoint)
    model.eval()
    df = seq_preprocess(df, mod_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_embedding(sequence, emb_type):
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            max_length=4_000,
            truncation=True,
            padding=True,
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.cpu().numpy()
        if emb_type == "per_res":
            """
            if mod_type in ["pt", "ankh"]:
                outputs = outputs[0, :]#this was [:-1, :], was not working 
            elif mod_type == "esm":
                outputs = np.squeeze(outputs, axis=0)[:-1, :]
            """
            return outputs
        elif emb_type == "per_prot":
            return outputs.mean(axis=1).flatten()
        else:
            raise ValueError("Input valid embedding type")

    print("Generating embeddings...")
    with ExitStack() as stack:
        files = []
        hdf_dict = {emb_type:stack.enter_context(h5py.File(output_file, "w")) for output_file,emb_type in zip(output_files,emb_types)}
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sequence = row["sequence"]
            header = row["header"]
            cut_pos = row["cut_pos"]
            embedding_per_res = compute_embedding(sequence, "per_res")
            if "per_res" in hdf_dict and header not in hdf_dict["per_res"]:
                hdf_dict["per_res"].create_dataset(name=header, data=embedding_per_res[0])
            if "per_prot" in hdf_dict and header not in hdf_dict["per_prot"]:
                embedding_per_prot = embedding_per_res.mean(axis=1).flatten()
                hdf_dict["per_prot"].create_dataset(name=header, data=embedding_per_prot)
            if "per_exon" in hdf_dict and header not in hdf_dict["per_exon"]:
                embedding_per_exon = generate_splits_for_embed(embedding_per_res,cut_pos)
                hdf_dict["per_exon"].create_dataset(name=header, data=embedding_per_exon)
            if "fixed_length_chunks" in hdf_dict and header not in hdf_dict["fixed_length_chunks"]:
                embedding_fixed_length_chunks = generate_same_size_splits(embedding_per_res)
                hdf_dict["fixed_length_chunks"].create_dataset(name=header, data=embedding_fixed_length_chunks)
            if "fixed_total_chunks" in hdf_dict and header not in hdf_dict["fixed_total_chunks"]:
                embedding_fixed_total_chunks = generate_same_total_splits(embedding_per_res)
                hdf_dict["fixed_total_chunks"].create_dataset(name=header, data=embedding_fixed_total_chunks)
            

    del model, tokenizer
    torch.cuda.empty_cache()



def embed(fasta_filename,metadata_csv,embedding_types = ["per_prot"]):
    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    
    fasta_path = Path(fasta_filename)
    
    output_files = [str(fasta_path.with_suffix(""))+f"_{embedding_type}.h5" for embedding_type in embedding_types ]
    headers, sequences = read_fasta(fasta_path)
    df = pd.DataFrame({"header": headers, "sequence": sequences})
    metadata_df = pd.read_csv(metadata_csv)
    df = df.merge(metadata_df,left_on="header",right_on="identifier")
    df = df[["header","sequence","cut_pos"]]
    print(f"Processing {len(df)} sequences...")
    create_embedding(
        model_name,
        df,
        emb_types=embedding_types,
        output_files=output_files
    )
    
    #print(f"\nEmbeddings saved to {output_file}")

def combine_h5(files, output_file ='final4\\combined6.h5' ):
    with h5py.File(output_file ,mode='w') as h5fw:
        row1 = 0
        for h5name in files:
            h5fr = h5py.File(h5name,'r') 
            for key, item in h5fr.items():
              h5fw.create_dataset(name=key, data=item) 
     


#["per_prot","per_exon","per_res","fixed_length_chunks","fixed_total_chunks"]
embed("CYP_PA_Attempt5.fasta","CYP_PA_Attempt5.csv",embedding_types=["per_prot","per_exon","per_res","fixed_length_chunks","fixed_total_chunks"])

