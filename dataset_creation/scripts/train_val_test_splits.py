import csv
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
import h5py
def reorder(to_reorder, order,output_name):
    tochange = pd.read_csv(to_reorder)
    ordered_list = list(pd.read_csv(order)["identifier"])
    tochange = tochange.loc[ordered_list]
    tochange.to_csv(output_name, index=False)
def filter_from_csv(original, filter_out, output):
    df = pd.read_csv(original)
    to_be_removed = set(pd.read_csv(filter_out)["identifier"])
    df = df[~df["identifier"].isin(to_be_removed)]
    df.to_csv(output,index = False)   
def make_csv_h5(df,embedding_file,prefix):
    identifiers = df["identifier"]
    with h5py.File(prefix+".h5", "w") as hdf:
        for identifier in identifiers:
           hdf.create_dataset(name=identifier,data=embedding_file[identifier][:])
    df.to_csv(prefix+".csv",index = False)
            
        
def split_train_val_test(csv,min_per_class=60):
    df = pd.read_csv(csv)
    counts = df["gene"].value_counts()
    to_keep = list(counts[counts >= min_per_class].index)
    df = df[df["gene"].isin(to_keep)].reset_index(drop=True)
    print(df)
    labels = df["gene"]
    train_val_ind, test_ind, train_val_labels, _ = train_test_split(
        range(len(labels)),
        labels,
        train_size = 0.9,
        stratify = labels,
        random_state=42
    )
    
    train_ind, val_ind, _, _ = train_test_split(
        train_val_ind,
        train_val_labels,
        test_size = 0.1111111111111,
        stratify = train_val_labels,
        random_state=42
    )
    df["test_split"] = 0
    df.loc[val_ind,"test_split"] = 1
    df.loc[test_ind,"test_split"] = 2
    df.to_csv(csv,index = False)
    """
    train_df = df.iloc[train_ind]
    val_df = df.iloc[val_ind]
    test_df = df.iloc[test_ind]
    for x in range(len(h5_files)):
        prefix = os.path.join(folder,prefixs[x])
        embeddings_file = h5py.File(h5_files[x], 'r')
        make_csv_h5(train_df,embeddings_file,prefix+"_train")
        make_csv_h5(val_df,embeddings_file,prefix+"_val")
        make_csv_h5(test_df,embeddings_file,prefix+"_test")
        embeddings_file.close()
    """
###
#split_train_val_test("CYP_PA_Attempt4.csv",["CYP_PA_Attempt4_per_prot.h5","CYP_PA_Attempt4_per_res.h5","CYP_PA_Attempt4_per_exon.h5","CYP_PA_Attempt4_fixed_length_chunks.h5","CYP_PA_Attempt4_fixed_total_chunks.h5"],["per_prot","per_res","per_exon","fixed_length_chunks","fixed_total_chunks"])
#split_train_val_test("CYP_PA_Attempt5.csv","CYP_PA_Attempt5_train_val_test")
