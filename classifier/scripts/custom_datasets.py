
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import h5py
class MultiClassDataset(Dataset):
    def __init__(self, embeddings_path: str, csv_path: str, label_encoder = None):
        super().__init__()
        self.embeddings_file_path = embeddings_path
        self.metadata_df = pd.read_csv(csv_path, dtype={'identifier': str})
        self.id_column = 'identifier'
        self.gene_column = 'gene'
        self.h5f = None
        self.max_length = None
        if self.id_column not in self.metadata_df.columns or self.gene_column not in self.metadata_df.columns:
            raise ValueError(f"Required columns 'identifier' or 'gene' not found in {csv_path}.")
        if label_encoder is None:
            self.label_encoder = OneHotEncoder(sparse_output=False).fit(
                self.metadata_df[[self.gene_column]].values
            )
        else:
            self.label_encoder = label_encoder
            known_genes = self.label_encoder.categories_[0]
            self.metadata_df = self.metadata_df[self.metadata_df[self.gene_column].isin(known_genes)]

        self.encodings = self.label_encoder.transform(
            self.metadata_df[[self.gene_column]].values
        )
        self.metadata_df = self.metadata_df.copy()
        self.metadata_df["label"] = list(self.encodings)
        self.num_classes = len(self.label_encoder.categories_[0])
        print(f"INFO: Label encoder has {self.num_classes} classes.")
        with h5py.File(self.embeddings_file_path, "r") as h5f:
            embedding_ids = set(h5f.keys())
        original_len = len(self.metadata_df)
        self.metadata_df = self.metadata_df[
            self.metadata_df[self.id_column].isin(embedding_ids)
        ].reset_index(drop=True)
        # Recompute encodings after filtering
        self.encodings = self.label_encoder.transform(
            self.metadata_df[[self.gene_column]].values
        )
        if original_len != len(self.metadata_df):
            print(f"INFO: Filtered metadata for HDF5 keys. Kept {len(self.metadata_df)}/{original_len} entries.")
        
        with h5py.File(self.embeddings_file_path, 'r') as h5f:
            sample_key = list(h5f.keys())[0]
            self.embedding_dim = h5f[sample_key].shape[-1]
        self._compute_max_length()
        self.random_perm= None
        #self.random_perm = np.random.permutation(len(self.metadata_df))
    
        

    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.embeddings_file_path, 'r')

        row = self.metadata_df.iloc[index]
        # Load as float32 — mixed precision autocast handles the rest
        embedding = torch.tensor(self.h5f[str(row[self.id_column])][:], dtype=torch.float32)
        #def not the most elegant solution lol, but im pretty sure i cant modify it as we are grabing directly from h5  
        #rand_ind = np.random.RandomState(seed=index).permutation(embedding.shape[0])
        #embedding = embedding[rand_ind]
        if self.max_length > 1:
            seq_len = embedding.shape[0]
            pad_len = self.max_length - seq_len
            embedding = torch.cat([
                embedding,
                torch.zeros(pad_len, self.embedding_dim, dtype=torch.float32),
            ])
        else:
            seq_len = 1
            embedding = embedding.unsqueeze(0)
        if not self.random_perm is None:
            label = torch.tensor(self.metadata_df.iloc[self.random_perm[index]]["label"], dtype=torch.float32)
        else:
            label = torch.tensor(row["label"], dtype=torch.float32)
        return embedding, label, str(row[self.id_column]), seq_len

    def __len__(self):
        return len(self.metadata_df)

    def _compute_max_length(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.embeddings_file_path, 'r')
        self.max_length = 1

        for key in self.h5f.keys():
            shape = self.h5f[key].shape
            if len(shape) == 1:
                self.max_length = 1
                continue
            if shape[0] > self.max_length:
                self.max_length = shape[0]
    def get_max_length(self):
        if self.max_length is None:
            self._compute_max_length()
        return self.max_length 

    def get_labels(self):
        return self.metadata_df['label'].values
    
    def get_data(self):
        return self.metadata_df
    def __del__(self):
        """Close the HDF5 handle when the dataset is destroyed."""
        if getattr(self, 'h5f', None) is not None:
            try:
                self.h5f.close()
            except Exception:
                pass
            self.h5f = None


def _h5_worker_init_fn(worker_id):
    """Each DataLoader worker clears any inherited handle and re-opens
    its own on first __getitem__ call (inside the worker process)."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, 'h5f'):
        info.dataset.h5f = None
    
#omega scuffed but whatever, keeps it easier
class MultiClassSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.data = dataset.get_data().iloc[indices]
        self.encodings = dataset.encodings[indices]
        self.num_classes = dataset.num_classes
        self.label_encoder = dataset.label_encoder
        self.embeddings_file_path = dataset.embeddings_file_path
        self.embedding_dim = dataset.embedding_dim
        self.id_column = dataset.id_column
        self.h5f = None
    def get_labels(self):
        return self.data['label'].values
    def get_data(self):
        return self.data

def split_dataset_into_subsets(dataset):
    df = dataset.get_data()
    max_length = dataset.get_max_length()
    train_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==0)[0])
    val_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==1)[0])
    test_dataset = MultiClassSubset(dataset, np.where(df["test_split"]==2)[0])
    return train_dataset, val_dataset, test_dataset, max_length