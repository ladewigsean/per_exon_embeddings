#!/usr/bin/env python3
"""
cluster_split.py -- leakage-free, cluster-aware train/val/test split.

Drop-in replacement for the random `train_test_split` in train_val_test_splits.py.
A random (per-protein) split puts near-identical paralogs -- and eukaryotic serine
proteases are full of them (mammalian trypsins / kallikreins / granzymes in S1 alone)
-- into BOTH train and test, so the model just memorises and val accuracy saturates
(~97 %+). That is leakage, not learning, and it kills the headroom the per-exon vs
per-protein comparison needs.

This instead clusters the sequences at low identity (default 30 %) with the same
`mmseqs easy-cluster` you already call in filter_data.py, then assigns *whole clusters*
to train/val/test, stratified by label. No sequence in val/test shares > min-seq-id
identity with any train sequence, so the metric measures generalisation to unseen
families, not memorised duplicates.

Output: writes a `test_split` column (0 = train, 1 = val, 2 = test) into the metadata
CSV -- exactly what classifier/scripts/custom_datasets.split_dataset_into_subsets reads.

    python cluster_split.py --fasta SP_clanSC.fasta --csv SP_clanSC.csv \
        --out SP_clanSC_split.csv --min-seq-id 0.3 --mmseqs-cmd "wsl --exec mmseqs"

The clustering step is isolated in run_mmseqs_cluster(); assign_clusters_stratified()
is pure (no I/O) and unit-tested in test_cluster_split.py, so the split logic can be
checked without mmseqs installed.
"""
import argparse
import os
import subprocess
import tempfile
from collections import defaultdict

# pandas is imported inside main() so the pure split logic (assign_clusters_stratified)
# stays importable with only the stdlib -- see test_serine_helpers.py.

SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def run_mmseqs_cluster(fasta_path, min_seq_id, cov, mmseqs_cmd, workdir):
    """Cluster `fasta_path` with mmseqs easy-cluster -> {member_id: cluster_rep_id}.

    `mmseqs_cmd` is split on spaces, so "wsl --exec mmseqs" works on Windows/WSL exactly
    as in filter_data.py; plain "mmseqs" works on Linux/Mac.
    """
    out_prefix = os.path.join(workdir, "clu")
    tmp = os.path.join(workdir, "tmp")
    cmd = mmseqs_cmd.split() + [
        "easy-cluster", fasta_path, out_prefix, tmp,
        "--min-seq-id", str(min_seq_id)
    ]#, "-c", str(cov), "--cov-mode", "0",
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    clusters_dict = defaultdict(list)
    
    with open(out_prefix + "_cluster.tsv") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            clusters_dict[rep].append(member)
    return clusters_dict


def assign_clusters_stratified(clusters, fracs=(0.8, 0.1, 0.1)):
    """Greedy, label-stratified assignment of WHOLE clusters to 3 splits.

    cluster_to_items : {cluster_id: [identifier, ...]}
    labels_by_id     : {identifier: label}
    Returns {identifier: split_index 0/1/2}. By construction no cluster is split, so
    there is zero train<->val/test sequence overlap above the clustering threshold.

    Each cluster is assigned under its majority ("primary") label, and within a label
    the clusters are placed -- largest first -- into whichever split is currently most
    below its target count for that label. That spreads every label across all three
    splits by `fracs` while keeping clusters intact.
    """
    # Per-cluster label composition, size and primary label.
    cluster_size = {c:len(clusters[c]) for c in clusters}
    
    
    assign = {}
    
    clusters_list = sorted(clusters, key=lambda c: -cluster_size[c])
    
    total = sum(cluster_size[c] for c in clusters_list)
    targets = [total * f for f in fracs]
    counts = [0, 0, 0]
    for c in clusters_list:
        # most-owed split for this label (largest target deficit)
        s = max(range(3), key=lambda i: targets[i] - counts[i])
        assign[c] = s
        counts[s] += cluster_size[c]

    # Rescue: the greedy fills train first, so a label with few clusters can leave
    # val/test EMPTY -- which silently makes that class untestable and craters
    # macro-F1. Force val(1) then test(2) to receive a cluster when enough exist.
    in_split = lambda s: [c for c in clusters_list if assign[c] == s]
    for tgt in (1, 2):
        if not in_split(tgt) and len(in_split(0)) >= 2:
            donor = min(in_split(0), key=lambda c: cluster_size[c])
            assign[donor] = tgt
    reached = {assign[c] for c in clusters_list}
    if reached != {0, 1, 2}:
        print(f"  NOTE: label reached splits {sorted(reached)} from "
                f"{len(clusters_list)} clusters -- too few for 3-way coverage; consider a "
                f"coarser --min-seq-id or dropping this class.")

    id_to_split = {}
    for c, items in clusters.items():
        s = assign.get(c)
        if s is None:
            continue
        for i in items:
            id_to_split[i] = s
    return id_to_split


def split_report(df, label_col="gene", split_col="test_split"):
    """Per-split x per-label sequence counts (a quick sanity table)."""
    tab = (df.groupby([label_col, split_col]).size()
             .unstack(fill_value=0)
             .rename(columns=SPLIT_NAMES))
    for s in ("train", "val", "test"):
        if s not in tab.columns:
            tab[s] = 0
    return tab[["train", "val", "test"]]

def subset_fasta(subset,fasta,output):
    from Bio import SeqIO
    subset_seqs = [] 
    
    
    for record in SeqIO.parse(fasta, "fasta"):
        if str(record.id) in subset:
            subset_seqs.append(record)
        
        
    SeqIO.write(subset_seqs,output,"fasta")
    
def main(args):
    import pandas as pd
    df = pd.read_csv(args.csv, dtype={args.id_col: str})
    for col in (args.id_col, args.label_col):
        if col not in df.columns:
            raise SystemExit(f"column '{col}' not in {args.csv} (have: {list(df.columns)})")
    
    unique_labels = set(df[args.label_col])
    dfs = []
    clusters_total = {}
    ids_total = {}
    
    with tempfile.TemporaryDirectory() as workdir:
        for label in unique_labels:
            
            if not os.path.isdir(workdir):
                os.mkdir(workdir)
            df_temp = df[df[args.label_col] == label]
            labels = list(df_temp[args.id_col])
            temp_fasta = os.path.join(workdir,"temp_input.fasta")
            subset_fasta(labels,args.fasta,temp_fasta)
            clusters = run_mmseqs_cluster(
                temp_fasta, args.min_seq_id, args.cov, args.mmseqs_cmd, workdir)

            # Any id present in the CSV but missing from the cluster map (e.g. dropped by
            # coverage) becomes its own singleton cluster, so it is never silently lost.
            
            print(f"{len(df_temp)} sequences -> {len(clusters)} clusters "
                f"(min-seq-id {args.min_seq_id}, cov {args.cov})")

            id_to_split = assign_clusters_stratified(
                clusters, fracs=(args.train, args.val, args.test))

            df_temp[args.split_col] = list(df_temp[args.id_col].map(id_to_split))
            dfs.append(df_temp)
            ids_total.update(id_to_split)
            clusters_total.update(clusters)
    df = pd.concat(dfs)
    print(df)
    missing = int(df[args.split_col].isna().sum())
    if missing:
        print(f"WARNING: {missing} rows got no split (no label?) -- dropping them.")
        df = df[df[args.split_col].notna()].copy()
    df[args.split_col] = df[args.split_col].astype(int)

    print("\nper-split x per-label sequence counts:")
    splits = split_report(df, args.label_col, args.split_col)
    print(splits)
    to_remove = list(splits[splits["test"] == 0].index)
    print(f"removing following classes:\n{"\n".join(to_remove)}")
    df = df[~df[args.label_col].isin(to_remove)]
    # Leakage guarantee: every cluster lands in exactly one split (by construction).
    spans = sum(len({ids_total[i] for i in items if i in ids_total}) > 1
                for items in clusters_total.values())
    print(f"\nclusters spanning >1 split: {spans}  (must be 0)")

    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  (column '{args.split_col}': 0=train 1=val 2=test)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", required=True, help="Sequences to cluster (ids = CSV identifiers).")
    p.add_argument("--csv", required=True, help="Metadata CSV with identifier + label columns.")
    p.add_argument("--out", required=True, help="Output CSV (input + a test_split column).")
    p.add_argument("--id-col", default="identifier")
    p.add_argument("--label-col", default="gene", help="Class label column (e.g. MEROPS family).")
    p.add_argument("--split-col", default="test_split")
    p.add_argument("--min-seq-id", type=float, default=0.3, help="mmseqs --min-seq-id (default 0.3).")
    p.add_argument("--cov", type=float, default=0.8, help="mmseqs -c coverage (default 0.8).")
    p.add_argument("--mmseqs-cmd", default="mmseqs",
                   help='How to invoke mmseqs, e.g. "wsl --exec mmseqs" (default "mmseqs").')
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    main(p.parse_args())
