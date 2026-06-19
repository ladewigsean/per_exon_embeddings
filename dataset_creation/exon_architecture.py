#!/usr/bin/env python3
"""
exon_architecture.py -- explicit gene-architecture features from exon boundaries.

The "per-exon" embedding in embed_pers.py is the MEAN of the per-residue embeddings
within each exon (generate_splits_for_embed). That keeps sequence *content* but throws
away the gene architecture itself -- how many exons, how long each is, and the intron
phase (where each intron falls relative to the codon frame). Those are exactly the
genealogical signals the per-exon hypothesis is supposed to exploit, so mean-pooled
per-exon can at best tie per-protein and will rarely beat it on any dataset: it carries
the same information, only coarser.

This computes a small FIXED-LENGTH architecture vector per gene straight from the
`cut_pos` already stored in your metadata CSV (the nucleotide exon-boundary positions,
same field embed_pers.py reads) plus the protein length from the FASTA. It z-scores the
features across the dataset and writes them to an h5 keyed by `identifier` -- the same
format MultiClassDataset reads -- so they slot straight into your training code.

The experiment this enables (all arms share the same cluster-split labels, and -- this
matters -- the SAME classifier head and hyperparameters, so only the input changes):
    arm A : per_prot                      sequence-only baseline
    arm B : per_exon (mean-pooled)        your current "per-exon"
    arm C : architecture-only             this file's h5
    arm D : per_prot (+) architecture     --concat per_prot.h5

Read the comparisons honestly:
  * PRIMARY test = D vs A. "Does adding architecture to the sequence representation
    help?" A-vs-C is NOT the test: A is a 1024-d pretrained PLM, C is ~17 hand features,
    so C losing is expected and proves nothing about architecture.
  * A win must survive two controls (both buildable from this script):
      - --shuffle SEED : the same arch block with the architecture<->label link broken.
        D must beat shuffled-D. If D ~= shuffled-D, architecture adds nothing (airtight).
      - --feature-set length_only : a size baseline. C/D must beat it, or the "win" is
        just a length classifier (within-clan families differ a lot in length).
  * 1-D arms (A, C, D, max_length=1): run with --nn_model Basic or Pooling, NOT
    Transformer (a 17-d / 1041-d model is not divisible by nhead and crashes; with no
    sequence axis a transformer is pointless anyway). Keep the head identical across arms.

    python exon_architecture.py --csv SP_clanSC_split.csv --fasta SP_clanSC.fasta \
        --out-h5 SP_clanSC_arch.h5 --scaler SP_clanSC_arch_scaler.json
    # optional: build the per_prot (+) architecture arm from an existing embedding h5
    python exon_architecture.py --csv ... --fasta ... --out-h5 arch.h5 \
        --concat SP_clanSC_per_prot.h5 --concat-out SP_clanSC_per_prot_plus_arch.h5

architecture_features() is pure and unit-tested in test_exon_architecture.py.
"""
import argparse
import json
import math
import statistics

# numpy/h5py/pandas are imported inside the functions that use them, so the pure
# feature maths (architecture_features) stays importable with only the stdlib --
# see test_serine_helpers.py.

FEATURE_NAMES = [
    "n_exons", "log_n_exons", "total_cds_nt", "log_total_cds_nt",
    "exon_len_mean", "exon_len_std", "exon_len_min", "exon_len_max", "exon_len_median",
    "exon_len_cv", "first_exon_nt", "last_exon_nt", "internal_exon_len_mean",
    "phase0_frac", "phase1_frac", "phase2_frac", "single_exon",
]

# Feature subsets, so the same script builds the controls the comparison needs:
#   full        -- all 17 features
#   length_only -- a baseline: if arch "wins", check it isn't just classifying by size
#                  (within-clan serine families differ a lot in length: S9 ~700 aa vs
#                  S33 ~310 aa). A genealogical claim must beat THIS, not just per_prot.
#   phase_only  -- isolates the actual genealogical signal (intron phase); the only
#                  architecture content arm B's mean-pooling cannot carry.
#   no_length   -- everything scale-free (drop raw-nt lengths), keep shape + phase.
FEATURE_SETS = {
    "full": FEATURE_NAMES,
    "length_only": ["log_total_cds_nt", "n_exons"],
    "phase_only": ["phase0_frac", "phase1_frac", "phase2_frac"],
    "no_length": ["n_exons", "log_n_exons", "exon_len_cv",
                  "phase0_frac", "phase1_frac", "phase2_frac", "single_exon"],
}


def architecture_features(cut_pos, cds_nt):
    """Fixed-length architecture vector from intron positions + total CDS length.

    cut_pos : list of intron positions in NUCLEOTIDES (cumulative, exclusive of 0 and
              the final end), exactly as embed_pers.generate_splits_for_embed consumes.
    cds_nt  : total coding length in nucleotides (protein length in aa * 3).
    Returns a list aligned with FEATURE_NAMES.
    """
    cuts = sorted(c for c in cut_pos if 0 < c < cds_nt)
    boundaries = [0] + cuts + [cds_nt]
    exon_lens = [max(1, boundaries[i + 1] - boundaries[i]) for i in range(len(boundaries) - 1)]
    n_exons = len(exon_lens)

    mean = sum(exon_lens) / n_exons
    std = statistics.pstdev(exon_lens) if n_exons > 1 else 0.0
    median = statistics.median(exon_lens)
    cv = std / mean if mean > 0 else 0.0

    phases = [c % 3 for c in cuts]
    n_phase = len(phases) or 1
    p0 = phases.count(0) / n_phase
    p1 = phases.count(1) / n_phase
    p2 = phases.count(2) / n_phase

    internal = exon_lens[1:-1] if n_exons > 2 else []
    internal_mean = (sum(internal) / len(internal)) if internal else mean

    return [
        float(n_exons), math.log1p(n_exons), float(cds_nt), math.log1p(cds_nt),
        mean, std, float(min(exon_lens)), float(max(exon_lens)), float(median),
        cv, float(exon_lens[0]), float(exon_lens[-1]), internal_mean,
        p0, p1, p2, 1.0 if n_exons == 1 else 0.0,
    ]


def read_fasta_lengths(fasta_path):
    """{id: protein_length_aa} parsed with the stdlib (no Biopython dependency)."""
    lengths, cur, n = {}, None, 0
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    lengths[cur] = n
                cur, n = line[1:].strip().split()[0], 0
            else:
                n += len(line.strip())
    if cur is not None:
        lengths[cur] = n
    return lengths


def standardize(matrix):
    """Z-score columns. Returns (standardized, mean, std) with zero-variance cols safe."""
    import numpy as np
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std_safe, mean, std


def main(args):
    import h5py
    import numpy as np
    import pandas as pd
    df = pd.read_csv(args.csv, dtype={args.id_col: str})
    if args.id_col not in df.columns or args.cut_col not in df.columns:
        raise SystemExit(f"need '{args.id_col}' and '{args.cut_col}' in {args.csv} "
                         f"(have: {list(df.columns)})")
    lengths = read_fasta_lengths(args.fasta)

    ids, rows, skipped = [], [], 0
    for _, r in df.iterrows():
        ident = r[args.id_col]
        aa = lengths.get(ident)
        if aa is None:
            skipped += 1
            continue
        cut_pos = r[args.cut_col]
        if isinstance(cut_pos, str):
            cut_pos = json.loads(cut_pos)          # CSV stores it as a JSON string
        rows.append(architecture_features(list(cut_pos), aa * 3))
        ids.append(ident)
    if skipped:
        print(f"skipped {skipped} rows with no sequence in the FASTA")

    matrix = np.array(rows, dtype=np.float64)
    # Select the requested feature subset (full / length_only / phase_only / no_length).
    keep_names = FEATURE_SETS[args.feature_set]
    keep_idx = [FEATURE_NAMES.index(n) for n in keep_names]
    matrix = matrix[:, keep_idx]
    print(f"{len(ids)} genes x {matrix.shape[1]} architecture features "
          f"(feature-set: {args.feature_set})")
    # A quick look at the raw (pre-standardisation) feature means -- a sanity check.
    for name, m in zip(keep_names, matrix.mean(axis=0)):
        print(f"  mean {name:<22} {m:.3f}")

    standardized, mean, std = standardize(matrix)

    # Architecture-shuffle control: permute which feature vector attaches to which gene
    # (seeded). This breaks the architecture<->label link while keeping the exact same
    # marginal distribution and dimensionality. It is the airtight control for BOTH
    # outcomes: if the real arch h5 beats this shuffled one, the gain is real signal; if
    # real ~= shuffled, "architecture adds nothing" is shown, not assumed. Build a normal
    # h5 and a `--shuffle <seed>` h5, train both, and compare.
    if args.shuffle is not None:
        perm = np.random.RandomState(args.shuffle).permutation(len(standardized))
        standardized = standardized[perm]
        print(f"SHUFFLE CONTROL: feature rows permuted with seed {args.shuffle} "
              f"(architecture<->label link broken).")
    with h5py.File(args.out_h5, "w") as h5:
        for ident, vec in zip(ids, standardized):
            h5.create_dataset(name=ident, data=vec.astype(np.float32))
    print(f"wrote {args.out_h5}  ({len(ids)} vectors, dim {matrix.shape[1]})")

    if args.scaler:
        with open(args.scaler, "w") as fh:
            json.dump({"feature_names": keep_names,
                       "mean": mean.tolist(), "std": std.tolist()}, fh, indent=2)
        print(f"wrote {args.scaler}")

    if args.concat:
        arch = {i: v.astype(np.float32) for i, v in zip(ids, standardized)}
        n = 0
        with h5py.File(args.concat, "r") as src, h5py.File(args.concat_out, "w") as dst:
            for ident in src.keys():
                if ident not in arch:
                    continue
                emb = np.asarray(src[ident][:]).reshape(-1)        # per-protein is 1-D
                dst.create_dataset(name=ident, data=np.concatenate([emb, arch[ident]]))
                n += 1
        print(f"wrote {args.concat_out}  ({n} per_prot(+)arch vectors)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Metadata CSV with identifier + cut_pos.")
    p.add_argument("--fasta", required=True, help="Protein FASTA (for sequence lengths).")
    p.add_argument("--out-h5", required=True, help="Output architecture-feature h5 (arm C).")
    p.add_argument("--scaler", default=None, help="Optional JSON dump of the z-score mean/std.")
    p.add_argument("--id-col", default="identifier")
    p.add_argument("--cut-col", default="cut_pos")
    p.add_argument("--feature-set", default="full", choices=list(FEATURE_SETS),
                   help="full / length_only (size baseline) / phase_only / no_length.")
    p.add_argument("--shuffle", type=int, default=None, metavar="SEED",
                   help="Permute feature vectors across genes (control: breaks arch<->label).")
    p.add_argument("--concat", default=None, help="Existing per-protein h5 to concatenate (arm D).")
    p.add_argument("--concat-out", default="per_prot_plus_arch.h5", help="Output for --concat.")
    main(p.parse_args())
