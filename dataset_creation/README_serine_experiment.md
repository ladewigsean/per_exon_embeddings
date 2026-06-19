# Serine-protease per-exon experiment — protocol

These three helpers turn the saturating serine-protease benchmark into a test that can
actually answer the thesis question: **does gene architecture carry class signal beyond
the protein sequence?** They diagnose and fix why both models hit ~97 %.

## Why it saturated (the three causes)
1. **Leaky split.** `train_val_test_splits.py` splits proteins at random. Eukaryotic
   serine proteases are full of near-identical paralogs, so duplicates land in both
   train and test and the model memorises. → `cluster_split.py`.
2. **Cross-clan labels.** Labelling across MEROPS clans is fold classification, trivial
   for ProtT5. Use families **within one clan** (same fold) so they actually compete.
3. **Lossy "per-exon".** `embed_pers.py` mean-pools per-residue embeddings per exon, so
   it keeps sequence content but discards the architecture (exon counts/lengths/intron
   phase). Those are the genealogical signals the hypothesis is about. → `exon_architecture.py`.

## The arms (all share ONE classifier head + the same cluster split)
| arm | input | built by |
|---|---|---|
| A | per_prot (ProtT5 mean, 1024-d) | existing `embed_pers.py` |
| B | per_exon (mean-pooled) | existing `embed_pers.py` |
| C | architecture-only (~17-d) | `exon_architecture.py` |
| D | per_prot **+** architecture | `exon_architecture.py --concat` |

## Read the result honestly
- **Primary endpoint = D vs A.** Does adding architecture to the sequence help?
  A-vs-C is *not* the test (1024-d pretrained PLM vs 17 hand features — C losing proves
  nothing).
- A win must beat **two controls**, both built from `exon_architecture.py`:
  - **shuffle control** (`--shuffle SEED`): same features, architecture↔label link broken.
    D must beat shuffled-D, else architecture adds nothing (this makes a *negative*
    airtight too).
  - **length baseline** (`--feature-set length_only`): C/D must beat a 2-feature
    size classifier, or the "win" is just "S9 is long, S33 is short".
- Report **macro-F1 + per-class P/R** (`report_metrics.py`), never accuracy.

## Run order
```bash
# 1. leakage-free split (writes test_split 0/1/2 into the CSV)
python cluster_split.py --fasta SP_clanSC.fasta --csv SP_clanSC.csv \
    --out SP_clanSC_split.csv --min-seq-id 0.3 --mmseqs-cmd "wsl --exec mmseqs"
#    -> READ the printed per-split x per-label table. Require >=~5 per class in BOTH
#       val and test. If 30% starves a class, raise --min-seq-id to 0.4-0.5 (still
#       removes the near-duplicate leakage) or drop that class.

# 2. architecture features + controls
python exon_architecture.py --csv SP_clanSC_split.csv --fasta SP_clanSC.fasta \
    --out-h5 arch.h5 --scaler arch_scaler.json                    # arm C (full)
python exon_architecture.py ... --out-h5 arch_shuffN.h5 --shuffle 0       # control
python exon_architecture.py ... --out-h5 arch_len.h5 --feature-set length_only
python exon_architecture.py ... --out-h5 arch.h5 \
    --concat SP_clanSC_per_prot.h5 --concat-out per_prot_plus_arch.h5      # arm D

# 3. train EACH arm with the SAME head (1-D arms must use Basic/Pooling, not Transformer)
#    then score with report_metrics.report(y_true, y_pred, class_names=...)
```

## Pre-flight checks (cheap, but blocking)
- **Verify the `cut_pos` origin** on 2–3 genes against the RefSeq gene model: phase is
  `cut_pos % 3`, which is true intron phase only if `cut_pos` is measured from the CDS
  start (the A of ATG). Confirm before trusting the phase features.
- **Family set:** clan SC `S9 / S10 / S28` is a solid same-fold, Metazoa-populated set.
  `S33` has no confirmed UniProt handle in `serine_protease_input.json` (`"verify": true`)
  and is taxonomically thin — fill its handle or drop it, don't silently run on 3 classes.
- **Keep the head + hyperparameters identical across arms** — otherwise you're comparing
  models, not representations.

## Interpretation
- **D > A and D > shuffled-D and C > length_only** → architecture carries real signal
  beyond sequence: the per-exon hypothesis holds (positive thesis).
- **D ≈ shuffled-D** → architecture adds nothing; mean-pooled per-exon can't win *because*
  the architecture doesn't separate these families — a clean characterised negative with
  the mechanism shown.
