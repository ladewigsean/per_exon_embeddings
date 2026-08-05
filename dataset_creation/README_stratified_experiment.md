# Identity-stratified analysis

The question: **does any arm degrade more slowly than the sequence baseline as
test-to-train identity drops?**

Everywhere else the answer is close to baked in by design. ProtT5 is sequence-only (it
never sees nucleotides, frame or introns) and it is chop-invariant (trained on
fragments), and gene-family classification is sequence-solvable. On that setup "exon
chunk is just an arbitrary chunk" and "architecture is redundant" are near-predictions,
not findings. The low-identity regime is the one place where the outcome is genuinely
open: intron position and phase can survive after sequence has diverged, so if
architecture carries ancestral signal, that is where it has to show up.

## You do not need a perfect split

`cluster_split.py` says "no sequence in val/test shares > min-seq-id identity with any
train sequence". Do not rely on that. `mmseqs easy-cluster --min-seq-id X` only bounds
identity between a member and **its own cluster representative**. Members of two
*different* clusters can sit well above X, for three independent reasons:

1. Clustering is **greedy set-cover**. Which sequence becomes a representative depends on
   length and order, so B can be far above X to A and still end up in its own cluster.
2. The **k-mer prefilter can miss the pair outright**. At ~30 % identity (the twilight
   zone) default sensitivity is not enough; a pair that is never aligned can never be
   joined.
3. **`-c 0.8 --cov-mode 0`** refuses to cluster a pair without 80 % *mutual* coverage,
   however identical the shared region. Multi-domain and length-variable families get
   split by coverage, not by identity.

So the diagnosis "min-seq-id only constrains representatives" is right. But the fix is
**not** a stricter global `--min-seq-id`. That knob is redundancy reduction over the
whole set; it never creates a train/test boundary (two proteins at 55 % identity both
survive a 0.6 filter and can still land on opposite sides of the split), it throws away
data the small families cannot spare, and it inherits the same greedy/prefilter/coverage
blind spots.

The fix is to **measure instead of assume**, then put the measurement on the x axis.
Once accuracy is reported *as a function of* identity-to-train, residual leakage stops
being a confound and becomes a covariate you condition on. The low bins are the hard
holdout; the high bins are a free bonus showing the crossover. This is the main reason
not to spend more time tightening the split.

## Protocol

**1. Measure identity to train.** Permissive on purpose (`-s 7.5`, `-e 10000`): a missed
hit silently becomes "identity 0", faking a hard test case that is not hard, which is the
one error that would flatter the result.

```bash
python dataset_creation/scripts/identity_to_train.py \
    --fasta input_data/NCBI_HOX/HOX.fasta \
    --csv   input_data/NCBI_HOX/HOX.csv \
    --out   input_data/NCBI_HOX/HOX_ident.csv \
    --mmseqs-cmd "wsl --exec mmseqs"
```

Adds `train_max_pident`, `train_max_pident_cov`, `train_best_hit`, `train_n_hits`.
Two details that matter:

- **Use `train_max_pident_cov` when filtering.** Raw identity from a *local* alignment
  is misleading on its own: a 30-residue exact match inside a 600-residue protein scores
  1.0 and is not leakage. Whichever you use, say which.
- **Queries with no hit are kept at 0.0**, never dropped. They are the hardest and most
  valuable test cases; dropping them deletes exactly the regime the experiment is about.
  The script prints how many there are.

**2. Get per-example test predictions.** `classifier_runner_whole_dir.py` now evaluates
split 2 and writes `predictions/<arm>.csv` with `identifier, true_label, pred_label,
correct, margin, nll`. If the trained checkpoints from an earlier sweep are still in
`model_weights/`, call `test_model(..., pred_out=...)` on those directly rather than
repeating the sweep.

**3. Analyse.**

```bash
python dataset_creation/scripts/stratified_analysis.py \
    --identity input_data/NCBI_HOX/HOX_ident.csv \
    --predictions per_prot=predictions/NCBIHOX_per_prot.csv \
                  per_exon=predictions/NCBIHOX_per_exon.csv \
                  per_prot_meta=predictions/NCBIHOX_per_prot_meta.csv \
                  meta_only=predictions/NCBIHOX_meta_only.csv \
    --baseline per_prot --out-prefix results/HOX
```

## Reading the output

You cannot box-plot `correct`. Inside a bin it only takes the values 0 and 1, so every
quartile collapses onto 0 or 1. "Stratified" never meant box-plot the correctness; it
meant report the metric *as a function of* identity. Three views, in the order to trust
them:

1. **`_delta.png` (primary).** Accuracy minus baseline per bin, paired bootstrap 95 % CI.
   All arms are scored on the *same* proteins, so compare them per protein: the
   protein-to-protein variance cancels and a 3-point difference two noisy curves cannot
   resolve often becomes significant. It also cancels a real confound: class composition
   shifts across identity bins (the hardest bin is not a random sample of classes), which
   biases each raw curve but not their difference.
2. **`_accuracy.png`.** Per-bin accuracy with Wilson 95 % CIs and n printed on the axis.
   Wilson, not normal-approximation, because the tails have small n and p near 0 or 1,
   where the normal approximation returns bounds outside [0, 1]. The lowest bin is
   always the smallest, i.e. least reliable exactly where it matters most; the CI says so.
3. **`_margin.png`.** Per-example margin (logit of the true class minus the best
   competing logit) by bin. **This** is where a box plot is right, because margin is
   continuous and has real spread inside a bin.

   On distrusting softmax: correct for calibration claims ("the model is 90 % sure"),
   wrong here. This is not an absolute confidence claim; it ranks the *same* proteins
   under *different* arms, and any monotone miscalibration shared by the arms cancels.
   Margin is pre-softmax, so it needs no calibration assumption at all. Accuracy stays
   primary and the margin is the higher-power secondary; if they disagree, believe
   accuracy and find out why.

The printed table also gives a bin-free **slope comparison**: negative means that arm's
accuracy rises less steeply with identity, i.e. it depends less on a close homolog being
in train. Both slopes compress when arms are near ceiling, so read it next to the
accuracy plot, never alone.

Per-bin numbers are **accuracy, not macro-F1**: a bin routinely holds 0-2 examples of
some class, whose per-class recall is then 0 or 1 and macro-F1 becomes noise. Report
macro-F1 on the full test set and accuracy per bin, and say so.

`--cluster-col` makes the bootstrap resample **clusters** rather than proteins. Test
proteins from one cluster are not independent draws; without it the CIs are too narrow.

## Two things this changes about the existing benchmark

**The committed `output_csvs/*.csv` are validation numbers, not test numbers.**
`classifier_runner_whole_dir.py` never called `test_model`; it built `test_dataset` and
left it unused. HPO ran on train, early stopping used val, and the best-of-5-seeds
checkpoint was chosen by val accuracy, so those figures are optimistically biased and
were never held out. The upside is that split 2 is genuinely untouched, which is the
cleanest possible resource for the final thesis numbers. The runner now evaluates it and
writes `test_acc` / `test_macro_f1` columns (`--skip_test` restores the old behaviour).

**`per_prot_meta` was compared against the wrong baseline.** Read the shuffle control in
the committed results:

| macro-F1 | CYP | FGF | HOX | KLF | PCDH |
|---|---|---|---|---|---|
| `per_prot` | 0.599 | 0.529 | 0.822 | 0.822 | 0.773 |
| `per_prot_meta` | 0.613 | 0.522 | 0.680 | 0.796 | 0.550 |
| `per_prot_meta_shuffle` | 0.597 | 0.478 | 0.687 | 0.809 | 0.437 |

`per_prot_meta_shuffle` carries **row-permuted, information-free** features, yet it falls
0.135 below `per_prot` on HOX and 0.336 below on PCDH. A healthy arm would learn to
ignore useless inputs and land on top of `per_prot`. It does not, so the concatenation
arm is damaged by something other than the features' content, and "`per_prot_meta` ≤
`per_prot`, therefore architecture is redundant" does not follow.

Against its **matched** control instead, architecture helps on 3 of 5 (CYP +0.016,
FGF +0.044, PCDH +0.113) and ties on HOX (−0.007) and KLF (−0.013). Two candidate causes
worth one diagnostic each before drawing any conclusion:

- **Scale.** `exon_architecture.py` z-scores the 17 features to unit variance, then
  concatenates them onto raw ProtT5 dims whose per-dimension std is roughly 0.1. The 17
  features arrive with about ten times the scale of the 1024 they join, so they dominate
  the first layer. Check the per-dimension std of both blocks in the concatenated h5.
- **Dropped proteins.** The concat loop keeps only ids present in *both* the per_prot h5
  and the architecture dict, so any protein missing from the FASTA silently disappears
  and can take a whole class with it. Compare `len(h5.keys())` and the per-class counts
  between `X_per_prot.h5` and `X_per_prot_meta.h5`.

The one clean architecture result is untouched by this: `meta_only` beats
`meta_only_shuffle` by +0.14 to +0.21 macro-F1 on **all five** families. Architecture
carries real, replicated class signal. It is weak, but it is not nothing, and it is the
positive result the write-up should state plainly.
