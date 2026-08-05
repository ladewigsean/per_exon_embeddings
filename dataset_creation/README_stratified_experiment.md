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

`cluster_split.py` used to claim "no sequence in val/test shares > min-seq-id identity
with any train sequence". That claim was too strong and has been corrected in the
docstring. `mmseqs easy-cluster --min-seq-id X` only bounds identity between a member and
**its own cluster representative**. Members of two *different* clusters can sit well
above X, for three independent reasons:

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
survive a 0.6 filter and can still land on opposite sides of the split), it costs data
the small families cannot spare, and it inherits the same three blind spots.

The fix is to **measure instead of assume**, then put the measurement on the x axis.
Once accuracy is reported *as a function of* identity-to-train, residual leakage stops
being a confound and becomes a covariate you condition on. The low bins are the hard
holdout; the high bins are a free bonus showing the crossover. This is the main reason
not to spend more time tightening the split.

## Protocol

All commands are run **from the repository root** unless marked otherwise.

**0. Add the cluster column** (one-off; the committed metadata CSVs do not have one).
Test proteins from the same cluster are not independent draws, so the confidence
intervals resample clusters. `--annotate-only` adds the column and leaves `test_split`
exactly as it is; a full re-split would reassign train/val/test and invalidate every
model you have already trained.

```bash
python dataset_creation/scripts/cluster_split.py --annotate-only \
    --fasta dataset_creation/data/NCBI_HOX/NCBIHOX.fasta \
    --csv   classifier/input_data/NCBI_HOX/NCBIHOX.csv \
    --out   classifier/input_data/NCBI_HOX/NCBIHOX.csv
```

**1. Measure identity to train.** Permissive on purpose (`-s 7.5`, `-e 10000`): a missed
hit silently becomes "identity 0", faking a hard test case that is not hard, which is the
one error that would flatter the result.

```bash
python dataset_creation/scripts/identity_to_train.py \
    --fasta dataset_creation/data/NCBI_HOX/NCBIHOX.fasta \
    --csv   classifier/input_data/NCBI_HOX/NCBIHOX.csv \
    --out   results/NCBIHOX_ident.csv
```

Write the output **outside** `classifier/input_data/<dataset>/`:
`classifier_runner_whole_dir.py` requires exactly one CSV in that directory and refuses
to start if it finds two.

Adds `train_max_pident`, `train_max_pident_cov`, `train_best_hit`, `train_n_hits`.
Two details that matter:

- **`train_max_pident_cov` is the default axis** in the analysis, and it is the one to
  filter on. Raw identity from a *local* alignment is misleading alone: a 30-residue
  exact match inside a 600-residue protein scores 1.0 and is not leakage. Whichever you
  report, say which.
- **Queries with no hit are kept at 0.0**, never dropped. They are the hardest and most
  valuable test cases; dropping them deletes exactly the regime the experiment is about.
  The script prints how many there are.

On Windows with `--mmseqs-cmd "wsl --exec mmseqs"`, also pass `--workdir`, because the default
scratch path is a Windows temp directory that mmseqs inside WSL cannot open.

**2. Get per-example test predictions.** From `classifier/`:

```bash
cd classifier
python classifier_runner_whole_dir.py --dir input_data/NCBI_HOX --force_test
```

`--force_test` re-runs **only** the test stage for arms already in `output_csvs/`,
reusing `yaml/<arm>_HPO.yaml` and the surviving `model_weights/val_seed_*_<arm>_test.pt`.
That is a forward pass per arm, not a repeat of the HPO + 5-seed sweep. Without it,
already-recorded arms are skipped and you would get no test numbers at all. Writes
`classifier/predictions/<arm>.csv` with `identifier, true_label, pred_label, correct,
margin, nll`.

**3. Analyse.** Back at the repository root:

```bash
python dataset_creation/scripts/stratified_analysis.py \
    --identity results/NCBIHOX_ident.csv \
    --predictions per_prot=classifier/predictions/NCBIHOX_per_prot.csv \
                  per_exon=classifier/predictions/NCBIHOX_per_exon.csv \
                  per_prot_meta=classifier/predictions/NCBIHOX_per_prot_meta.csv \
                  meta_only=classifier/predictions/NCBIHOX_meta_only.csv \
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
3. **`_margin.png`.** Per-example margin (true-class output minus the best competing
   output) by bin. **This** is where a box plot is right, because margin is continuous
   and has real spread inside a bin.

   On distrusting softmax: your caution is right for *calibration* claims, and this plot
   does not make one. Margin is taken before any softmax, so no calibration assumption
   enters. What it does need is a scale fix: each arm is a separately trained network
   with its own output scale, and these models train with **MSE against one-hot targets**
   (the default `criterion` in `runners_eval.py`; the CE option is commented out of the
   HPO space), so the outputs are not logits in the cross-entropy sense. Margins are
   therefore **not comparable across arms raw**, and `--score-normalise rank` (the
   default) percentile-ranks within each arm first. Read it as "which proteins does this
   arm find hard relative to its own test set". Accuracy stays the primary endpoint.

The printed table also gives a bin-free **slope comparison**: negative means that arm's
accuracy rises less steeply with identity. Beware both ends. An arm near the ceiling
*and* an arm near chance both have slope ≈ 0 for reasons that have nothing to do with
homology. The script suppresses the significance star when an arm's overall accuracy is
more than `--slope-acc-tolerance` below the baseline's, and says so; `meta_only` at
0.18-0.27 macro-F1 against `per_prot`'s 0.53-0.82 is exactly that case.

Stars are **uncorrected for multiplicity** and the script prints how many comparisons it
ran. Treat a lone star as a lead; the shape of the delta curve across bins is the
evidence.

Per-bin numbers are **accuracy, not macro-F1**: a bin routinely holds 0-2 examples of
some class, whose per-class recall is then 0 or 1 and macro-F1 becomes noise. Report
macro-F1 on the full test set and accuracy per bin, and say so.

**What the CIs do not cover.** `train_model` keeps only the val-best checkpoint, so the
test evaluation and the per-example predictions come from a **single seed**. Every
interval here is sampling noise over proteins only; training noise is not in them, and
the 5-seed spread on the validation columns (`std_macro_f1` up to 0.06) is a fair
indication of how large it can be. Either say this in the write-up, or keep all five
checkpoints and evaluate all five.

## Two things this changes about the existing benchmark

**The committed `output_csvs/*.csv` are validation numbers, not test numbers.**
`classifier_runner_whole_dir.py` never called `test_model`; it built `test_dataset` and
left it unused. The bias is specific: `train_and_validate` checkpoints on the best val
macro-F1 across ~35 epochs and then reports *that* checkpoint's val metrics, so each
number is a maximum over epochs on the set it is scored against. (HPO itself used 5-fold
CV inside train and leaked nothing, and the reported mean is over all 5 seeds rather than
the best, so neither of those is the problem.) The upside is that split 2 is genuinely
untouched, which is the cleanest possible resource for the final thesis numbers. The
runner now evaluates it and writes `test_acc` / `test_macro_f1` / `test_status`
(`--skip_test` restores the old behaviour).

**`per_prot_meta` was compared against the wrong baseline.** Read the shuffle control in
the committed results (macro-F1, ± the 5-seed sd from the same file):

| | CYP | FGF | HOX | KLF | PCDH |
|---|---|---|---|---|---|
| `per_prot` | 0.599 | 0.529 | 0.822 | 0.822 | 0.773 |
| `per_prot_meta` | 0.613 | 0.522 | 0.680 | 0.796 | 0.550 |
| `per_prot_meta_shuffle` | 0.597 | 0.478 | 0.687 | 0.809 | 0.437 |

`per_prot_meta_shuffle` carries **row-permuted, information-free** features. A healthy
arm would learn to ignore useless inputs and land on `per_prot`. On CYP, KLF and FGF it
roughly does (0.06, 0.44 and 1.28 combined sd below). On **HOX (2.3 sd) and PCDH (6.4
sd)** it does not, and on those two the concatenation arm is damaged by something other
than the features' content. That is enough to stop "`per_prot_meta` ≤ `per_prot`,
therefore architecture is redundant" from following on those two families; it is not
evidence of damage on the other three.

Against the **matched** control instead, the differences are small and mostly inside
noise: CYP +0.016 (0.4 sd), FGF +0.044 (2.0 sd), PCDH +0.113 (1.6 sd), HOX −0.007
(−0.2 sd), KLF −0.013 (−0.3 sd). Only FGF and PCDH are suggestive, and neither is
decisive on its own. Three candidate
causes worth one diagnostic each before drawing any conclusion:

- **Scale.** `exon_architecture.py` z-scores the 17 features to unit variance, then
  concatenates them onto raw ProtT5 dims whose per-dimension std is roughly 0.1. The 17
  features arrive with about ten times the scale of the 1024 they join. Check the
  per-dimension std of both blocks in the concatenated h5.
- **Dropped proteins.** The concat loop keeps only ids present in *both* the per_prot h5
  and the architecture dict, so any protein missing from the FASTA silently disappears
  and can take a whole class with it. Compare `len(h5.keys())` and the per-class counts
  between `X_per_prot.h5` and `X_per_prot_meta.h5`.
- **HPO luck.** Every arm gets its own **unseeded** 30-trial Optuna study, so arm-to-arm
  differences confound feature content with search luck. Seed the study, or repeat it.

The one clean architecture result is untouched by any of this: `meta_only` beats
`meta_only_shuffle` by **+0.138 to +0.207** macro-F1 on **all five** families, at 5.5-8.8
combined sd. Architecture carries real, replicated class signal. It is weak, but it is
not nothing, and it is the positive result the write-up should state plainly.

## Also fixed here

`download_unfiltered.py` wrote `*_phase_only.h5` with `feature_set="length_only"`, so the
phase arm has never actually run. The committed numbers are *consistent with* that (
`phase_only` and `length_only` agree within noise on all five families) but do not prove
it, since phase features could independently be uninformative. The definitive check is
one line: `h5py.File("X_phase_only.h5")[key].shape` is 2 for the length features and 3
for the phase fractions. Worth re-running either way: intron phase is the one
architecture feature that mean-pooled per-exon embeddings *cannot* carry, which is why
`exon_architecture.py` calls it "the actual genealogical signal".
