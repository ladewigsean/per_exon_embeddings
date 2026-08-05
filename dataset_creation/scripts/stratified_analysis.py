#!/usr/bin/env python3
"""
stratified_analysis.py -- does any arm degrade MORE SLOWLY than the sequence baseline as
test-to-train identity drops?

That is the whole question of the identity-stratified experiment. Intron phase and exon
architecture can survive after sequence has diverged, so if architecture carries
ancestral signal it can only show up where sequence has washed out. Everywhere else the
answer is baked in by design: a frozen sequence model on a sequence-solvable task.

How to plot a BINARY outcome (the thing that blocks people here)
---------------------------------------------------------------
You cannot box-plot `correct`. A box plot shows the spread of a CONTINUOUS variable, and
within an identity bin `correct` only ever takes the values 0 and 1, so every quartile
collapses onto 0 or 1 and the plot says nothing. "Stratified" never meant box-plot the
correctness; it meant report the METRIC AS A FUNCTION OF identity. Three ways, in the
order you should trust them:

  1. PAIRED DELTA (primary, `delta` plot). Two arms are evaluated on the SAME test
     proteins, so compare them per protein, not per curve. Plot
     accuracy(arm) - accuracy(baseline) per identity bin with a paired bootstrap CI.
     Two independent noisy curves eyeballed side by side will not resolve a 3-point
     difference; their paired difference often will, because the protein-to-protein
     variance cancels. It also cancels a real confound: class composition shifts across
     identity bins (the hardest bin is not a random sample of classes), which biases
     each raw curve but not their difference, since both arms see the same proteins.

  2. BINNED ACCURACY + WILSON CI (`accuracy` plot). What a bare binned-average plot
     becomes once it is honest. The lowest-identity bin is always the smallest, so its
     point is the least reliable exactly where it matters most; the CI and the printed
     n say so. Wilson rather than normal-approximation because n is small and p is near
     0 or 1 in the tails, where the normal approximation gives nonsense (intervals
     outside [0, 1]).

  3. PER-EXAMPLE SCORE (`score` plot) -- and THIS is where a box plot is right. Binary
     correctness throws away most of the information the model gives you. A continuous
     per-example score does not, and it has a genuine spread inside a bin, so a box plot
     per bin per arm works and has much more power than 1/0.

     On distrusting softmax: your caution is right for CALIBRATION claims ("the model is
     90 % sure"), and it is not what this plot does. The score used here is the margin
     (true-class output minus the best competing output), which is taken before any
     softmax, so no calibration assumption enters.

     But raw margins are NOT comparable ACROSS arms: each arm is a separately trained
     network with its own output scale, and these models are trained with MSE against
     one-hot targets (runners_eval default criterion), so the outputs are not logits in
     the cross-entropy sense either. `--score-normalise rank` (the default) therefore
     converts each arm's margins to within-arm percentile ranks before plotting, which
     is scale-free and keeps the comparison honest. Pass `--score-normalise none` to see
     the raw values for a single arm. Read the score plot as "which proteins does this
     arm find hard, relative to the rest of its own test set", and keep ACCURACY as the
     primary endpoint; if the two disagree, believe accuracy and find out why.

Metric note: this reports ACCURACY per bin, not macro-F1. Macro-F1 within a bin is
unstable, because a bin routinely contains 0-2 examples of some class and the per-class
recall for those is then 0 or 1. Report macro-F1 on the FULL test set and accuracy per
bin, and say so.

Non-independence: test proteins from the same cluster are not independent draws. Pass
`--cluster-col` and the bootstrap resamples CLUSTERS instead of proteins, which is the
honest unit. Without it the CIs are too narrow.

    python stratified_analysis.py --identity split_ident.csv \
        --predictions per_prot=preds/per_prot.csv per_exon=preds/per_exon.csv \
                      per_prot_meta=preds/per_prot_meta.csv meta_only=preds/meta_only.csv \
        --baseline per_prot --out-prefix results/HOX

All statistics here are pure stdlib and unit-tested in test_stratified_helpers.py;
pandas/matplotlib are imported inside main().
"""
import argparse
import csv
import math
import random
from collections import defaultdict

DEFAULT_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]


def wilson_interval(k, n, z=1.96):
    """95 % Wilson score interval for k successes in n trials -> (lo, hi).

    Stays inside [0, 1] and stays sensible at k = 0 or k = n, unlike the normal
    approximation, which is why it is used for the sparse low-identity bins.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def assign_bins(values, edges):
    """-> list of bin indices; None for values outside [edges[0], edges[-1]].

    Half-open [lo, hi) except the last bin, which is closed so identity 1.0 is kept.
    """
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
            continue
        idx = None
        for i in range(len(edges) - 1):
            hi_closed = (i == len(edges) - 2)
            if edges[i] <= v < edges[i + 1] or (hi_closed and v == edges[i + 1]):
                idx = i
                break
        out.append(idx)
    return out


def bin_labels(edges):
    return [f"{edges[i]:.0%}-{edges[i + 1]:.0%}" for i in range(len(edges) - 1)]


def _resample_units(units, rng):
    """Draw len(units) units with replacement -> flat list of member indices."""
    picked = []
    n = len(units)
    for _ in range(n):
        picked.extend(units[rng.randrange(n)])
    return picked


def paired_bootstrap_delta(correct_a, correct_b, groups=None, n_boot=2000, seed=0,
                           alpha=0.05):
    """Paired bootstrap CI for mean(correct_a) - mean(correct_b).

    `correct_a` / `correct_b` are aligned 0/1 lists over the SAME examples -- that
    pairing is the point, so resampling picks an example (or cluster) and takes BOTH
    arms' outcomes for it, keeping the correlation between arms intact.

    `groups`: optional cluster id per example. Given, whole clusters are resampled, which
    is the honest unit when test proteins share clusters.

    Returns (delta, lo, hi). NaNs if there is nothing to resample.
    """
    n = len(correct_a)
    if n == 0 or n != len(correct_b):
        return (float("nan"), float("nan"), float("nan"))
    delta = sum(correct_a) / n - sum(correct_b) / n

    if groups is None:
        units = [[i] for i in range(n)]
    else:
        by_group = defaultdict(list)
        for i, g in enumerate(groups):
            by_group[g].append(i)
        units = list(by_group.values())
    if len(units) < 2:
        return (delta, float("nan"), float("nan"))

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = _resample_units(units, rng)
        m = len(idx)
        if m == 0:
            continue
        deltas.append(sum(correct_a[i] for i in idx) / m
                      - sum(correct_b[i] for i in idx) / m)
    if not deltas:
        return (delta, float("nan"), float("nan"))
    deltas.sort()
    lo = deltas[int(alpha / 2 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int((1 - alpha / 2) * len(deltas)))]
    return (delta, lo, hi)


def _ols_slope(xs, ys):
    """Least-squares slope of ys on xs; None if x has no variance."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def slope_difference(identity, correct_a, correct_b, groups=None, n_boot=2000, seed=0,
                     alpha=0.05):
    """Paired bootstrap CI for slope(arm A) - slope(arm B) of correctness on identity.

    This is the bin-free version of "does A decay more slowly than B": a NEGATIVE
    difference means A's accuracy rises less steeply with identity, i.e. A depends less
    on having a close homolog in train, which is the hypothesis.

    Fits a linear probability model (OLS on a 0/1 outcome), which is crude in absolute
    terms but adequate for comparing two slopes on identical x values; the binned paired
    delta above is the assumption-free version and stays primary.

    TWO ways this misleads if read alone, which is why the caller suppresses the
    significance star unless the two arms have comparable overall accuracy:
      - CEILING: an arm that is right almost everywhere has slope ~0 because it cannot
        rise, not because it is identity-independent.
      - FLOOR: an arm near chance ALSO has slope ~0, for the same non-reason. A weak arm
        (meta_only sits at 0.18-0.27 macro-F1 against per_prot's 0.53-0.82) will show a
        strongly negative slope delta purely because it is bad everywhere. That is not
        robustness to low identity.
    Always read this next to the accuracy plot.
    """
    sa = _ols_slope(identity, correct_a)
    sb = _ols_slope(identity, correct_b)
    if sa is None or sb is None:
        return (float("nan"), float("nan"), float("nan"))
    point = sa - sb

    n = len(identity)
    if groups is None:
        units = [[i] for i in range(n)]
    else:
        by_group = defaultdict(list)
        for i, g in enumerate(groups):
            by_group[g].append(i)
        units = list(by_group.values())
    if len(units) < 2:
        return (point, float("nan"), float("nan"))

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample_units(units, rng)
        xs = [identity[i] for i in idx]
        ra = _ols_slope(xs, [correct_a[i] for i in idx])
        rb = _ols_slope(xs, [correct_b[i] for i in idx])
        if ra is not None and rb is not None:
            diffs.append(ra - rb)
    if not diffs:
        return (point, float("nan"), float("nan"))
    diffs.sort()
    lo = diffs[int(alpha / 2 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int((1 - alpha / 2) * len(diffs)))]
    return (point, lo, hi)


def read_predictions(path):
    """Read a per-example prediction CSV -> {identifier: row dict}.

    Required columns: identifier, true_label, pred_label.
    Optional: margin, nll (used for the per-example score plot).
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    for col in ("identifier", "true_label", "pred_label"):
        if col not in rows[0]:
            raise SystemExit(f"{path} needs a '{col}' column (have: {list(rows[0])})")
    out = {}
    for r in rows:
        rec = {"correct": 1 if r["true_label"] == r["pred_label"] else 0,
               "true_label": r["true_label"]}
        for score in ("margin", "nll"):
            if r.get(score) not in (None, ""):
                try:
                    rec[score] = float(r[score])
                except ValueError:
                    pass
        out[r["identifier"]] = rec
    return out


def main(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = args.bins
    labels = bin_labels(edges)

    ident_df = pd.read_csv(args.identity, dtype={args.id_col: str})
    for col in (args.id_col, args.identity_col):
        if col not in ident_df.columns:
            raise SystemExit(f"column '{col}' not in {args.identity} "
                             f"(have: {list(ident_df.columns)})")
    ident_df = ident_df[ident_df[args.identity_col].notna()]
    identity_by_id = dict(zip(ident_df[args.id_col], ident_df[args.identity_col]))
    cluster_by_id = (dict(zip(ident_df[args.id_col], ident_df[args.cluster_col]))
                     if args.cluster_col and args.cluster_col in ident_df.columns else None)
    if args.cluster_col and cluster_by_id is None:
        print(f"WARNING: --cluster-col '{args.cluster_col}' not in {args.identity}; "
              f"bootstrapping over proteins instead. CIs will be too narrow.")

    arms = {}
    for spec in args.predictions:
        if "=" not in spec:
            raise SystemExit(f"--predictions wants name=path, got '{spec}'")
        name, path = spec.split("=", 1)
        arms[name] = read_predictions(path)
    if args.baseline not in arms:
        raise SystemExit(f"--baseline '{args.baseline}' is not among {list(arms)}")

    # Restrict to proteins scored by EVERY arm and carrying an identity value, so all
    # arms are compared on exactly the same set -- the pairing depends on it.
    common = set(identity_by_id)
    for preds in arms.values():
        common &= set(preds)
    common = sorted(common)
    if not common:
        raise SystemExit("no identifiers shared between the identity CSV and every "
                         "prediction CSV -- check that they use the same ids")
    for name, preds in arms.items():
        dropped = len(preds) - len(common)
        if dropped:
            print(f"note: {name} has {dropped} example(s) not shared by all arms; excluded")
    print(f"{len(common)} test proteins scored by all {len(arms)} arms")

    identity = [float(identity_by_id[i]) for i in common]
    groups = [cluster_by_id[i] for i in common] if cluster_by_id else None
    if groups is None:
        print("NOTE: no --cluster-col, so the bootstrap resamples PROTEINS. Proteins in "
              "one cluster are not independent draws, so the CIs below are too narrow. "
              "Re-run cluster_split.py to get a 'cluster' column.")
    idx_by_bin = defaultdict(list)
    binned = assign_bins(identity, edges)
    for pos, b in enumerate(binned):
        if b is not None:
            idx_by_bin[b].append(pos)
    outside = sum(1 for b in binned if b is None)
    if outside:
        print(f"NOTE: {outside} of {len(binned)} proteins fall outside --bins "
              f"[{edges[0]}, {edges[-1]}]; they are excluded from the per-bin table but "
              f"still included in the slope comparison.")
    if not idx_by_bin:
        raise SystemExit(f"no protein falls inside --bins [{edges[0]}, {edges[-1]}]. "
                         f"Is '{args.identity_col}' a fraction (0-1) rather than a "
                         f"percentage (0-100)?")

    correct = {name: [arms[name][i]["correct"] for i in common] for name in arms}
    overall_acc = {name: sum(v) / len(v) for name, v in correct.items()}

    # ---- table ----------------------------------------------------------------
    records = []
    print(f"\nper-bin accuracy (baseline = {args.baseline})")
    for b in range(len(labels)):
        rows_in_bin = idx_by_bin.get(b, [])
        n = len(rows_in_bin)
        print(f"\n  {labels[b]}  n={n}")
        if n == 0:
            continue
        base_c = [correct[args.baseline][i] for i in rows_in_bin]
        for name in arms:
            arm_c = [correct[name][i] for i in rows_in_bin]
            k = sum(arm_c)
            lo, hi = wilson_interval(k, n)
            rec = {"bin": labels[b], "bin_lo": edges[b], "bin_hi": edges[b + 1],
                   "arm": name, "n": n, "n_correct": k, "accuracy": k / n,
                   "wilson_lo": lo, "wilson_hi": hi}
            if name != args.baseline:
                g = [groups[i] for i in rows_in_bin] if groups else None
                d, dlo, dhi = paired_bootstrap_delta(arm_c, base_c, groups=g,
                                                     n_boot=args.n_boot, seed=args.seed)
                rec.update({"delta_vs_baseline": d, "delta_lo": dlo, "delta_hi": dhi})
                sig = "" if (dlo <= 0 <= dhi or math.isnan(dlo)) else "  *"
                print(f"    {name:<24} acc {k / n:.3f} [{lo:.3f},{hi:.3f}]   "
                      f"delta {d:+.3f} [{dlo:+.3f},{dhi:+.3f}]{sig}")
            else:
                print(f"    {name:<24} acc {k / n:.3f} [{lo:.3f},{hi:.3f}]   (baseline)")
            records.append(rec)

    # ---- bin-free slope comparison --------------------------------------------
    print(f"\nslope of correctness on identity (negative delta = decays more slowly "
          f"than {args.baseline}):")
    slope_records = []
    for name in arms:
        if name == args.baseline:
            continue
        pt, lo, hi = slope_difference(identity, correct[name], correct[args.baseline],
                                      groups=groups, n_boot=args.n_boot, seed=args.seed)
        # An arm far below the baseline overall has slope ~0 because it is bad
        # everywhere, not because it is identity-independent -- exactly the reading this
        # number invites. Withhold the star and say why rather than let it be misread.
        gap = overall_acc[args.baseline] - overall_acc[name]
        floored = gap > args.slope_acc_tolerance
        sig = "" if (lo <= 0 <= hi or math.isnan(lo)) else "  *"
        note = ""
        if floored:
            sig = ""
            note = (f"   [no star: overall accuracy {overall_acc[name]:.3f} vs baseline "
                    f"{overall_acc[args.baseline]:.3f}; a weak arm has a flat slope "
                    f"regardless]")
        print(f"  {name:<24} slope delta {pt:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}{note}")
        slope_records.append({"arm": name, "slope_delta_vs_baseline": pt,
                              "slope_lo": lo, "slope_hi": hi,
                              "overall_accuracy": overall_acc[name],
                              "baseline_overall_accuracy": overall_acc[args.baseline],
                              "accuracy_gap_suppresses_star": floored})
    n_tests = sum(1 for r in records if r.get("delta_vs_baseline") is not None) + len(slope_records)
    print(f"\n* = 95 % CI excludes 0, UNCORRECTED for multiplicity ({n_tests} comparisons "
          f"here, so ~{max(1, round(0.05 * n_tests))} spurious star(s) expected). Treat a "
          f"lone star as a lead, not a result; the shape of the delta curve across bins "
          f"is the evidence.")

    out_csv = f"{args.out_prefix}_stratified.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    if slope_records:
        pd.DataFrame(slope_records).to_csv(f"{args.out_prefix}_slopes.csv", index=False)
        print(f"wrote {args.out_prefix}_slopes.csv")

    # ---- plots ----------------------------------------------------------------
    df = pd.DataFrame(records)
    present = [b for b in range(len(labels)) if idx_by_bin.get(b)]
    xs = list(range(len(present)))
    xlabels = [f"{labels[b]}\nn={len(idx_by_bin[b])}" for b in present]

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in arms:
        sub = df[df["arm"] == name].set_index("bin")
        ys, los, his = [], [], []
        for b in present:
            r = sub.loc[labels[b]]
            ys.append(r["accuracy"])
            los.append(r["accuracy"] - r["wilson_lo"])
            his.append(r["wilson_hi"] - r["accuracy"])
        ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=name)
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("max identity of test protein to any train protein")
    ax.set_ylabel("accuracy (Wilson 95 % CI)")
    ax.set_title(f"{args.title or args.out_prefix}: accuracy vs identity to train")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_accuracy.png", dpi=150)
    print(f"wrote {args.out_prefix}_accuracy.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in arms:
        if name == args.baseline:
            continue
        sub = df[df["arm"] == name].set_index("bin")
        ys, los, his = [], [], []
        for b in present:
            r = sub.loc[labels[b]]
            ys.append(r["delta_vs_baseline"])
            los.append(r["delta_vs_baseline"] - r["delta_lo"])
            his.append(r["delta_hi"] - r["delta_vs_baseline"])
        ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=name)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("max identity of test protein to any train protein")
    ax.set_ylabel(f"accuracy minus {args.baseline} (paired bootstrap 95 % CI)")
    ax.set_title(f"{args.title or args.out_prefix}: paired difference vs {args.baseline}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_delta.png", dpi=150)
    print(f"wrote {args.out_prefix}_delta.png")

    # Per-example score: the one plot where a box plot is the right choice.
    score = args.score
    if args.score_normalise == "rank":
        # Each arm is a separately trained net with its own output scale (and these are
        # MSE-trained, so the outputs are not logits), which makes raw margins
        # incomparable BETWEEN arms. Percentile-rank within each arm first.
        for name in arms:
            vals = [(i, arms[name][i][score]) for i in common if score in arms[name][i]]
            if not vals:
                continue
            order = sorted(range(len(vals)), key=lambda k: vals[k][1])
            denom = max(1, len(vals) - 1)
            for rank, k in enumerate(order):
                arms[name][vals[k][0]][score] = rank / denom
    if any(score in arms[name][i] for name in arms for i in common[:50]):
        fig, ax = plt.subplots(figsize=(11, 5))
        names = list(arms)
        width = 0.8 / len(names)
        for j, name in enumerate(names):
            data, pos = [], []
            for k, b in enumerate(present):
                vals = [arms[name][common[i]][score] for i in idx_by_bin[b]
                        if score in arms[name][common[i]]]
                if vals:
                    data.append(vals)
                    pos.append(k + (j - (len(names) - 1) / 2) * width)
            if data:
                bp = ax.boxplot(data, positions=pos, widths=width * .9, patch_artist=True,
                                showfliers=False, manage_ticks=False)
                colour = plt.cm.tab10(j % 10)
                for box in bp["boxes"]:
                    box.set_facecolor(colour); box.set_alpha(.7)
                ax.plot([], [], color=colour, lw=6, label=name)
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel("max identity of test protein to any train protein")
        ax.set_ylabel(f"per-example {score}"
                      + (" (within-arm percentile rank)" if args.score_normalise == "rank" else ""))
        ax.set_title(f"{args.title or args.out_prefix}: per-example {score} by identity")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{args.out_prefix}_{score}.png", dpi=150)
        print(f"wrote {args.out_prefix}_{score}.png")
    else:
        print(f"\nno '{score}' column in the prediction CSVs, so the per-example score "
              f"box plot was skipped. Re-run test_model with --pred-out to emit it; the "
              f"score plot is the higher-power version of the accuracy plot.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--identity", required=True,
                   help="CSV with identifier + identity-to-train (identity_to_train.py).")
    p.add_argument("--predictions", required=True, nargs="+", metavar="NAME=PATH",
                   help="Per-example prediction CSVs, one per arm, e.g. per_prot=p.csv")
    p.add_argument("--baseline", required=True,
                   help="Arm every other arm is differenced against (e.g. per_prot).")
    p.add_argument("--out-prefix", required=True, help="Prefix for output CSVs/PNGs.")
    p.add_argument("--id-col", default="identifier")
    p.add_argument("--identity-col", default="train_max_pident_cov",
                   help="Coverage-weighted identity by default: a short local hit at "
                        "100 %% identity is not leakage. Use train_max_pident for raw.")
    p.add_argument("--cluster-col", default="cluster",
                   help="Cluster id column; bootstrap resamples clusters, not proteins.")
    p.add_argument("--score", default="margin", choices=["margin", "nll"],
                   help="Per-example score for the box plot (default: margin).")
    p.add_argument("--score-normalise", default="rank", choices=["rank", "none"],
                   help="'rank' percentile-ranks each arm's scores before plotting, so "
                        "arms with different output scales stay comparable (default).")
    p.add_argument("--slope-acc-tolerance", type=float, default=0.05,
                   help="Suppress the slope significance star when an arm's overall "
                        "accuracy is more than this below the baseline's (default 0.05).")
    p.add_argument("--bins", type=float, nargs="+", default=DEFAULT_BINS)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", default=None)
    main(p.parse_args())
