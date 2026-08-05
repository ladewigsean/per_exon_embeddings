#!/usr/bin/env python3
"""
identity_to_train.py -- measure, per test protein, its MAXIMUM identity to any train protein.

Why this exists
---------------
`cluster_split.py` claims "no sequence in val/test shares > min-seq-id identity with any
train sequence". That claim is too strong and should not be relied on. `mmseqs
easy-cluster --min-seq-id X` only guarantees X between a member and ITS OWN cluster
REPRESENTATIVE. It does not bound identity between members of two DIFFERENT clusters,
for three separate reasons:

  1. Clustering is greedy set-cover. Which sequence becomes a representative depends on
     length/order, so B can be >X identical to A yet end up in its own cluster.
  2. The k-mer prefilter can miss a pair outright. At ~30 % identity (the twilight zone)
     the default sensitivity is not enough; pairs never get aligned, so they cannot be
     joined. Raise `-s` if you rely on clustering at all.
  3. `-c 0.8 --cov-mode 0` refuses to cluster a pair that does not share 80 % MUTUAL
     coverage, however identical the region they do share. Multi-domain families and
     length-variable families are split apart by coverage, not by identity.

So do not argue about the split -- MEASURE it. This script runs an all-vs-all search of
test (and val) against train and records the real number for every query. Two uses:

  (a) as a filter: drop test proteins above a threshold to get a defensible hard holdout;
  (b) better -- as the X AXIS of the stratified analysis (see stratified_analysis.py).
      Once performance is reported AS A FUNCTION of identity-to-train, residual leakage
      stops being a confound and becomes a covariate you condition on. The stratified
      experiment does NOT need a perfectly clean split, which is the main reason not to
      spend more time tightening one.

Note on the metric: raw `fident` from a LOCAL alignment is misleading on its own -- a
30-residue exact match inside a 600-residue protein scores 1.0 and is not leakage. This
writes both the raw value and a coverage-weighted one (`fident * qcov`); prefer the
latter for filtering, and say which one you used.

Queries with NO hit at all are the HARDEST and most valuable test cases. They are kept
with identity 0.0 and `train_n_hits` 0, never dropped -- dropping them would silently
delete exactly the regime the experiment is about.

    python identity_to_train.py --fasta all.fasta --csv split.csv --out split_ident.csv

If you call mmseqs through WSL from native Windows (--mmseqs-cmd "wsl --exec mmseqs"),
pass --workdir too: the default scratch directory is a Windows temp path that mmseqs
inside WSL cannot open, and it fails with an opaque error. Point --workdir at something
both sides can see.

`aggregate_best_hits()` and `write_split_fastas()` are pure (no mmseqs, no pandas) and
unit-tested in test_stratified_helpers.py.
"""
import argparse
import contextlib
import os
import subprocess
import tempfile

# pandas is imported inside main() so the pure logic stays importable with only the
# stdlib -- same convention as cluster_split.py / exon_architecture.py.

# mmseqs easy-search --format-output field order used throughout this script.
M8_FIELDS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]


def read_fasta(path):
    """-> {identifier: sequence}. Identifier is the first whitespace-delimited token."""
    seqs, cur, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                cur, buf = line[1:].strip().split()[0], []
            else:
                buf.append(line.strip())
    if cur is not None:
        seqs[cur] = "".join(buf)
    return seqs


def write_split_fastas(seqs, id_to_split, workdir, query_splits=(1, 2)):
    """Write train.fasta (split 0) and query.fasta (splits in `query_splits`).

    Returns (train_path, query_path, n_train, n_query). Ids absent from `id_to_split`
    are skipped -- they belong to no split and so are neither target nor query.
    """
    train_path = os.path.join(workdir, "train.fasta")
    query_path = os.path.join(workdir, "query.fasta")
    n_train = n_query = 0
    with open(train_path, "w") as tr, open(query_path, "w") as qu:
        for ident, seq in seqs.items():
            split = id_to_split.get(ident)
            if split is None:
                continue
            if split == 0:
                tr.write(f">{ident}\n{seq}\n")
                n_train += 1
            elif split in query_splits:
                qu.write(f">{ident}\n{seq}\n")
                n_query += 1
    return train_path, query_path, n_train, n_query


def run_mmseqs_search(query_fasta, train_fasta, mmseqs_cmd, workdir,
                      sensitivity=7.5, evalue=10000.0, max_seqs=300):
    """easy-search query vs train -> list of dicts with M8_FIELDS keys.

    Deliberately permissive: max sensitivity and a huge E-value cutoff, because we WANT
    the weak hits. A missed hit here silently becomes "identity 0", i.e. it would fake a
    hard test case that is not actually hard -- the one error mode that would flatter the
    result, so the search is tuned to over-report rather than under-report.
    """
    out_m8 = os.path.join(workdir, "hits.m8")
    tmp = os.path.join(workdir, "tmp_search")
    cmd = mmseqs_cmd.split() + [
        "easy-search", query_fasta, train_fasta, out_m8, tmp,
        "-s", str(sensitivity), "-e", str(evalue), "--max-seqs", str(max_seqs),
        "--format-output", ",".join(M8_FIELDS),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    rows = []
    with open(out_m8) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(M8_FIELDS):
                continue
            r = dict(zip(M8_FIELDS, parts))
            for k in ("fident", "alnlen", "qcov", "tcov", "evalue", "bits"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def aggregate_best_hits(rows, query_ids):
    """Per query: best raw identity, best coverage-weighted identity, hit count.

    Returns {query_id: {"train_max_pident", "train_max_pident_cov", "train_best_hit",
                        "train_n_hits"}} covering EVERY id in `query_ids`, so a query
    with no hit is present with zeros rather than missing (see module docstring).

    "Best" is chosen independently for the two metrics: the highest-identity hit and the
    highest identity*coverage hit need not be the same alignment, and reporting the raw
    identity of the coverage-best hit would understate leakage.
    """
    agg = {q: {"train_max_pident": 0.0, "train_max_pident_cov": 0.0,
               "train_best_hit": "", "train_n_hits": 0} for q in query_ids}
    for r in rows:
        q = r["query"]
        if q not in agg:
            continue
        a = agg[q]
        a["train_n_hits"] += 1
        if r["fident"] > a["train_max_pident"]:
            a["train_max_pident"] = r["fident"]
            a["train_best_hit"] = r["target"]
        cov_weighted = r["fident"] * r["qcov"]
        if cov_weighted > a["train_max_pident_cov"]:
            a["train_max_pident_cov"] = cov_weighted
    return agg


def summarise(agg, label="query"):
    """Print the distribution that decides whether the split is usable."""
    vals = sorted(a["train_max_pident"] for a in agg.values())
    n = len(vals)
    if n == 0:
        print(f"no {label} sequences")
        return
    def pct(p):
        return vals[min(n - 1, int(p * n))]
    no_hit = sum(1 for a in agg.values() if a["train_n_hits"] == 0)
    print(f"\n{label}: n={n}  no-hit-to-train={no_hit} ({100.0 * no_hit / n:.1f} %)")
    print(f"  train_max_pident  min {vals[0]:.3f}  p25 {pct(.25):.3f}  median "
          f"{pct(.5):.3f}  p75 {pct(.75):.3f}  p95 {pct(.95):.3f}  max {vals[-1]:.3f}")
    for thr in (0.3, 0.4, 0.5):
        over = sum(1 for v in vals if v > thr)
        print(f"  above {thr:.0%} identity to train: {over} ({100.0 * over / n:.1f} %)")


@contextlib.contextmanager
def _scratch_dir(workdir):
    """Yield a scratch directory: `workdir` if given, else a self-cleaning temp dir.

    An explicit one is needed when mmseqs runs behind `wsl --exec`, because the default
    temp path is a Windows path that mmseqs inside WSL cannot open.
    """
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        yield workdir
    else:
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp


def main(args):
    import pandas as pd
    df = pd.read_csv(args.csv, dtype={args.id_col: str})
    for col in (args.id_col, args.split_col):
        if col not in df.columns:
            raise SystemExit(f"column '{col}' not in {args.csv} (have: {list(df.columns)})")
    id_to_split = dict(zip(df[args.id_col], df[args.split_col]))
    seqs = read_fasta(args.fasta)

    missing = [i for i in df[args.id_col] if i not in seqs]
    if missing:
        print(f"WARNING: {len(missing)} ids in the CSV have no sequence in the FASTA "
              f"(e.g. {missing[:3]}); they get no identity value.")

    with _scratch_dir(args.workdir) as workdir:
        train_fa, query_fa, n_train, n_query = write_split_fastas(
            seqs, id_to_split, workdir, query_splits=tuple(args.query_splits))
        print(f"{n_train} train sequences, {n_query} query sequences "
              f"(splits {args.query_splits} vs split 0)")
        if n_train == 0 or n_query == 0:
            raise SystemExit("need a non-empty train AND query set -- check --split-col")
        rows = run_mmseqs_search(query_fa, train_fa, args.mmseqs_cmd, workdir,
                                 sensitivity=args.sensitivity, evalue=args.evalue,
                                 max_seqs=args.max_seqs)
        print(f"{len(rows)} alignments reported")
        query_ids = [i for i, s in id_to_split.items()
                     if s in args.query_splits and i in seqs]
        agg = aggregate_best_hits(rows, query_ids)

    for split in args.query_splits:
        sub = {i: a for i, a in agg.items() if id_to_split.get(i) == split}
        summarise(sub, label=f"split {split}")

    for col in ("train_max_pident", "train_max_pident_cov", "train_best_hit",
                "train_n_hits"):
        default = "" if col == "train_best_hit" else 0
        values = [agg.get(i, {}).get(col, default) for i in df[args.id_col]]
        # float, not int, so the blanking below does not trip pandas' incompatible-dtype
        # warning (an error from pandas 3).
        df[col] = values if col == "train_best_hit" else [float(v) for v in values]
    # Train rows have no meaningful identity-to-train; blank them so they cannot be
    # silently plotted as if they were held-out points.
    is_query = df[args.split_col].isin(args.query_splits)
    for col in ("train_max_pident", "train_max_pident_cov", "train_n_hits"):
        df.loc[~is_query, col] = float("nan")

    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  (+ train_max_pident, train_max_pident_cov, "
          f"train_best_hit, train_n_hits)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", required=True, help="All sequences (ids = CSV identifiers).")
    p.add_argument("--csv", required=True, help="Metadata CSV with identifier + test_split.")
    p.add_argument("--out", required=True, help="Output CSV (input + identity columns).")
    p.add_argument("--id-col", default="identifier")
    p.add_argument("--split-col", default="test_split")
    p.add_argument("--query-splits", type=int, nargs="+", default=[1, 2],
                   help="Splits to measure against train (default: 1 2 = val and test).")
    p.add_argument("--mmseqs-cmd", default="mmseqs",
                   help='How to invoke mmseqs, e.g. "wsl --exec mmseqs".')
    p.add_argument("--workdir", default=None,
                   help="Scratch directory. Required with a WSL --mmseqs-cmd, where the "
                        "default Windows temp path is unreadable from inside WSL.")
    p.add_argument("--sensitivity", type=float, default=7.5, help="mmseqs -s (default 7.5).")
    p.add_argument("--evalue", type=float, default=10000.0, help="mmseqs -e (default 10000).")
    p.add_argument("--max-seqs", type=int, default=300, help="mmseqs --max-seqs (default 300).")
    main(p.parse_args())
