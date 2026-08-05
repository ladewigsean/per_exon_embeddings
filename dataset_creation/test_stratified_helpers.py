#!/usr/bin/env python3
"""Unit tests for the pure logic in identity_to_train.py and stratified_analysis.py.

These need no mmseqs, no h5, no GPU, not even pandas -- they check the parts that decide
whether the stratified result is believable: the Wilson intervals and paired bootstrap
that carry the error bars, the slope comparison that IS the hypothesis test, and the
best-hit aggregation that must not silently drop a query with no homolog in train.
Run: python -m pytest test_stratified_helpers.py
"""
import math
import os
import tempfile

from dataset_creation.scripts.identity_to_train import (
    aggregate_best_hits, read_fasta, write_split_fastas)
from dataset_creation.scripts.stratified_analysis import (
    assign_bins, bin_labels, paired_bootstrap_delta, slope_difference, wilson_interval,
    _ols_slope)


# ---------------------------------------------------------------- Wilson intervals

def test_wilson_matches_known_value():
    lo, hi = wilson_interval(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3


def test_wilson_stays_inside_unit_interval_at_the_extremes():
    # The normal approximation gives negative / >1 bounds here; the tails are exactly
    # where the low-identity bins live, so this is the case that matters.
    lo, hi = wilson_interval(0, 8)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson_interval(8, 8)
    assert hi == 1.0 and 0.0 < lo < 1.0


def test_wilson_of_empty_bin_is_nan():
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_narrows_as_n_grows():
    small = wilson_interval(5, 10)
    large = wilson_interval(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


# ---------------------------------------------------------------- binning

def test_bins_are_half_open_with_a_closed_top():
    edges = [0.0, 0.1, 0.2, 1.0]
    assert assign_bins([0.0, 0.05, 0.1, 0.19, 0.2, 0.999, 1.0], edges) == [0, 0, 1, 1, 2, 2, 2]


def test_values_outside_the_range_and_nan_are_dropped_not_misbinned():
    edges = [0.2, 0.4, 0.6]
    assert assign_bins([0.1, 0.7, float("nan"), None, 0.3], edges) == [None, None, None, None, 0]


def test_bin_labels_match_edge_count():
    edges = [0.0, 0.1, 0.3, 1.0]
    assert bin_labels(edges) == ["0%-10%", "10%-30%", "30%-100%"]


# ---------------------------------------------------------------- paired bootstrap

def test_identical_arms_give_zero_delta_and_a_ci_containing_zero():
    a = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    d, lo, hi = paired_bootstrap_delta(a, list(a), n_boot=500, seed=1)
    assert d == 0.0 and lo <= 0 <= hi


def test_perfectly_separated_arms_give_delta_one_with_a_tight_ci():
    n = 30
    d, lo, hi = paired_bootstrap_delta([1] * n, [0] * n, n_boot=500, seed=1)
    assert d == 1.0 and lo == 1.0 and hi == 1.0


def test_delta_recovers_the_true_difference():
    # arm A correct on 8/10, arm B on 5/10 -> delta 0.3, CI must cover it.
    a = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    b = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    d, lo, hi = paired_bootstrap_delta(a, b, n_boot=1000, seed=7)
    assert abs(d - 0.3) < 1e-9 and lo <= 0.3 <= hi


def test_pairing_is_preserved_so_a_constant_offset_has_no_spread():
    # B is A with two extra successes; every resample keeps that pairing, so the delta
    # is identical in every bootstrap draw. Resampling the arms independently would
    # produce a spread here -- this is the test that the pairing is real.
    a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    _, lo, hi = paired_bootstrap_delta(a, b, n_boot=300, seed=3)
    assert lo == hi == 1.0


def test_cluster_bootstrap_is_wider_than_protein_bootstrap():
    # Same data; correlated within cluster. Resampling proteins pretends they are
    # independent draws and understates the interval, so the cluster CI must be wider.
    a, b, groups = [], [], []
    for c in range(10):
        hit = 1 if c % 2 == 0 else 0
        for _ in range(10):
            a.append(hit)
            b.append(0)
            groups.append(f"c{c}")
    _, plo, phi = paired_bootstrap_delta(a, b, groups=None, n_boot=1000, seed=5)
    _, clo, chi = paired_bootstrap_delta(a, b, groups=groups, n_boot=1000, seed=5)
    assert (chi - clo) > (phi - plo)


def test_empty_and_mismatched_inputs_are_nan_not_a_crash():
    for args in (([], []), ([1, 0], [1])):
        d, lo, hi = paired_bootstrap_delta(*args, n_boot=10)
        assert math.isnan(d) and math.isnan(lo) and math.isnan(hi)


# ---------------------------------------------------------------- slope comparison

def test_ols_slope_recovers_a_known_line():
    xs = [0.0, 1.0, 2.0, 3.0]
    assert abs(_ols_slope(xs, [1.0, 3.0, 5.0, 7.0]) - 2.0) < 1e-12


def test_ols_slope_is_none_without_x_variance():
    assert _ols_slope([0.5, 0.5, 0.5], [1, 0, 1]) is None


def test_flat_arm_has_a_negative_slope_delta_against_a_rising_arm():
    # THE hypothesis shape: arm A is identity-independent, baseline B needs a close
    # homolog. Negative slope delta = A decays more slowly.
    identity, a, b = [], [], []
    for i in range(100):
        x = i / 100.0
        identity.append(x)
        a.append(1 if i % 2 == 0 else 0)          # ~50 % everywhere
        b.append(1 if x > 0.5 else 0)             # only right at high identity
    pt, lo, hi = slope_difference(identity, a, b, n_boot=500, seed=11)
    assert pt < 0 and hi < 0, "a flat arm vs a rising baseline must be significantly negative"


def test_two_identical_arms_have_a_slope_delta_of_zero():
    identity = [i / 50.0 for i in range(50)]
    a = [1 if x > 0.5 else 0 for x in identity]
    pt, lo, hi = slope_difference(identity, a, list(a), n_boot=300, seed=2)
    assert pt == 0.0 and lo <= 0 <= hi


# ---------------------------------------------------------------- best-hit aggregation

def _row(q, t, fident, qcov):
    return {"query": q, "target": t, "fident": fident, "alnlen": 100.0,
            "qcov": qcov, "tcov": qcov, "evalue": 1e-3, "bits": 100.0}


def test_query_with_no_hit_is_kept_at_zero_identity():
    # The single most important behaviour: no-hit queries are the HARDEST test cases.
    # Dropping them would delete exactly the regime the experiment is about.
    agg = aggregate_best_hits([_row("q1", "t1", 0.5, 1.0)], ["q1", "q2"])
    assert set(agg) == {"q1", "q2"}
    assert agg["q2"]["train_max_pident"] == 0.0
    assert agg["q2"]["train_n_hits"] == 0
    assert agg["q2"]["train_best_hit"] == ""


def test_max_identity_and_hit_count_are_taken_over_all_hits():
    rows = [_row("q1", "t1", 0.25, 1.0), _row("q1", "t2", 0.61, 1.0),
            _row("q1", "t3", 0.40, 1.0)]
    agg = aggregate_best_hits(rows, ["q1"])
    assert abs(agg["q1"]["train_max_pident"] - 0.61) < 1e-9
    assert agg["q1"]["train_best_hit"] == "t2"
    assert agg["q1"]["train_n_hits"] == 3


def test_coverage_weighting_demotes_a_short_high_identity_local_hit():
    # A 100 % identical hit over 10 % of the query is not leakage; the raw number says
    # 1.0 and the coverage-weighted one says 0.1. Both are reported for exactly this.
    rows = [_row("q1", "short", 1.00, 0.10), _row("q1", "full", 0.45, 0.95)]
    agg = aggregate_best_hits(rows, ["q1"])
    assert agg["q1"]["train_max_pident"] == 1.00
    assert agg["q1"]["train_best_hit"] == "short"
    assert abs(agg["q1"]["train_max_pident_cov"] - 0.4275) < 1e-9


def test_hits_from_queries_outside_the_query_set_are_ignored():
    agg = aggregate_best_hits([_row("stray", "t1", 0.9, 1.0)], ["q1"])
    assert set(agg) == {"q1"} and agg["q1"]["train_n_hits"] == 0


# ---------------------------------------------------------------- fasta handling

def test_read_fasta_and_split_by_partition():
    with tempfile.TemporaryDirectory() as d:
        fa = os.path.join(d, "all.fasta")
        with open(fa, "w") as fh:
            fh.write(">a description here\nMKV\nLLA\n>b\nGGG\n>c\nWWW\n>d\nYYY\n")
        seqs = read_fasta(fa)
        assert seqs == {"a": "MKVLLA", "b": "GGG", "c": "WWW", "d": "YYY"}

        # d has no split entry -> belongs to neither train nor query.
        train, query, n_train, n_query = write_split_fastas(
            seqs, {"a": 0, "b": 0, "c": 2}, d, query_splits=(1, 2))
        assert (n_train, n_query) == (2, 1)
        assert set(read_fasta(train)) == {"a", "b"}
        assert set(read_fasta(query)) == {"c"}
