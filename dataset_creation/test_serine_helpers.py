#!/usr/bin/env python3
"""Unit tests for the pure logic in cluster_split.py and exon_architecture.py.

These need no mmseqs, no h5, no GPU -- they check the parts that decide correctness:
the cluster-aware assignment (no leakage, label-stratified) and the architecture-
feature maths (exon lengths, intron phase). Run: python -m pytest test_serine_helpers.py
"""
from cluster_split import assign_clusters_stratified
from exon_architecture import architecture_features, FEATURE_NAMES


def test_no_cluster_is_split_across_partitions():
    # Two big single-label clusters + singletons; every cluster must stay intact.
    cluster_to_items = {
        "cA": [f"a{i}" for i in range(20)],
        "cB": [f"b{i}" for i in range(20)],
        "cC": [f"c{i}" for i in range(4)],
        "cD": ["d0"],
    }
    labels = {i: "S9" for i in sum(cluster_to_items.values(), [])}
    id_to_split = assign_clusters_stratified(cluster_to_items, labels)
    for items in cluster_to_items.values():
        splits = {id_to_split[i] for i in items}
        assert len(splits) == 1, "a cluster was split across partitions (leakage!)"


def test_every_label_reaches_every_split_when_enough_clusters():
    # 9 clusters per label -> train/val/test should each receive some of each label.
    cluster_to_items, labels = {}, {}
    for lab in ("S9", "S10", "S28"):
        for k in range(9):
            cid = f"{lab}_{k}"
            members = [f"{lab}_{k}_{j}" for j in range(5)]
            cluster_to_items[cid] = members
            for m in members:
                labels[m] = lab
    id_to_split = assign_clusters_stratified(cluster_to_items, labels)
    seen = {lab: set() for lab in ("S9", "S10", "S28")}
    for ident, s in id_to_split.items():
        seen[labels[ident]].add(s)
    for lab, splits in seen.items():
        assert splits == {0, 1, 2}, f"label {lab} missing from a split: {splits}"


def test_three_clusters_reach_all_splits():
    # The greedy alone would dump all 3 in train; the rescue must spread them 1/1/1.
    cluster_to_items = {f"c{k}": [f"c{k}_{j}" for j in range(5)] for k in range(3)}
    labels = {m: "S9" for ms in cluster_to_items.values() for m in ms}
    id_to_split = assign_clusters_stratified(cluster_to_items, labels)
    assert {s for s in id_to_split.values()} == {0, 1, 2}


def test_two_clusters_reach_train_and_val_not_test():
    # With only 2 clusters you can't cover all 3 splits; expect train+val, test empty.
    cluster_to_items = {f"c{k}": [f"c{k}_{j}" for j in range(5)] for k in range(2)}
    labels = {m: "S9" for ms in cluster_to_items.values() for m in ms}
    id_to_split = assign_clusters_stratified(cluster_to_items, labels)
    splits = {s for s in id_to_split.values()}
    assert splits == {0, 1}, f"expected train+val for 2 clusters, got {splits}"


def test_roughly_80_10_10():
    cluster_to_items = {f"c{k}": [f"c{k}_{j}" for j in range(2)] for k in range(50)}
    labels = {m: "S9" for ms in cluster_to_items.values() for m in ms}
    id_to_split = assign_clusters_stratified(cluster_to_items, labels)
    n = len(id_to_split)
    train = sum(s == 0 for s in id_to_split.values()) / n
    assert 0.7 <= train <= 0.9, f"train fraction off: {train}"


def test_architecture_three_exons_phase_zero():
    # exons of nt-length 300,150,240 -> cut_pos cumulative [300,450], cds_nt 690.
    feats = dict(zip(FEATURE_NAMES, architecture_features([300, 450], 690)))
    assert feats["n_exons"] == 3
    assert feats["total_cds_nt"] == 690
    assert feats["first_exon_nt"] == 300
    assert feats["last_exon_nt"] == 240
    assert feats["exon_len_max"] == 300
    assert feats["exon_len_min"] == 150
    assert feats["single_exon"] == 0.0
    # both introns at multiples of 3 -> phase 0
    assert feats["phase0_frac"] == 1.0
    assert feats["phase1_frac"] == 0.0


def test_architecture_intron_phase():
    # cut_pos 301 (phase 1) and 452 (phase 2) -> mixed phases.
    feats = dict(zip(FEATURE_NAMES, architecture_features([301, 452], 690)))
    assert feats["phase1_frac"] == 0.5
    assert feats["phase2_frac"] == 0.5
    assert feats["phase0_frac"] == 0.0


def test_architecture_single_exon():
    feats = dict(zip(FEATURE_NAMES, architecture_features([], 600)))
    assert feats["n_exons"] == 1
    assert feats["single_exon"] == 1.0
    assert feats["exon_len_mean"] == 600
    assert feats["phase0_frac"] == 0.0  # no introns -> all phase fracs 0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\nall {len(fns)} tests passed")
    sys.exit(0)
