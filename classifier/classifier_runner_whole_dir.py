#!/usr/bin/env python3

import argparse
import pandas as pd
import pathlib
import os
import re
import sys
from scripts.custom_datasets import split_dataset_into_subsets, MultiClassDataset
from scripts.runners_eval import run, test_model, train_model
YAML_FOLDER = "yaml"
MODEL_WEIGHTS_FOLDER = "model_weights"
OUTPUT_CSVS_FOLDER = "output_csvs"
PREDICTIONS_FOLDER = "predictions"
# mean_*/std_* are VALIDATION metrics over 5 seeds; test_* is a single held-out
# evaluation of the val-best checkpoint (see the note next to the test_model call).
COLUMNS = ["method", "mean_acc", "std_acc", "mean_macro_f1", "std_macro_f1",
           "test_acc", "test_macro_f1", "test_status"]


def find_surviving_checkpoint(entity):
    """The one checkpoint train_model kept for `entity`, or None if that is ambiguous.

    train_model names them f"val_seed_{seed}_{wandb_project}.pt" and deletes all but the
    val-best, so a COMPLETED arm leaves exactly one file behind. But --force_test exists
    to recover runs that did not complete, which is exactly when several seeds survive
    and there is no record of which was best. Refuse to guess rather than silently score
    an arbitrary seed and report it as the result.

    The regex matters too: a bare glob lets `*` swallow a family prefix, so entity
    "HOX_per_prot" would match "val_seed_1_NCBIHOX_per_prot_test.pt" from a different
    dataset (model_weights/ is one shared folder and the repo has both HOX and NCBI_HOX).
    """
    pattern = re.compile(rf"val_seed_\d+_{re.escape(entity)}_test\.pt$")
    matches = sorted(p for p in pathlib.Path(MODEL_WEIGHTS_FOLDER).glob("val_seed_*.pt")
                     if pattern.match(p.name))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        print(f"WARNING: {len(matches)} surviving checkpoints for {entity} "
              f"({', '.join(p.name for p in matches)}); cannot tell which seed was "
              f"val-best, so refusing to guess. Retrain this arm.", file=sys.stderr)
    return None


if __name__ == '__main__':
    #python classifier_runner.py --entity SPE_per_exon --nn_model Transformer --h5 input_data/SPE/SerProtEuk_per_exon.h5 --csv input_data/SPE/SerProtEuk.csv
    parser = argparse.ArgumentParser(description="Train or optimize a multi-class SCPP classifier.")
    parser.add_argument("--dir", required=True, help="Path to dataset directory")

    parser.add_argument("--project", default="per-exon-testing")
    parser.add_argument("--wandb_disable", action="store_true")

    parser.add_argument("--hpo_trials",type=int, default=30,help="number of hpo trials. default 30")
    parser.add_argument("--skip_per_res", action="store_true", help="if flag given, skips over per_res.h5")
    parser.add_argument("--skip_test", action="store_true",
                        help="skip held-out test evaluation (the run then reports VALIDATION "
                             "numbers only, which is what this script used to do)")
    parser.add_argument("--force_test", action="store_true",
                        help="for arms already in the output CSV but with no test_acc, re-run "
                             "ONLY the test stage on the surviving checkpoint. Skips HPO and "
                             "training, so it costs a forward pass rather than a whole sweep.")
    args = parser.parse_args()
    dir_name = pathlib.Path(args.dir).stem
    print(dir_name)
    csv_file = None
    csvlist = [path for path in pathlib.Path(args.dir).rglob('*.csv')]
    if len(csvlist) == 0:
        raise FileNotFoundError("No CSV found in given dir")
    elif len(csvlist) > 1:
        raise FileNotFoundError("multiple CSV found in given dir")
    else:
        csv_file = str(csvlist[0])
    h5s = pathlib.Path(args.dir).rglob('*.h5')
    output = os.path.join(OUTPUT_CSVS_FOLDER,f"{dir_name}.csv")

    # Results are held in memory and the CSV is REWRITTEN after every arm, rather than
    # appended to. Appending a wider row onto a narrower existing file produced a ragged
    # CSV that pd.read_csv then refused to open on the next resume, destroying results
    # that cost hours; four of the committed output_csvs also lack a trailing newline, so
    # an append would have concatenated onto the last result row. Rewriting removes both
    # failure modes and costs nothing at this size.
    rows = {}
    if os.path.isfile(output):
        try:
            # dtype=str so legacy cells round-trip as their original text. Reading them
            # as floats and writing them back reformats at ~16 significant digits
            # (0.9519379844961241 -> 0.951937984496124), perturbing values that this
            # script did not compute.
            old_data = pd.read_csv(output, dtype=str)
        except pd.errors.EmptyDataError:
            print(f"WARNING: {output} is empty; starting a fresh results file",
                  file=sys.stderr)
            old_data = None
        if old_data is not None:
            # Older files were written with a 5-column header, and two of them with an
            # EMPTY first header; normalise both shapes onto COLUMNS without losing a row.
            old_data.columns = [COLUMNS[0]] + list(old_data.columns[1:])
            for col in COLUMNS:
                if col not in old_data.columns:
                    old_data[col] = ""
            old_data = old_data[COLUMNS].fillna("")
            for _, r in old_data.iterrows():
                rows[str(r[COLUMNS[0]])] = [r[c] for c in COLUMNS[1:]]
            if len(rows) != len(old_data):
                print(f"WARNING: {len(old_data) - len(rows)} duplicate method row(s) in "
                      f"{output} collapsed to the last occurrence", file=sys.stderr)
            print(f"resuming from {output}: {len(rows)} arms already recorded")

    def flush():
        # Write-then-rename: to_csv truncates in place, so a kill or a full disk during
        # any of the N+1 flushes in a sweep would otherwise leave a truncated results
        # file that pandas cannot reopen, with no way back.
        os.makedirs(OUTPUT_CSVS_FOLDER, exist_ok=True)
        tmp = output + ".tmp"
        pd.DataFrame([[k] + v for k, v in rows.items()],
                     columns=COLUMNS).to_csv(tmp, index=False)
        os.replace(tmp, output)

    flush()
    for h5 in h5s:
        entity = str(h5.stem)

        # --force_test re-runs ONLY the cheap test stage for arms trained before the
        # runner evaluated split 2, so held-out numbers do not cost a repeat of the
        # HPO + 5-seed sweep.
        test_only = False
        if entity in rows:
            already_tested = str(rows[entity][COLUMNS[1:].index("test_acc")]).strip() != ""
            if args.force_test and not args.skip_test and not already_tested:
                test_only = True
                print(f"--force_test: evaluating {entity} on split 2 without retraining")
            else:
                continue
        h5 = str(h5)
        if "per_res" in entity and args.skip_per_res:
            print(f"Skip_per_res flag given, Skipping {h5}\n")
            continue

        # Check the artefacts --force_test depends on BEFORE opening the h5. Building the
        # dataset first means a per_res arm reads a multi-GB file and scans every key,
        # only to bail two lines later because there is no checkpoint to score.
        yaml_path = best_model = None
        if test_only:
            yaml_path = os.path.join(YAML_FOLDER, f"{entity}_HPO.yaml")
            best_model = find_surviving_checkpoint(entity)
            if not os.path.isfile(yaml_path) or best_model is None:
                reason = (f"missing {yaml_path}" if not os.path.isfile(yaml_path)
                          else f"no unambiguous checkpoint in {MODEL_WEIGHTS_FOLDER}")
                print(f"WARNING: cannot --force_test {entity}: {reason}", file=sys.stderr)
                rows[entity] = list(rows[entity][:4]) + ["", "", "no_checkpoint"]
                flush()
                continue

        train_dataset, val_dataset, test_dataset, max_length =split_dataset_into_subsets(MultiClassDataset(embeddings_path=h5,csv_path=csv_file))
        embed_size = train_dataset.embedding_dim
        print(f"Max Length: {max_length}")
        if max_length>1:
            nn_model = "Transformer"
        else:
            nn_model = "Basic"

        if test_only:
            # yaml_path / best_model were resolved and validated above.
            data = rows[entity][:4]
        else:
            yaml_path = run(train_dataset,entity+"_HPO",args.project,nn_model=nn_model,n_trials=args.hpo_trials,num_epochs=35,patience=5,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,yaml_folder=YAML_FOLDER, checkpoint_folder=MODEL_WEIGHTS_FOLDER,)
            best_model, data= train_model(train_dataset,val_dataset,entity+"_test",args.project,yaml_path,nn_model = nn_model,wandb_disable=args.wandb_disable,max_length=max_length,embed_size=embed_size,checkpoints_folder=MODEL_WEIGHTS_FOLDER)

        # `data` is VALIDATION: train_and_validate checkpoints on the best val macro-F1
        # over ~35 epochs and then reports that same checkpoint's val metrics, so it is a
        # maximum over epochs on the set it is scored against. Split 2 has never been
        # touched -- evaluate it here, and write the per-example rows the stratified
        # analysis needs. NOTE this test number comes from ONE seed (train_model keeps
        # only the val-best checkpoint), so it has no error bar, unlike the 5-seed val
        # columns next to it.
        test_acc = test_f1 = ""
        test_status = "skipped" if args.skip_test else "ok"
        if not args.skip_test:
            try:
                if len(test_dataset) == 0:
                    raise ValueError("test split is empty -- check cluster_split's split_report")
                os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
                test_report = test_model(
                    test_dataset, entity+"_final_test", args.project, yaml_path, best_model,
                    nn_model=nn_model, wandb_disable=args.wandb_disable,
                    max_length=max_length, embed_size=embed_size,
                    pred_out=os.path.join(PREDICTIONS_FOLDER, f"{entity}.csv"),
                    cm_path=os.path.join(PREDICTIONS_FOLDER, f"{entity}_confusion_matrix.png"))
                test_acc = test_report["accuracy"]
                test_f1 = test_report["macro avg"]["f1-score"]
            except Exception as exc:
                # Never let one arm's failure kill a multi-hour sweep, but record it as
                # "failed" rather than leaving a blank that reads as "not run yet" --
                # and --force_test will then retry it on the next pass.
                print(f"WARNING: test evaluation FAILED for {entity}: {exc}", file=sys.stderr)
                test_status = f"failed: {type(exc).__name__}"
        rows[entity] = [str(d) for d in data] + [str(test_acc), str(test_f1), test_status]
        flush()

    # Report on the test_acc CELL, not on test_status: "failed", "no_checkpoint" and
    # "skipped" all leave the result blank, and filtering on one prefix silently passed
    # over the other two.
    test_ix = COLUMNS[1:].index("test_acc")
    missing = [k for k, v in rows.items() if not str(v[test_ix]).strip()]
    if missing:
        print(f"\n{len(missing)} of {len(rows)} arm(s) have no test result:", file=sys.stderr)
        for k in missing:
            print(f"  {k}: {rows[k][-1] or 'not run'}", file=sys.stderr)
        print("re-run with --force_test to retry just those.", file=sys.stderr)
    print(f"wrote {output}")
