#!/usr/bin/env python3
"""
report_metrics.py -- imbalance-aware classification report.

Accuracy hides class imbalance: on a 1:23 dataset a model can look ~96 % while
ignoring the rare families entirely. For the per-exon vs per-protein comparison the
honest headline is macro-F1 (every family weighted equally) plus the per-class
precision/recall, so a gain on the rare classes is visible.

Call report() with integer (or string) class labels -- not one-hot. With your
OneHotEncoder, get the class index back with argmax before calling:

    import numpy as np
    y_true = np.argmax(onehot_true, axis=1)
    y_pred = np.argmax(logits.cpu().numpy(), axis=1)
    report(y_true, y_pred, class_names=label_encoder.categories_[0])
"""
import numpy as np


def report(y_true, y_pred, class_names=None):
    """Print per-class precision/recall/F1 + macro/micro/weighted F1; return the dict."""
    from sklearn.metrics import (classification_report, f1_score,
                                 balanced_accuracy_score, confusion_matrix)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    names = [str(c) for c in class_names] if class_names is not None else None

    print(classification_report(y_true, y_pred, target_names=names, digits=3, zero_division=0))
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    print(f"macro-F1 {out['macro_f1']:.3f} | micro-F1 {out['micro_f1']:.3f} | "
          f"weighted-F1 {out['weighted_f1']:.3f} | balanced-acc {out['balanced_accuracy']:.3f}")
    print("\nconfusion matrix (rows = true, cols = pred):")
    print(confusion_matrix(y_true, y_pred))
    return out
