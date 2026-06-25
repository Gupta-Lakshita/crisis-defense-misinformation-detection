"""
evaluate.py
-----------
Evaluation utilities: per-class, per-source, per-crisis-type metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Return accuracy, macro precision/recall/F1, per-class F1, and ROC-AUC."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    p0, r0, f0, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], average="macro")
    p1, r1, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average="macro")

    result = dict(
        accuracy=round(acc * 100, 2),
        macro_precision=round(p * 100, 2),
        macro_recall=round(r * 100, 2),
        macro_f1=round(f * 100, 2),
        fake_precision=round(p0 * 100, 2),
        fake_recall=round(r0 * 100, 2),
        fake_f1=round(f0 * 100, 2),
        real_precision=round(p1 * 100, 2),
        real_recall=round(r1 * 100, 2),
        real_f1=round(f1 * 100, 2),
    )

    if y_prob is not None:
        try:
            result["roc_auc"] = round(roc_auc_score(y_true, y_prob) * 100, 2)
        except Exception:
            result["roc_auc"] = None

    return result


def per_source_metrics(df_test: pd.DataFrame, y_true, y_pred) -> dict:
    """
    Compute Macro-F1 for each data source separately.

    Parameters
    ----------
    df_test : DataFrame with a 'source' column aligned with y_true / y_pred
    y_true  : array-like ground truth labels
    y_pred  : array-like predicted labels
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    results = {}
    for src in df_test["source"].unique():
        mask = (df_test["source"] == src).values
        if mask.sum() < 5:
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_true[mask], y_pred[mask], average="macro"
        )
        results[src] = dict(
            n=int(mask.sum()),
            precision=round(p * 100, 2),
            recall=round(r * 100, 2),
            f1=round(f * 100, 2),
        )
    return results


def per_crisis_type_metrics(df_test: pd.DataFrame, y_true, y_pred) -> dict:
    """Compute Macro-F1 grouped by crisis_type column."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    results = {}
    for ct in df_test["crisis_type"].unique():
        mask = (df_test["crisis_type"] == ct).values
        if mask.sum() < 5:
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_true[mask], y_pred[mask], average="macro"
        )
        results[ct] = dict(
            n=int(mask.sum()),
            precision=round(p * 100, 2),
            recall=round(r * 100, 2),
            f1=round(f * 100, 2),
        )
    return results


def print_results_table(results: dict):
    """Pretty-print a results dictionary."""
    header = f"{'Model':<40} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7} {'AUC':>7}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        auc_str = f"{r.get('roc_auc','N/A'):>7}" if r.get("roc_auc") else f"{'N/A':>7}"
        print(
            f"{name:<40} {r['accuracy']:>6.2f}% "
            f"{r['macro_precision']:>6.2f}% "
            f"{r['macro_recall']:>6.2f}% "
            f"{r['macro_f1']:>6.2f}% "
            f"{auc_str}"
        )
