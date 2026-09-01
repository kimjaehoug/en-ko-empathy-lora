"""Classification metrics for TAFFC eval (confusion, per-class F1, error buckets)."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def _safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def binary_multilabel_metrics(
    y_true: list[list[int]], y_pred: list[list[int]]
) -> dict[str, float | None]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        for ti, pi in zip(t, p):
            if pi == 1 and ti == 1:
                tp += 1
            elif pi == 1 and ti == 0:
                fp += 1
            elif pi == 0 and ti == 1:
                fn += 1
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = (
        2 * prec * rec / (prec + rec)
        if prec is not None and rec is not None and (prec + rec) > 0
        else None
    )
    return {"micro_precision": prec, "micro_recall": rec, "micro_f1": f1, "tp": tp, "fp": fp, "fn": fn}


def confusion_matrix_from_ids(
    y_true: list[int], y_pred: list[int], n_classes: int
) -> list[list[int]]:
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    return cm


def per_class_report(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
) -> dict[str, Any]:
    n = len(labels)
    cm = confusion_matrix_from_ids(y_true, y_pred, n)
    rows = []
    supports = [sum(cm[i][j] for j in range(n)) for i in range(n)]
    macro_f1s = []
    for i, name in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = (
            2 * prec * rec / (prec + rec)
            if prec is not None and rec is not None and (prec + rec) > 0
            else 0.0
        )
        macro_f1s.append(f1 if supports[i] else 0.0)
        rows.append(
            {
                "label": name,
                "precision": prec,
                "recall": rec,
                "f1": f1 if supports[i] else None,
                "support": supports[i],
            }
        )
    acc = _safe_div(sum(int(t == p) for t, p in zip(y_true, y_pred)), len(y_true))
    weighted = _safe_div(
        sum((rows[i]["f1"] or 0) * supports[i] for i in range(n)),
        sum(supports),
    )
    return {
        "accuracy": acc,
        "macro_f1": _safe_div(sum(macro_f1s), sum(1 for s in supports if s)),
        "weighted_f1": weighted,
        "per_class": rows,
        "confusion_matrix": cm,
    }


def error_buckets(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    *,
    top_k_confusions: int = 10,
) -> dict[str, Any]:
    """Top confusions and per-class win/loss vs a reference (optional)."""
    n = len(labels)
    cm = confusion_matrix_from_ids(y_true, y_pred, n)
    pairs = []
    for t in range(n):
        for p in range(n):
            if t != p and cm[t][p]:
                pairs.append(
                    {
                        "true": labels[t],
                        "pred": labels[p],
                        "count": cm[t][p],
                    }
                )
    pairs.sort(key=lambda x: -x["count"])
    per_class_acc = {}
    for i, name in enumerate(labels):
        sup = sum(cm[i])
        per_class_acc[name] = {
            "acc": _safe_div(cm[i][i], sup),
            "support": sup,
            "errors": sup - cm[i][i],
        }
    return {
        "top_confusions": pairs[:top_k_confusions],
        "per_class_acc": per_class_acc,
    }


def compare_pred_vectors(a: list[int], b: list[int]) -> dict[str, Any]:
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0, "match_rate": None, "identical": True}
    match = sum(x == y for x, y in zip(a[:n], b[:n]))
    return {
        "n": n,
        "match_rate": match / n,
        "identical": match == n,
        "mismatch_indices_sample": [i for i in range(n) if a[i] != b[i]][:20],
    }


def aggregate_seeds(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(arr),
    }


def paired_ttest(a: list[float], b: list[float]) -> dict[str, float | None]:
    """Paired two-sided t-test on per-example or per-run deltas (requires scipy)."""
    if len(a) != len(b) or len(a) < 2:
        return {"t": None, "p": None, "mean_delta": None}
    try:
        from scipy import stats

        t, p = stats.ttest_rel(a, b)
        return {"t": float(t), "p": float(p), "mean_delta": float(np.mean(np.array(a) - np.array(b)))}
    except ImportError:
        d = np.array(a) - np.array(b)
        return {"t": None, "p": None, "mean_delta": float(d.mean()), "note": "scipy not installed"}


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float | None]:
    if not values:
        return {"low": None, "high": None, "mean": None}
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"low": lo, "high": hi, "mean": float(arr.mean())}
