#!/usr/bin/env python3
"""Evaluate multi-label predictions (MeSH, MOA, Target).

Parses **PREDICTION: [Label1; Label2; ...]** from model output and
computes exact-match accuracy, Jaccard similarity, micro/macro F1,
and per-class precision/recall/F1.

Usage:
    python scripts/eval_multilabel.py \
        --predictions outputs/.../generated_predictions.jsonl \
        --test_data data/prepared/mesh_test.jsonl \
        --output outputs/.../mesh_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set

import numpy as np


def extract_classes(text: str) -> Set[str]:
    """Extract class labels from **PREDICTION: [Label1; Label2; ...]**.

    Splits on semicolons (the canonical delimiter in our training data).
    Falls back to treating the whole string as a single label if no
    semicolons are present, since some class names contain commas
    (e.g. "Muscle Relaxants, Central").
    """
    text = text.strip()
    m = re.search(r"PREDICTION:\s*\[([^\]]*)\]", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
    else:
        m2 = re.search(r"PREDICTION:\s*(.+)", text, re.IGNORECASE)
        raw = m2.group(1).strip().strip("*").strip() if m2 else ""

    if not raw:
        return set()

    if ";" in raw:
        parts = [p.strip() for p in raw.split(";")]
    else:
        parts = [raw.strip()]

    return {p for p in parts if p}


def sets_to_binary(
    true_sets: List[Set[str]], pred_sets: List[Set[str]],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert list of label sets to binary indicator matrices."""
    all_labels = sorted(
        {lbl for s in true_sets for lbl in s}
        | {lbl for s in pred_sets for lbl in s}
    )
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    n = len(true_sets)
    k = len(all_labels)

    y_true = np.zeros((n, k), dtype=int)
    y_pred = np.zeros((n, k), dtype=int)

    for i, (ts, ps) in enumerate(zip(true_sets, pred_sets)):
        for lbl in ts:
            y_true[i, label_to_idx[lbl]] = 1
        for lbl in ps:
            if lbl in label_to_idx:
                y_pred[i, label_to_idx[lbl]] = 1

    return y_true, y_pred, all_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pred_texts: List[str] = []
    with open(args.predictions) as f:
        for line in f:
            d = json.loads(line)
            pred_texts.append(
                d.get("predict") or d.get("output") or d.get("generated") or ""
            )

    true_texts: List[str] = []
    with open(args.test_data) as f:
        for line in f:
            d = json.loads(line)
            true_texts.append(d["output"])

    assert len(pred_texts) == len(true_texts), (
        f"Mismatch: {len(pred_texts)} preds vs {len(true_texts)} labels"
    )

    pred_sets = [extract_classes(t) for t in pred_texts]
    true_sets = [extract_classes(t) for t in true_texts]

    n = len(true_sets)
    empty_preds = sum(1 for s in pred_sets if not s)

    exact_matches = sum(1 for t, p in zip(true_sets, pred_sets) if t == p)
    jaccards = []
    for t, p in zip(true_sets, pred_sets):
        if not t and not p:
            jaccards.append(1.0)
        elif not t or not p:
            jaccards.append(0.0)
        else:
            jaccards.append(len(t & p) / len(t | p))

    exact_match_acc = exact_matches / n if n > 0 else 0
    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0

    y_true, y_pred, all_labels = sets_to_binary(true_sets, pred_sets)

    tp = (y_true & y_pred).sum(axis=0).astype(float)
    fp = ((1 - y_true) & y_pred).sum(axis=0).astype(float)
    fn = (y_true & (1 - y_pred)).sum(axis=0).astype(float)

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0
        else 0
    )

    per_class_p = np.where(tp + fp > 0, tp / (tp + fp), 0)
    per_class_r = np.where(tp + fn > 0, tp / (tp + fn), 0)
    per_class_f1 = np.where(
        per_class_p + per_class_r > 0,
        2 * per_class_p * per_class_r / (per_class_p + per_class_r),
        0,
    )

    support = y_true.sum(axis=0).astype(int)
    active_mask = support > 0
    macro_p = per_class_p[active_mask].mean() if active_mask.any() else 0
    macro_r = per_class_r[active_mask].mean() if active_mask.any() else 0
    macro_f1 = per_class_f1[active_mask].mean() if active_mask.any() else 0

    print(f"\n{'=' * 60}")
    print("Multi-Label Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Total examples:       {n}")
    if empty_preds:
        print(f"Empty predictions:    {empty_preds}")
    print()
    print(f"Exact Match Accuracy: {exact_match_acc:.4f} ({exact_match_acc*100:.1f}%)")
    print(f"Mean Jaccard:         {mean_jaccard:.4f}")
    print(f"Micro Precision:      {micro_p:.4f}")
    print(f"Micro Recall:         {micro_r:.4f}")
    print(f"Micro F1:             {micro_f1:.4f}")
    print(f"Macro Precision:      {macro_p:.4f}")
    print(f"Macro Recall:         {macro_r:.4f}")
    print(f"Macro F1:             {macro_f1:.4f}")
    print()

    print("Per-Class Report (classes with support > 0):")
    print(f"{'Class':<50} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
    print("-" * 73)
    for i, lbl in enumerate(all_labels):
        if support[i] == 0:
            continue
        print(
            f"{lbl:<50} {per_class_p[i]:>6.3f} {per_class_r[i]:>6.3f} "
            f"{per_class_f1[i]:>6.3f} {support[i]:>5}"
        )

    if args.output:
        results = {
            "n_total": n,
            "n_empty_preds": int(empty_preds),
            "exact_match_accuracy": float(exact_match_acc),
            "mean_jaccard": float(mean_jaccard),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "n_classes": len(all_labels),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
