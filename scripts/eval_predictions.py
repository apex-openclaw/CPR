#!/usr/bin/env python3
"""Evaluate CPR predictions against ground truth.

Auto-detects task type from the test data filename:
  - bio_test.jsonl  -> binary bioactivity evaluation
  - mesh_test.jsonl, moa_test.jsonl, target_test.jsonl -> multi-label evaluation

Can also be forced via --task_type {bio, mesh, moa, target}.

Usage:
    python scripts/eval_predictions.py \
        --predictions outputs/.../generated_predictions.jsonl \
        --test_data data/prepared/bio_test.jsonl

    python scripts/eval_predictions.py \
        --predictions outputs/.../generated_predictions.jsonl \
        --test_data data/prepared/mesh_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

MULTILABEL_TASKS = {"mesh", "moa", "target"}


def detect_task_type(test_data: Path) -> str:
    """Infer task type from the test file name."""
    name = test_data.stem.lower()
    for task in MULTILABEL_TASKS:
        if task in name:
            return task
    if "bio" in name:
        return "bio"
    return "bio"


# ---------------------------------------------------------------------------
# Binary bioactivity evaluation
# ---------------------------------------------------------------------------

def extract_label(text: str) -> str | None:
    text = text.strip()
    exact_map = {
        "**PREDICTION: ACTIVE**": "ACTIVE",
        "**PREDICTION: INACTIVE**": "INACTIVE",
    }
    if text in exact_map:
        return exact_map[text]
    m = re.search(r"PREDICTION:\s*(ACTIVE|INACTIVE)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    upper = text.upper()
    if "INACTIVE" in upper:
        return "INACTIVE"
    if "ACTIVE" in upper:
        return "ACTIVE"
    return None


def load_texts(path: Path, key: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            texts.append(
                d.get("predict") or d.get(key) or d.get("generated") or ""
            )
    return texts


def eval_bioactivity(args: argparse.Namespace) -> None:
    raw_preds = load_texts(args.predictions, "predict")
    raw_labels = load_texts(args.test_data, "output")

    assert len(raw_preds) == len(raw_labels), (
        f"Mismatch: {len(raw_preds)} predictions vs {len(raw_labels)} labels"
    )

    pred_labels = [extract_label(p) for p in raw_preds]
    true_labels = [extract_label(l) for l in raw_labels]

    pred_failures = sum(1 for p in pred_labels if p is None)
    if pred_failures:
        print(
            f"WARNING: Could not extract label from "
            f"{pred_failures}/{len(pred_labels)} predictions"
        )
        shown = 0
        for i, p in enumerate(pred_labels):
            if p is None:
                print(f"  Example {i}: {repr(raw_preds[i][:200])}")
                shown += 1
                if shown >= 5:
                    break

    valid = [
        (t, p) for t, p in zip(true_labels, pred_labels)
        if t is not None and p is not None
    ]
    y_true = [1 if t == "ACTIVE" else 0 for t, _ in valid]
    y_pred = [1 if p == "ACTIVE" else 0 for _, p in valid]

    n = len(valid)
    print(f"\n{'=' * 60}")
    print("Bioactivity Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Total examples:     {len(raw_labels)}")
    print(f"Valid predictions:  {n} ({n / len(raw_labels) * 100:.1f}%)")
    if pred_failures:
        print(f"Parse failures:     {pred_failures}")
    print()

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = float("nan")

    print(f"Accuracy:           {acc:.4f} ({acc * 100:.1f}%)")
    print(f"Balanced Accuracy:  {bal_acc:.4f} ({bal_acc * 100:.1f}%)")
    print(f"AUROC:              {auroc:.4f}")
    print(f"F1 (ACTIVE):        {f1:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["INACTIVE", "ACTIVE"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"              Pred INACTIVE  Pred ACTIVE")
    print(f"  INACTIVE    {cm[0, 0]:>12}  {cm[0, 1]:>11}")
    print(f"  ACTIVE      {cm[1, 0]:>12}  {cm[1, 1]:>11}")
    print()

    n_active = sum(y_true)
    n_inactive = n - n_active
    print(
        f"Class distribution: {n_inactive} INACTIVE "
        f"({n_inactive / n * 100:.1f}%), "
        f"{n_active} ACTIVE ({n_active / n * 100:.1f}%)"
    )
    majority = max(n_active, n_inactive) / n
    print(f"Majority baseline:  {majority:.4f} ({majority * 100:.1f}%)")
    print(
        f"Lift over baseline: {acc - majority:+.4f} "
        f"({(acc - majority) * 100:+.1f}pp)"
    )

    if args.output:
        results = {
            "task": "bioactivity",
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "auroc": auroc,
            "f1_active": f1,
            "mcc": mcc,
            "n_total": len(raw_labels),
            "n_valid": n,
            "n_parse_failures": pred_failures,
            "confusion_matrix": cm.tolist(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


# ---------------------------------------------------------------------------
# Multi-label dispatch
# ---------------------------------------------------------------------------

def eval_multilabel(args: argparse.Namespace) -> None:
    """Delegate to eval_multilabel.py."""
    script = Path(__file__).with_name("eval_multilabel.py")
    cmd = [sys.executable, str(script),
           "--predictions", str(args.predictions),
           "--test_data", str(args.test_data)]
    if args.output:
        cmd += ["--output", str(args.output)]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate CPR predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON")
    parser.add_argument(
        "--task_type", type=str, default=None,
        choices=["bio", "mesh", "moa", "target"],
        help="Force task type (auto-detected from filename if omitted)",
    )
    args = parser.parse_args()

    task = args.task_type or detect_task_type(args.test_data)
    print(f"Detected task type: {task}")

    if task in MULTILABEL_TASKS:
        eval_multilabel(args)
    else:
        eval_bioactivity(args)


if __name__ == "__main__":
    main()
