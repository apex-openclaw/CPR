#!/usr/bin/env python3
"""Evaluate CPR predictions against ground truth.

Usage:
    python scripts/eval_predictions.py \
        --predictions outputs/qwen35-cpr-lora/preds/generated_predictions.jsonl \
        --test_data data/prepared/test.jsonl

    # Or evaluate a specific checkpoint:
    python scripts/eval_predictions.py \
        --predictions outputs/qwen35-cpr-lora/preds-ckpt800/generated_predictions.jsonl \
        --test_data data/prepared/test.jsonl
"""
import argparse
import json
import re
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


def extract_label(text: str) -> str | None:
    """Extract ACTIVE/INACTIVE from model output."""
    text = text.strip()
    # Try exact match first
    if text in ("**PREDICTION: ACTIVE**", "**PREDICTION: INACTIVE**"):
        return "ACTIVE" if "ACTIVE" in text else "INACTIVE"
    # Regex fallback for free-form generation
    m = re.search(r"PREDICTION:\s*(ACTIVE|INACTIVE)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Last resort: look for the word anywhere
    upper = text.upper()
    if "INACTIVE" in upper:
        return "INACTIVE"
    if "ACTIVE" in upper:
        return "ACTIVE"
    return None


def load_predictions(path: Path) -> list[str]:
    """Load LlamaFactory generated_predictions.jsonl."""
    preds = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # LlamaFactory uses 'predict' key for the generated text
            text = d.get("predict") or d.get("output") or d.get("generated") or ""
            preds.append(text)
    return preds


def load_ground_truth(path: Path) -> list[str]:
    """Load ground truth from test.jsonl."""
    labels = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            labels.append(d["output"])
    return labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate CPR predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON")
    args = parser.parse_args()

    raw_preds = load_predictions(args.predictions)
    raw_labels = load_ground_truth(args.test_data)

    assert len(raw_preds) == len(raw_labels), (
        f"Mismatch: {len(raw_preds)} predictions vs {len(raw_labels)} labels"
    )

    pred_labels = [extract_label(p) for p in raw_preds]
    true_labels = [extract_label(l) for l in raw_labels]

    # Check for extraction failures
    pred_failures = sum(1 for p in pred_labels if p is None)
    if pred_failures:
        print(f"WARNING: Could not extract label from {pred_failures}/{len(pred_labels)} predictions")
        # Show some failure examples
        for i, (p, raw) in enumerate(zip(pred_labels, raw_preds)):
            if p is None:
                print(f"  Example {i}: {repr(raw_preds[i][:200])}")
                if sum(1 for x in pred_labels[:i+1] if x is None) >= 5:
                    break

    # Filter to examples where both labels extracted
    valid = [(t, p) for t, p in zip(true_labels, pred_labels) if t is not None and p is not None]
    y_true = [1 if t == "ACTIVE" else 0 for t, _ in valid]
    y_pred = [1 if p == "ACTIVE" else 0 for _, p in valid]

    n = len(valid)
    print(f"\n{'='*60}")
    print(f"CPR Evaluation Results")
    print(f"{'='*60}")
    print(f"Total examples:     {len(raw_labels)}")
    print(f"Valid predictions:  {n} ({n/len(raw_labels)*100:.1f}%)")
    if pred_failures:
        print(f"Parse failures:     {pred_failures}")
    print()

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"Accuracy:           {acc:.4f} ({acc*100:.1f}%)")
    print(f"Balanced Accuracy:  {bal_acc:.4f} ({bal_acc*100:.1f}%)")
    print(f"F1 (ACTIVE):        {f1:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["INACTIVE", "ACTIVE"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"              Pred INACTIVE  Pred ACTIVE")
    print(f"  INACTIVE    {cm[0,0]:>12}  {cm[0,1]:>11}")
    print(f"  ACTIVE      {cm[1,0]:>12}  {cm[1,1]:>11}")
    print()

    # Per-class counts
    n_active = sum(y_true)
    n_inactive = len(y_true) - n_active
    print(f"Class distribution: {n_inactive} INACTIVE ({n_inactive/n*100:.1f}%), {n_active} ACTIVE ({n_active/n*100:.1f}%)")

    # Majority baseline
    majority = max(n_active, n_inactive) / n
    print(f"Majority baseline:  {majority:.4f} ({majority*100:.1f}%)")
    print(f"Lift over baseline: {acc - majority:+.4f} ({(acc-majority)*100:+.1f}pp)")

    if args.output:
        results = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
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


if __name__ == "__main__":
    main()
