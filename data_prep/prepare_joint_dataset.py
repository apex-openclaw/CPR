#!/usr/bin/env python3
"""Prepare joint multi-task dataset for CPR SFT training.

Combines cpg0012 bioactivity prediction with BBBC036 multi-label tasks
(MeSH pharmacological classes, Broad MOA, Broad gene targets) into a
unified Alpaca-style JSONL dataset with compound-level split consistency.

Usage:
    python data_prep/prepare_joint_dataset.py --config data_prep/joint_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BIOACTIVITY_PROMPT = """\
You are an expert in drug discovery, medicinal chemistry, and cell biology.

## Task
Predict whether the following compound is ACTIVE or INACTIVE in the given bioassay.

## Compound
SMILES: {smiles}

## Cell Painting Morphological Profile
This compound's treatment of cells produced these notable phenotypic changes (z-scores vs DMSO control):
{feature_summary}

## Bioassay
Name: {assay_name}
{endpoint_note}Description: {assay_description}

## Instructions
Reason step-by-step:
1. What is the molecular structure and likely pharmacological properties?
2. What do the morphological changes suggest about mechanism of action?
3. Is this consistent with activity in the described assay?

Then output your final prediction as exactly: **PREDICTION: ACTIVE** or **PREDICTION: INACTIVE**
"""

MULTILABEL_PROMPT = """\
You are an expert in drug discovery, medicinal chemistry, and cell biology.

## Task
Predict the {task_description} of the following compound based on \
its molecular structure and Cell Painting morphological profile.

## Compound
Name: {cpd_name}
SMILES: {smiles}

## Cell Painting Morphological Profile
This compound's treatment of cells produced these notable phenotypic changes (z-scores vs DMSO control):
{feature_summary}

## Candidate Classes
{candidate_classes}

## Instructions
Reason step-by-step:
1. What is the molecular structure and likely pharmacological properties?
2. What do the morphological changes suggest about mechanism of action?
3. Which candidate classes are consistent with the compound's profile?

Output your prediction as: **PREDICTION: [Label1; Label2; ...]**
Only include labels you are confident apply. Use exact names from the candidate list above.\
"""

LABEL_MAP = {0: "INACTIVE", 1: "ACTIVE"}

TASK_DEFS = {
    "mesh": {
        "task_description": "MeSH pharmacological classification",
        "task_name": "MeSH Pharmacological Classes",
    },
    "moa": {
        "task_description": "Broad mechanism of action (MOA)",
        "task_name": "Broad MOA",
    },
    "target": {
        "task_description": "gene target",
        "task_name": "Broad Gene Target",
    },
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

BRD_RE = re.compile(r"(BRD-[A-Z]\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).with_name("joint_config.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fmt_path(value: Optional[str], fmt: dict) -> Optional[Path]:
    if not value:
        return None
    try:
        formatted = value.format(**fmt)
    except KeyError:
        formatted = value
    return Path(formatted).expanduser().resolve()


def summarize_features(vector: np.ndarray, names: np.ndarray, top_k: int) -> str:
    abs_vals = np.abs(vector)
    top_idx = np.argsort(abs_vals)[-top_k:][::-1]
    lines: List[str] = []
    for idx in top_idx:
        val = vector[idx]
        if not np.isfinite(val):
            continue
        direction = "Elevated" if val >= 0 else "Reduced"
        score = f"{val:+.2e}" if abs(val) > 9999 else f"{val:+.2f}"
        lines.append(f"{direction} {names[idx]} (z={score})")
    return "\n".join(lines) if lines else "No reliable morphological signal."


def count_direction(summary: str) -> tuple[int, int]:
    lines = summary.splitlines()
    return (
        sum(ln.startswith("Elevated") for ln in lines),
        sum(ln.startswith("Reduced") for ln in lines),
    )


def write_jsonl(records: List[dict], path: Path) -> None:
    with open(path, "w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_brd_base(broad_id: str) -> Optional[str]:
    m = BRD_RE.match(str(broad_id))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Data loading: CellProfiler features + BRD metadata
# ---------------------------------------------------------------------------

@dataclass
class CompoundInfo:
    brd_base: str
    inchikey: str
    smiles: str
    cpd_name: str
    split: str


def load_plate_features_and_brd_map(
    features_dir: Path, plate_well_map: Path,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, dict]]:
    """Load CellProfiler features and extract BRD->compound metadata.

    Returns
    -------
    feature_matrix : (N, F) float32
    feature_names  : (F,) str array
    ik_to_idx      : INCHIKEY -> row index in feature_matrix
    brd_map        : BRD_BASE -> {inchikey, smiles, cpd_name}
    """
    mapping = pd.read_csv(plate_well_map)
    plates = mapping["plate_id"].unique()

    feature_chunks: List[pd.DataFrame] = []
    brd_rows: List[dict] = []

    for plate_id in tqdm(plates, desc="Loading plates"):
        csv_path = (
            features_dir / str(plate_id)
            / f"{plate_id}_normalized_feature_select_negcon_all.csv.gz"
        )
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        plate_wells = mapping[mapping["plate_id"] == plate_id][["well", "INCHIKEY"]]
        df = df.merge(
            plate_wells, left_on="Metadata_Well", right_on="well", how="inner",
        )

        if "Metadata_broad_sample" in df.columns:
            meta = df.dropna(subset=["Metadata_broad_sample"]).drop_duplicates(
                "Metadata_broad_sample"
            )
            for _, row in meta.iterrows():
                brd_rows.append({
                    "broad_id": row["Metadata_broad_sample"],
                    "inchikey": row["INCHIKEY"],
                    "smiles": row.get("Metadata_smiles", ""),
                    "cpd_name": row.get("Metadata_cpd_name", ""),
                })

        feat_cols = [
            c for c in df.columns
            if not c.startswith("Metadata_") and c not in ("well", "INCHIKEY")
        ]
        feature_chunks.append(df[["INCHIKEY"] + feat_cols])

    print(f"Loaded features from {len(feature_chunks)}/{len(plates)} plates")
    all_features = pd.concat(feature_chunks, ignore_index=True)
    feat_cols = [c for c in all_features.columns if c != "INCHIKEY"]

    grouped = all_features.groupby("INCHIKEY")[feat_cols].mean()
    feature_names = np.array(grouped.columns.tolist())
    feature_matrix = grouped.values.astype(np.float32)
    ik_to_idx = {ik: i for i, ik in enumerate(grouped.index)}
    print(
        f"Feature matrix: {feature_matrix.shape[0]} compounds "
        f"x {feature_matrix.shape[1]} features"
    )

    brd_map: Dict[str, dict] = {}
    for row in brd_rows:
        base = extract_brd_base(row["broad_id"])
        if base and base not in brd_map:
            brd_map[base] = {
                "inchikey": row["inchikey"],
                "smiles": row["smiles"] if pd.notna(row["smiles"]) else "",
                "cpd_name": row["cpd_name"] if pd.notna(row["cpd_name"]) else "",
            }
    print(f"BRD->compound mappings from CellProfiler: {len(brd_map)}")

    return feature_matrix, feature_names, ik_to_idx, brd_map


# ---------------------------------------------------------------------------
# Compound split registry
# ---------------------------------------------------------------------------

def build_compound_registry(
    split_csvs: Dict[str, Path],
    brd_map: Dict[str, dict],
    ik_to_idx: Dict[str, int],
    bbbc036_brd_ids: Set[str],
    ratios: List[float],
    seed: int,
) -> Dict[str, CompoundInfo]:
    """Build compound registry with consistent split assignment.

    1. cpg0012 compounds keep their datasplit1 assignment.
    2. BBBC036-only compounds get proportional random split (70/15/15).
    3. Compounds without CellProfiler features are skipped.
    """
    registry: Dict[str, CompoundInfo] = {}

    for split_name, csv_path in split_csvs.items():
        df = pd.read_csv(csv_path)
        df["BRD_BASE"] = df["BROAD_ID"].apply(extract_brd_base)
        for _, row in df.drop_duplicates("BRD_BASE").iterrows():
            brd = row["BRD_BASE"]
            if brd is None or brd in registry:
                continue
            ik = row["INCHIKEY"]
            if ik not in ik_to_idx:
                continue
            registry[brd] = CompoundInfo(
                brd_base=brd,
                inchikey=ik,
                smiles=str(row["SMILES"]) if pd.notna(row.get("SMILES")) else "",
                cpd_name=str(row["CPD_NAME"]) if pd.notna(row.get("CPD_NAME")) else "",
                split=split_name,
            )

    cpg_count = len(registry)
    print(f"cpg0012 compounds with features: {cpg_count}")

    new_brds = []
    for brd in sorted(bbbc036_brd_ids):
        if brd in registry:
            continue
        info = brd_map.get(brd)
        if info is None or info["inchikey"] not in ik_to_idx:
            continue
        new_brds.append((brd, info))

    rng = random.Random(seed)
    rng.shuffle(new_brds)
    n = len(new_brds)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    split_labels = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )

    for (brd, info), split_name in zip(new_brds, split_labels):
        registry[brd] = CompoundInfo(
            brd_base=brd,
            inchikey=info["inchikey"],
            smiles=info["smiles"],
            cpd_name=info["cpd_name"],
            split=split_name,
        )

    new_count = len(registry) - cpg_count
    print(f"BBBC036-only compounds with features: {new_count}")
    print(f"Total registered compounds: {len(registry)}")

    for s in ("train", "val", "test"):
        cnt = sum(1 for c in registry.values() if c.split == s)
        print(f"  {s}: {cnt}")

    return registry


# ---------------------------------------------------------------------------
# Bioactivity task
# ---------------------------------------------------------------------------

def load_assay_metadata(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        payload = json.load(f)
    meta = {}
    for entry in payload:
        aid = str(entry.get("aid"))
        meta[aid] = {
            "name": entry.get("name") or f"Assay {aid}",
            "description": entry.get("description") or entry.get("abstract") or "",
        }
    return meta


def select_assays(
    labels_df: pd.DataFrame,
    train_iks: Set[str],
    assays: List[str],
    cfg: dict,
    assay_meta: dict,
) -> List[str]:
    require_desc = cfg.get("require_description", False)
    min_total = cfg.get("min_labeled", 0)
    min_pos = cfg.get("min_pos_ratio", 0.0)
    max_pos = cfg.get("max_pos_ratio", 1.0)
    train_df = labels_df[labels_df["INCHIKEY"].isin(train_iks)]

    eligible: List[str] = []
    skipped_no_desc = 0
    for aid in assays:
        if require_desc and assay_meta:
            m = assay_meta.get(aid, {})
            if not m.get("description") and not m.get("abstract"):
                skipped_no_desc += 1
                continue
        series = train_df[aid]
        labeled = series[series >= 0]
        if len(labeled) < min_total:
            continue
        pos_ratio = (labeled == 1).mean() if len(labeled) > 0 else 0
        if pos_ratio < min_pos or pos_ratio > max_pos:
            continue
        eligible.append(aid)
    if skipped_no_desc:
        print(f"  Skipped {skipped_no_desc} assays without descriptions")
    return eligible


def build_endpoint_note(aid: str) -> str:
    if "_" not in aid:
        return ""
    _, suffix = aid.rsplit("_", 1)
    if not suffix.isdigit():
        return ""
    return (
        f"Activity threshold: endpoint {int(suffix) + 1} "
        "(stricter activity cutoff than the primary endpoint)\n"
    )


def load_curated_traces(traces_dir: Optional[Path]) -> dict:
    if traces_dir is None or not traces_dir.exists():
        return {}
    trace_map: dict = {}
    for path in traces_dir.glob("*_train_traces.jsonl"):
        aid = path.stem.split("_")[0]
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                trace_map[(entry.get("inchikey"), aid)] = {
                    "prompt": entry.get("prompt"),
                    "response": entry.get("response"),
                }
    return trace_map


def generate_bioactivity_samples(
    registry: Dict[str, CompoundInfo],
    labels_csv: Path,
    assay_desc_json: Optional[Path],
    traces_dir: Optional[Path],
    feature_matrix: np.ndarray,
    feature_names: np.ndarray,
    ik_to_idx: Dict[str, int],
    prompt_cfg: dict,
    reasoning_cfg: dict,
    rng: random.Random,
) -> Dict[str, tuple[List[dict], List[dict]]]:
    """Generate bioactivity (ACTIVE/INACTIVE) samples per split."""
    ik_to_split: Dict[str, str] = {}
    ik_to_smiles: Dict[str, str] = {}
    for c in registry.values():
        ik_to_split[c.inchikey] = c.split
        if c.smiles:
            ik_to_smiles[c.inchikey] = c.smiles

    labels_df = pd.read_csv(labels_csv)
    assay_cols = [c for c in labels_df.columns if c not in ("INCHIKEY", "SMILES")]
    for ik, smi in zip(labels_df["INCHIKEY"], labels_df["SMILES"]):
        if pd.notna(smi):
            ik_to_smiles[ik] = str(smi)

    train_iks = {c.inchikey for c in registry.values() if c.split == "train"}
    assay_meta = load_assay_metadata(assay_desc_json)
    target_assays = select_assays(
        labels_df, train_iks, assay_cols, prompt_cfg, assay_meta,
    )
    print(f"  Selected {len(target_assays)} assays after filtering")

    curated_traces = (
        load_curated_traces(traces_dir)
        if reasoning_cfg.get("use_curated_traces", False)
        else {}
    )
    fallback_style = reasoning_cfg.get("fallback_style", "template")
    top_k = prompt_cfg.get("top_k_features", 15)

    results: Dict[str, tuple[List[dict], List[dict]]] = {
        s: ([], []) for s in ("train", "val", "test")
    }

    for aid in tqdm(target_assays, desc="Bioactivity assays"):
        col = labels_df[["INCHIKEY", aid]]
        labeled = col[col[aid] >= 0]
        assay_name = assay_meta.get(aid, {}).get("name", f"Assay {aid}")
        assay_desc = assay_meta.get(aid, {}).get("description", "")

        for _, row in labeled.iterrows():
            ik = row["INCHIKEY"]
            split = ik_to_split.get(ik)
            feat_idx = ik_to_idx.get(ik)
            smiles = ik_to_smiles.get(ik)
            if split is None or feat_idx is None or not smiles:
                continue

            label = int(row[aid])
            feature_vec = feature_matrix[feat_idx]
            summary = summarize_features(feature_vec, feature_names, top_k)

            trace_entry = None
            if split == "train" and curated_traces:
                trace_entry = curated_traces.get((ik, aid))

            if trace_entry:
                instruction = trace_entry["prompt"].strip()
                output = trace_entry["response"].strip()
            else:
                endpoint_note = build_endpoint_note(aid)
                instruction = BIOACTIVITY_PROMPT.format(
                    smiles=smiles,
                    feature_summary=summary,
                    assay_name=assay_name,
                    assay_description=assay_desc,
                    endpoint_note=endpoint_note,
                )
                if fallback_style == "label_only":
                    output = f"**PREDICTION: {LABEL_MAP[label]}**"
                else:
                    elevated, reduced = count_direction(summary)
                    label_text = LABEL_MAP[label]
                    output = (
                        "Step-by-step reasoning:\n"
                        f"1. The compound's structure (see SMILES) suggests a "
                        f"specific scaffold under evaluation for {assay_name}.\n"
                        f"2. Cell Painting reveals {elevated} elevated and "
                        f"{reduced} reduced features, for example:\n{summary}\n"
                        f"3. These phenotypic shifts are "
                        f"{'consistent' if label else 'not consistent'} "
                        f"with the expected response in {assay_name}.\n"
                        f"Therefore I conclude **PREDICTION: {label_text}**"
                    )

            results[split][0].append({
                "instruction": instruction,
                "input": "",
                "output": output,
            })
            results[split][1].append({
                "split": split,
                "task": "bioactivity",
                "assay_id": aid,
                "compound_id": ik,
                "inchikey": ik,
                "smiles": smiles,
                "cpd_name": "",
                "label_summary": f"{aid}={LABEL_MAP[label]}",
            })

    return results


# ---------------------------------------------------------------------------
# Multi-label tasks (MeSH, MOA, Target)
# ---------------------------------------------------------------------------

def generate_multilabel_samples(
    task_key: str,
    labels_csv: Path,
    registry: Dict[str, CompoundInfo],
    feature_matrix: np.ndarray,
    feature_names: np.ndarray,
    ik_to_idx: Dict[str, int],
    prompt_cfg: dict,
) -> Dict[str, tuple[List[dict], List[dict]]]:
    """Generate multi-label samples for one task."""
    task_def = TASK_DEFS[task_key]
    top_k = prompt_cfg.get("top_k_features", 300)
    min_support = prompt_cfg.get("min_class_support", 5)

    label_df = pd.read_csv(labels_csv, index_col=0, delimiter=";")

    class_counts = label_df.sum(axis=0)
    valid_classes = sorted(class_counts[class_counts >= min_support].index.tolist())
    label_df = label_df[valid_classes]
    label_df = label_df[label_df.sum(axis=1) > 0]
    print(
        f"  {task_key}: {len(valid_classes)} classes, "
        f"{len(label_df)} compounds after filtering"
    )

    candidate_list = "\n".join(f"- {c}" for c in valid_classes)

    results: Dict[str, tuple[List[dict], List[dict]]] = {
        s: ([], []) for s in ("train", "val", "test")
    }

    for brd_id in label_df.index:
        compound = registry.get(brd_id)
        if compound is None:
            continue
        feat_idx = ik_to_idx.get(compound.inchikey)
        if feat_idx is None:
            continue

        active_labels = sorted(
            c for c in valid_classes if label_df.loc[brd_id, c] > 0
        )
        if not active_labels:
            continue

        feature_vec = feature_matrix[feat_idx]
        summary = summarize_features(feature_vec, feature_names, top_k)

        instruction = MULTILABEL_PROMPT.format(
            task_description=task_def["task_description"],
            cpd_name=compound.cpd_name or compound.brd_base,
            smiles=compound.smiles,
            feature_summary=summary,
            candidate_classes=candidate_list,
        )
        output = f"**PREDICTION: [{'; '.join(active_labels)}]**"

        split = compound.split
        results[split][0].append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })
        results[split][1].append({
            "split": split,
            "task": task_key,
            "compound_id": brd_id,
            "inchikey": compound.inchikey,
            "smiles": compound.smiles,
            "cpd_name": compound.cpd_name,
            "label_summary": "; ".join(active_labels),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(args.config)

    fmt = {
        "data_root": cfg.get("data_root", ""),
        "project_root": str(project_root),
    }
    p = cfg["paths"]

    features_dir = fmt_path(p["features_dir"], fmt)
    plate_well_map = fmt_path(p["plate_well_map"], fmt)
    labels_csv = fmt_path(p["labels_csv"], fmt)
    assay_desc_json = fmt_path(p.get("assay_desc_json"), fmt)
    traces_dir = fmt_path(p.get("traces_dir"), fmt)

    split_csvs = {
        "train": fmt_path(p["train_split_csv"], fmt),
        "val": fmt_path(p["val_split_csv"], fmt),
        "test": fmt_path(p["test_split_csv"], fmt),
    }

    tasks_cfg = cfg.get("tasks", {})
    prompt_cfg = cfg.get("prompt", {})
    reasoning_cfg = cfg.get("reasoning", {})
    split_cfg = cfg.get("split", {})

    output_dir = project_root / cfg.get("output", {}).get("dir", "data/prepared")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    seed = split_cfg.get("seed", args.seed)

    # ---- Step 1: Load CellProfiler features + BRD mapping ----
    print("=" * 60)
    print("Step 1: Loading CellProfiler features")
    print("=" * 60)
    feature_matrix, feature_names, ik_to_idx, brd_map = (
        load_plate_features_and_brd_map(features_dir, plate_well_map)
    )

    # ---- Step 2: Build compound registry ----
    print()
    print("=" * 60)
    print("Step 2: Building compound split registry")
    print("=" * 60)

    bbbc036_brds: Set[str] = set()
    for task_key in ("mesh", "moa", "target"):
        task_cfg_entry = tasks_cfg.get(task_key, {})
        if not task_cfg_entry.get("enabled", False):
            continue
        task_csv = fmt_path(task_cfg_entry["labels_csv"], fmt)
        df = pd.read_csv(task_csv, index_col=0, delimiter=";")
        bbbc036_brds.update(df.index.tolist())
    print(f"BBBC036 unique BRD IDs across enabled tasks: {len(bbbc036_brds)}")

    ratios = split_cfg.get("new_compound_ratios", [0.7, 0.15, 0.15])
    registry = build_compound_registry(
        split_csvs, brd_map, ik_to_idx, bbbc036_brds, ratios, seed,
    )

    reg_rows = sorted(
        [
            {
                "brd_base": c.brd_base,
                "inchikey": c.inchikey,
                "smiles": c.smiles,
                "cpd_name": c.cpd_name,
                "split": c.split,
            }
            for c in registry.values()
        ],
        key=lambda x: x["brd_base"],
    )
    pd.DataFrame(reg_rows).to_csv(output_dir / "compound_registry.csv", index=False)
    print(f"Saved compound registry: {output_dir / 'compound_registry.csv'}")

    # ---- Step 3: Generate samples per task ----
    all_train: List[dict] = []
    all_val: List[dict] = []
    all_metadata: List[dict] = []
    test_sets: Dict[str, List[dict]] = {}

    if tasks_cfg.get("bioactivity", {}).get("enabled", False):
        print()
        print("=" * 60)
        print("Generating bioactivity samples")
        print("=" * 60)
        bio_results = generate_bioactivity_samples(
            registry, labels_csv, assay_desc_json, traces_dir,
            feature_matrix, feature_names, ik_to_idx,
            prompt_cfg, reasoning_cfg, rng,
        )
        for split_name in ("train", "val", "test"):
            records, meta = bio_results[split_name]
            if split_name == "train":
                all_train.extend(records)
            elif split_name == "val":
                all_val.extend(records)
            else:
                test_sets["bio"] = records
            all_metadata.extend(meta)
            print(f"  bioactivity {split_name}: {len(records)} samples")

    for task_key in ("mesh", "moa", "target"):
        task_cfg_entry = tasks_cfg.get(task_key, {})
        if not task_cfg_entry.get("enabled", False):
            continue
        print()
        print("=" * 60)
        print(f"Generating {task_key} samples")
        print("=" * 60)
        task_csv = fmt_path(task_cfg_entry["labels_csv"], fmt)
        ml_results = generate_multilabel_samples(
            task_key, task_csv, registry,
            feature_matrix, feature_names, ik_to_idx,
            prompt_cfg,
        )
        for split_name in ("train", "val", "test"):
            records, meta = ml_results[split_name]
            if split_name == "train":
                all_train.extend(records)
            elif split_name == "val":
                all_val.extend(records)
            else:
                test_sets[task_key] = records
            all_metadata.extend(meta)
            print(f"  {task_key} {split_name}: {len(records)} samples")

    # ---- Step 4: Write output ----
    print()
    print("=" * 60)
    print("Writing output files")
    print("=" * 60)

    rng.shuffle(all_train)
    rng.shuffle(all_val)

    write_jsonl(all_train, output_dir / "train.jsonl")
    print(f"  train.jsonl: {len(all_train)} samples")

    write_jsonl(all_val, output_dir / "val.jsonl")
    print(f"  val.jsonl: {len(all_val)} samples")

    for task_key, records in test_sets.items():
        fname = f"{task_key}_test.jsonl"
        write_jsonl(records, output_dir / fname)
        print(f"  {fname}: {len(records)} samples")

    pd.DataFrame(all_metadata).to_csv(output_dir / "metadata.csv", index=False)
    print(f"  metadata.csv: {len(all_metadata)} rows")

    # ---- Summary ----
    print()
    print("=" * 60)
    total = (
        len(all_train) + len(all_val)
        + sum(len(r) for r in test_sets.values())
    )
    print(f"Done! Total samples: {total}")
    print(f"  Train: {len(all_train)}")
    print(f"  Val:   {len(all_val)}")
    for tk, recs in test_sets.items():
        print(f"  Test ({tk}): {len(recs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
