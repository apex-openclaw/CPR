#!/usr/bin/env python3
"""Prepare CPR reasoning datasets for LlamaFactory.

Reads the cpg0012 labels/CellProfiler features/assay descriptions, optionally
ingests curated reasoning traces, and writes Alpaca-style JSONL splits plus
metadata for downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROMPT_TEMPLATE = """You are an expert in drug discovery, medicinal chemistry, and cell biology.

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
Analyze the compound, Cell Painting profile, and bioassay internally. Do not include reasoning in your response.

Include your final answer exactly as: the final answer is: [active/inactive]
"""

LABEL_MAP = {0: "INACTIVE", 1: "ACTIVE"}
FINAL_ANSWER_MAP = {0: "inactive", 1: "active"}


def format_final_answer(label: int) -> str:
    return f"the final answer is: {FINAL_ANSWER_MAP[label]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("dataset_config.yaml"),
        help="Path to dataset_config.yaml",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible sampling"
    )
    return parser.parse_args()


@dataclass
class Paths:
    labels_csv: Path
    features_dir: Path
    plate_well_map: Path
    train_split_csv: Path
    val_split_csv: Path
    test_split_csv: Path
    assay_desc_json: Optional[Path]
    traces_dir: Optional[Path]


@dataclass
class SplitData:
    name: str
    inchikeys: set


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, project_root: Path) -> Paths:
    fmt = {"data_root": cfg.get("data_root", ""), "project_root": str(project_root)}

    def _fmt(value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        try:
            formatted = value.format(**fmt)
        except KeyError:
            formatted = value
        return Path(formatted).expanduser().resolve()

    p = cfg["paths"]
    return Paths(
        labels_csv=_fmt(p["labels_csv"]),
        features_dir=_fmt(p["features_dir"]),
        plate_well_map=_fmt(p["plate_well_map"]),
        train_split_csv=_fmt(p["train_split_csv"]),
        val_split_csv=_fmt(p["val_split_csv"]),
        test_split_csv=_fmt(p["test_split_csv"]),
        assay_desc_json=_fmt(p.get("assay_desc_json")),
        traces_dir=_fmt(p.get("traces_dir")),
    )


def load_splits(paths: Paths) -> Dict[str, SplitData]:
    splits = {}
    for name, csv_path in (
        ("train", paths.train_split_csv),
        ("val", paths.val_split_csv),
        ("test", paths.test_split_csv),
    ):
        df = pd.read_csv(csv_path)
        splits[name] = SplitData(name=name, inchikeys=set(df["INCHIKEY"].unique()))
    return splits


def load_assay_metadata(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r") as f:
        payload = json.load(f)
    meta = {}
    for entry in payload:
        aid = str(entry.get("aid"))
        meta[aid] = {
            "name": entry.get("name") or f"Assay {aid}",
            "description": entry.get("description")
            or entry.get("abstract")
            or "",
        }
    return meta


def load_plate_features(
    features_dir: Path, plate_well_map: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load per-plate CSVs and aggregate features per compound (mean across wells).

    Returns (feature_matrix, feature_names, inchikey_to_row_idx).
    """
    mapping = pd.read_csv(plate_well_map)
    plates = mapping["plate_id"].unique()

    chunks: List[pd.DataFrame] = []
    for plate_id in tqdm(plates, desc="Loading plates"):
        csv_path = features_dir / str(plate_id) / f"{plate_id}_normalized_feature_select_negcon_all.csv.gz"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        # Keep only treatment wells that appear in the mapping
        plate_wells = mapping[mapping["plate_id"] == plate_id][["well", "INCHIKEY"]]
        df = df.merge(plate_wells, left_on="Metadata_Well", right_on="well", how="inner")
        # Drop metadata columns, keep only morphological features + INCHIKEY
        feat_cols = [c for c in df.columns if not c.startswith("Metadata_") and c not in ("well", "INCHIKEY")]
        chunks.append(df[["INCHIKEY"] + feat_cols])

    print(f"Loaded features from {len(chunks)}/{len(plates)} plates")
    all_features = pd.concat(chunks, ignore_index=True)
    feat_cols = [c for c in all_features.columns if c != "INCHIKEY"]

    # Average across all wells/plates per compound
    grouped = all_features.groupby("INCHIKEY")[feat_cols].mean()
    feature_names = np.array(grouped.columns.tolist())
    feature_matrix = grouped.values.astype(np.float32)
    ik_to_idx = {ik: i for i, ik in enumerate(grouped.index)}

    print(f"Feature matrix: {feature_matrix.shape[0]} compounds x {feature_matrix.shape[1]} features")
    return feature_matrix, feature_names, ik_to_idx


def summarize_features(vector: np.ndarray, names: np.ndarray, top_k: int) -> str:
    abs_vals = np.abs(vector)
    # Non-positive top_k means "use all available features".
    if top_k <= 0:
        top_k = len(names)
    top_k = min(top_k, len(names))
    top_idx = np.argsort(abs_vals)[-top_k:][::-1]
    lines: List[str] = []
    for idx in top_idx:
        val = vector[idx]
        if not np.isfinite(val):
            continue
        name = names[idx]
        direction = "Elevated" if val >= 0 else "Reduced"
        if abs(val) > 9999:
            score = f"{val:+.2e}"
        else:
            score = f"{val:+.2f}"
        lines.append(f"{direction} {name} (z={score})")
    return "\n".join(lines) if lines else "No reliable morphological signal."\


def count_direction(summary: str) -> tuple[int, int]:
    elevated = sum(line.startswith("Elevated") for line in summary.splitlines())
    reduced = sum(line.startswith("Reduced") for line in summary.splitlines())
    return elevated, reduced


def build_template_response(
    assay_name: str,
    summary: str,
    label: int,
) -> str:
    elevated, reduced = count_direction(summary)
    label_text = LABEL_MAP[label]
    return (
        "Step-by-step reasoning:\n"
        f"1. The compound's structure (see SMILES) suggests a specific scaffold under evaluation for {assay_name}.\n"
        f"2. Cell Painting reveals {elevated} elevated and {reduced} reduced features, for example:\n{summary}\n"
        f"3. These phenotypic shifts are {'consistent' if label else 'not consistent'} with the expected response in {assay_name}.\n"
        f"{format_final_answer(label)}"
    )


def build_endpoint_note(aid: str) -> str:
    """Generate a note for assays with multiple activity thresholds (e.g. 600885_1)."""
    if "_" not in aid:
        return ""
    base, suffix = aid.rsplit("_", 1)
    if not suffix.isdigit():
        return ""
    level = int(suffix)
    return f"Activity threshold: endpoint {level + 1} (stricter activity cutoff than the primary endpoint)\n"


def build_prompt(
    smiles: str,
    feature_summary: str,
    assay_name: str,
    assay_description: str,
    endpoint_note: str = "",
) -> str:
    return PROMPT_TEMPLATE.format(
        smiles=smiles,
        feature_summary=feature_summary,
        assay_name=assay_name,
        assay_description=assay_description,
        endpoint_note=endpoint_note,
    )


def select_assays(
    labels_df: pd.DataFrame,
    train_inchikeys: set,
    assays: List[str],
    cfg: dict,
    assay_meta: Optional[dict] = None,
) -> List[str]:
    include = cfg.get("include_assays") or []
    if include:
        include = {str(aid) for aid in include}
        return [aid for aid in assays if aid in include]

    require_desc = cfg.get("require_description", False)

    train_mask = labels_df["INCHIKEY"].isin(train_inchikeys)
    train_df = labels_df[train_mask]
    eligible: List[str] = []
    skipped_no_desc = 0
    min_total = cfg.get("min_labeled", 0)
    min_pos = cfg.get("min_pos_ratio", 0.0)
    max_pos = cfg.get("max_pos_ratio", 1.0)
    for aid in assays:
        if require_desc and assay_meta is not None:
            meta = assay_meta.get(aid, {})
            if not meta.get("description") and not meta.get("abstract"):
                skipped_no_desc += 1
                continue
        series = train_df[aid]
        labeled = series[series >= 0]
        n_total = len(labeled)
        if n_total < min_total:
            continue
        pos_ratio = (labeled == 1).mean() if n_total > 0 else 0
        if pos_ratio < min_pos or pos_ratio > max_pos:
            continue
        eligible.append(aid)
    if skipped_no_desc:
        print(f"Skipped {skipped_no_desc} assays without descriptions")
    return eligible


def load_curated_traces(traces_dir: Optional[Path]) -> dict:
    if traces_dir is None or not traces_dir.exists():
        return {}
    trace_map: dict = {}
    for path in traces_dir.glob("*_train_traces.jsonl"):
        aid = path.stem.split("_")[0]
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                key = (entry.get("inchikey"), aid)
                trace_map[key] = {
                    "prompt": entry.get("prompt"),
                    "response": entry.get("response"),
                }
    return trace_map


def ensure_alias(src: Path, alias: Path) -> None:
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    try:
        alias.symlink_to(src.relative_to(alias.parent))
    except ValueError:
        alias.symlink_to(src)
    except OSError:
        shutil.copy2(src, alias)


def make_dataset(
    split: SplitData,
    labels_df: pd.DataFrame,
    assays: List[str],
    csv_ik_to_smiles: dict,
    feature_lookup: dict,
    feature_names: np.ndarray,
    assay_meta: dict,
    cfg: dict,
    curated_traces: dict,
    fallback_style: str,
    rng: random.Random,
) -> tuple[List[dict], List[dict]]:
    records: List[dict] = []
    metadata_rows: List[dict] = []
    split_df = labels_df[labels_df["INCHIKEY"].isin(split.inchikeys)]
    limit = cfg.get("per_assay_limit")
    top_k = cfg.get("top_k_features", 15)

    for aid in assays:
        col = split_df[["INCHIKEY", aid]]
        labeled = col[col[aid] >= 0]
        if labeled.empty:
            continue
        if limit:
            labeled = labeled.sample(n=min(limit, len(labeled)), random_state=rng.randint(0, 10**6))

        assay_name = assay_meta.get(aid, {}).get("name", f"Assay {aid}")
        assay_desc = assay_meta.get(aid, {}).get("description", "")

        for _, row in labeled.iterrows():
            inchikey = row["INCHIKEY"]
            label = int(row[aid])
            smiles = csv_ik_to_smiles.get(inchikey)
            feat_idx = feature_lookup["index"].get(inchikey)
            if smiles is None or feat_idx is None:
                continue
            feature_vec = feature_lookup["matrix"][feat_idx]
            feature_summary = summarize_features(feature_vec, feature_names, top_k)

            if split.name == "train" and curated_traces:
                key = (inchikey, aid)
                trace_entry = curated_traces.get(key)
            else:
                trace_entry = None

            if trace_entry:
                instruction = trace_entry["prompt"].strip()
                output = trace_entry["response"].strip()
            else:
                endpoint_note = build_endpoint_note(aid)
                instruction = build_prompt(smiles, feature_summary, assay_name, assay_desc, endpoint_note)
                if fallback_style == "label_only":
                    output = format_final_answer(label)
                else:
                    output = build_template_response(assay_name, feature_summary, label)

            # Keep labels maximally separable for token-level objectives.
            if fallback_style == "label_only":
                output = format_final_answer(label)

            records.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                }
            )
            metadata_rows.append(
                {
                    "split": split.name,
                    "assay_id": aid,
                    "assay_name": assay_name,
                    "inchikey": inchikey,
                    "label": label,
                    "smiles": smiles,
                }
            )

    return records, metadata_rows


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(args.config)
    paths = resolve_paths(cfg, project_root)

    output_cfg = cfg.get("output", {})
    output_dir = project_root / output_cfg.get("dir", "data/prepared")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_cfg = cfg.get("prompt", {})
    reasoning_cfg = cfg.get("reasoning", {})

    rng = random.Random(args.seed)

    labels_df = pd.read_csv(paths.labels_csv)
    assay_cols = [c for c in labels_df.columns if c not in ("INCHIKEY", "SMILES")]
    csv_ik_to_smiles = dict(zip(labels_df["INCHIKEY"], labels_df["SMILES"]))

    feature_matrix, feature_names, ik_to_idx = load_plate_features(
        paths.features_dir, paths.plate_well_map
    )
    feature_lookup = {"matrix": feature_matrix, "index": ik_to_idx}

    splits = load_splits(paths)
    assay_meta = load_assay_metadata(paths.assay_desc_json)
    target_assays = select_assays(labels_df, splits["train"].inchikeys, assay_cols, prompt_cfg, assay_meta)
    curated_traces = (
        load_curated_traces(paths.traces_dir)
        if reasoning_cfg.get("use_curated_traces", False)
        else {}
    )
    fallback_style = reasoning_cfg.get("fallback_style", "template")

    all_records: Dict[str, List[dict]] = {}
    metadata: List[dict] = []
    for split_name in ("train", "val", "test"):
        records, meta_rows = make_dataset(
            splits[split_name],
            labels_df,
            target_assays,
            csv_ik_to_smiles,
            feature_lookup,
            feature_names,
            assay_meta,
            prompt_cfg,
            curated_traces,
            fallback_style,
            rng,
        )
        all_records[split_name] = records
        metadata.extend(meta_rows)
        out_file = output_dir / output_cfg.get(f"{split_name}_file", f"{split_name}.jsonl")
        with open(out_file, "w") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} {split_name} examples to {out_file}")

        alias_name = output_cfg.get(f"{split_name}_alias") or f"cpr_{split_name}.jsonl"
        alias_path = output_dir / alias_name
        ensure_alias(out_file, alias_path)
        print(f"Created alias {alias_path}")

    meta_file = output_dir / output_cfg.get("metadata_file", "metadata.csv")
    pd.DataFrame(metadata).to_csv(meta_file, index=False)
    print(f"Saved metadata to {meta_file}")


if __name__ == "__main__":
    main()
