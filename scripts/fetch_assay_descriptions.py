#!/usr/bin/env python3
"""Fetch assay descriptions from PubChem PUG REST API and write reference/assay_descriptions.json."""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/summary/JSON"
RATE_LIMIT = 0.35  # seconds between requests (PubChem asks for ≤5 req/s)


def strip_suffix(aid: str) -> str:
    """Remove trailing _1, _2 etc. from assay IDs like '600885_1' -> '600885'."""
    return re.sub(r"_\d+$", "", aid)


def fetch_single(aid: str) -> dict | None:
    """Fetch summary for a single AID."""
    url = BASE_URL.format(aid=aid)
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError, json.JSONDecodeError):
        return None

    summaries = data.get("AssaySummaries", {}).get("AssaySummary", [])
    if not summaries:
        return None
    entry = summaries[0]
    return {
        "name": entry.get("AssayName", ""),
        "description": entry.get("Description", ""),
        "abstract": entry.get("Abstract", ""),
    }


def get_aids_from_metadata(metadata_path: Path) -> list[str]:
    """Extract unique assay IDs from metadata.csv."""
    aids = set()
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aids.add(row["assay_id"])
    return sorted(aids)


def get_aids_from_labels(labels_path: Path) -> list[str]:
    """Extract assay IDs from column headers of labels CSV."""
    with open(labels_path) as f:
        reader = csv.reader(f)
        header = next(reader)
    aids = [col for col in header if col not in ("INCHIKEY", "SMILES")]
    return sorted(set(aids))


def main():
    parser = argparse.ArgumentParser(description="Fetch PubChem assay descriptions")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/prepared/metadata.csv"),
        help="Path to metadata.csv (default: data/prepared/metadata.csv)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to labels CSV to extract all assay IDs from column headers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference/assay_descriptions.json"),
        help="Output JSON path (default: reference/assay_descriptions.json)",
    )
    args = parser.parse_args()

    # Get AIDs
    if args.labels and args.labels.exists():
        raw_aids = get_aids_from_labels(args.labels)
        print(f"Found {len(raw_aids)} assay IDs from labels CSV: {args.labels}")
    elif args.metadata.exists():
        raw_aids = get_aids_from_metadata(args.metadata)
        print(f"Found {len(raw_aids)} assay IDs from metadata: {args.metadata}")
    else:
        print(f"ERROR: neither --metadata ({args.metadata}) nor --labels found.")
        return

    # Deduplicate: strip _1/_2 suffixes, map original IDs to base PubChem AIDs
    base_to_originals: dict[str, list[str]] = {}
    for aid in raw_aids:
        base = strip_suffix(aid)
        base_to_originals.setdefault(base, []).append(aid)

    numeric_bases = {b for b in base_to_originals if b.isdigit()}
    print(f"  {len(raw_aids)} raw IDs -> {len(numeric_bases)} unique PubChem AIDs")

    # Fetch one at a time (avoids batch failures from bad IDs)
    fetched = {}
    for i, base_aid in enumerate(sorted(numeric_bases), 1):
        result = fetch_single(base_aid)
        if result:
            fetched[base_aid] = result
            status = "OK"
        else:
            status = "NOT FOUND"
        if i % 20 == 0 or i == len(numeric_bases):
            print(f"  Fetched {i}/{len(numeric_bases)} (latest: AID {base_aid} - {status})")
        time.sleep(RATE_LIMIT)

    print(f"\n  Successfully fetched: {len(fetched)}/{len(numeric_bases)}")

    # Build output: one entry per original AID, sharing description with base
    output_list = []
    missing = []
    for aid in raw_aids:
        base = strip_suffix(aid)
        info = fetched.get(base)
        if info:
            output_list.append({
                "aid": aid,
                "name": info["name"] or f"Assay {aid}",
                "description": info["description"],
                "abstract": info["abstract"],
            })
        else:
            missing.append(aid)
            output_list.append({
                "aid": aid,
                "name": f"Assay {aid}",
                "description": "",
                "abstract": "",
            })

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def sort_key(x):
        base = strip_suffix(x["aid"])
        return int(base) if base.isdigit() else 0

    output_list.sort(key=sort_key)
    with open(args.output, "w") as f:
        json.dump(output_list, f, indent=2)

    has_desc = sum(1 for r in output_list if r["description"] or r["abstract"])
    print(f"\nDone! Wrote {len(output_list)} assays to {args.output}")
    print(f"  {has_desc}/{len(output_list)} have descriptions")
    if missing:
        print(f"  {len(missing)} AIDs had no PubChem data (non-numeric or not found)")


if __name__ == "__main__":
    main()
