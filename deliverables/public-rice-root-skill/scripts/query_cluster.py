#!/usr/bin/env python3
"""Query the checked-in E-ENAD-52/GSE146035 public review subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA = (
    Path(__file__).resolve().parents[2]
    / "public_data"
    / "e_enad_52_review_subset.json"
)


def load() -> dict:
    if not DATA.is_file():
        raise SystemExit(
            "public subset is missing; run "
            "python deliverables/public_data/build_public_subset.py"
        )
    return json.loads(DATA.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    sub.add_parser("clusters")
    markers = sub.add_parser("markers")
    markers.add_argument("--cluster", required=True)
    cell = sub.add_parser("cell")
    cell.add_argument("--barcode", required=True)
    args = parser.parse_args()
    data = load()

    if args.command == "summary":
        output = {
            "dataset": data["dataset"],
            "total_cells": data["total_cells"],
            "checked_in_subset_cells": data["subset_cells"],
            "available_clusters": len(data["cluster_sizes"]),
            "model_execution": False,
        }
    elif args.command == "clusters":
        output = [
            {"cluster": cluster, "cells": cells}
            for cluster, cells in sorted(
                data["cluster_sizes"].items(), key=lambda item: int(item[0])
            )
        ]
    elif args.command == "markers":
        markers = data["top_markers"].get(str(args.cluster))
        if markers is None:
            raise SystemExit(f"unknown cluster: {args.cluster}")
        output = {
            "cluster": str(args.cluster),
            "markers": markers,
            "boundary": "Atlas unsupervised cluster; not a curated cell type",
        }
    else:
        output = next(
            (point for point in data["points"] if point["barcode"] == args.barcode),
            None,
        )
        if output is None:
            raise SystemExit(f"barcode is not in the checked-in subset: {args.barcode}")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
