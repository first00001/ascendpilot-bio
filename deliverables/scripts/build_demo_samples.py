#!/usr/bin/env python3
"""Extract compact, deterministic demo cells from a Matrix Market dataset."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_TYPES = ("Pollen", "Ovary", "Tapetum", "Cortex")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-genes", type=int, default=512)
    parser.add_argument("--cell-types", nargs="+", default=DEFAULT_TYPES)
    args = parser.parse_args()

    with (args.data_dir / "metadata.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        metadata = list(csv.DictReader(stream, delimiter="\t"))
    with (args.data_dir / "features.tsv").open("r", encoding="utf-8") as stream:
        genes = [line.rstrip("\n").split("\t")[1] for line in stream]

    chosen: dict[int, dict[str, str]] = {}
    remaining = set(args.cell_types)
    for index, row in enumerate(metadata):
        label = row["celltype_L1"]
        if label in remaining:
            chosen[index] = row
            remaining.remove(label)
    if remaining:
        raise ValueError(f"cell types not found: {sorted(remaining)}")

    expression: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with gzip.open(args.data_dir / "matrix.mtx.gz", "rt", encoding="ascii") as stream:
        dimensions_seen = False
        for line in stream:
            if line.startswith("%"):
                continue
            if not dimensions_seen:
                dimensions_seen = True
                continue
            gene_index, cell_index, value = line.split()
            zero_cell = int(cell_index) - 1
            if zero_cell in chosen:
                expression[zero_cell].append((int(gene_index) - 1, float(value)))

    samples = []
    for index, row in chosen.items():
        values = sorted(expression[index], key=lambda item: (-item[1], item[0]))[
            : args.top_genes
        ]
        samples.append(
            {
                "name": f'{row["celltype_L1"]} / {row["stage"]}',
                "barcode": row[""],
                "expected_celltype_L1": row["celltype_L1"],
                "celltype_L2": row["celltype_L2"],
                "stage": row["stage"],
                "tissue": row["tissue"],
                "genes": [genes[gene] for gene, _ in values],
                "counts": [[value for _, value in values]],
            }
        )
    samples.sort(key=lambda item: args.cell_types.index(item["expected_celltype_L1"]))
    payload = {
        "dataset": "ZH11_riceFM_eval",
        "note": "Each preset contains the 512 highest-count nonzero genes from one real cell.",
        "samples": samples,
    }
    output = args.output or args.data_dir / "demo_samples.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "samples": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
