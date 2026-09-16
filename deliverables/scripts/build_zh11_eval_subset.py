#!/usr/bin/env python3
"""Build a deterministic, cell-type-stratified ZH11 Matrix Market subset."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_cells(metadata: Path, per_type: int, seed: int):
    rng = random.Random(seed)
    reservoirs: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)

    with metadata.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        type_idx = header.index("celltype_L1")
        for cell_index, row in enumerate(reader, start=1):
            label = row[type_idx].strip()
            if not label or label.lower() == "unknown":
                continue
            seen[label] += 1
            bucket = reservoirs[label]
            item = (cell_index, row)
            if len(bucket) < per_type:
                bucket.append(item)
            else:
                replacement = rng.randrange(seen[label])
                if replacement < per_type:
                    bucket[replacement] = item

    selected = sorted(
        (item for bucket in reservoirs.values() for item in bucket),
        key=lambda item: item[0],
    )
    return header, selected, seen


def copy_selected_lines(source: Path, destination: Path, selected_indices: set[int]) -> None:
    with source.open("r", encoding="utf-8") as src, destination.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        for index, line in enumerate(src, start=1):
            if index in selected_indices:
                dst.write(line)


def subset_matrix(
    source: Path,
    destination: Path,
    old_to_new: dict[int, int],
    n_genes: int,
) -> int:
    body = destination.with_suffix(".body")
    nnz = 0
    dimensions_seen = False
    with source.open("r", encoding="ascii") as src, body.open(
        "w", encoding="ascii", newline="\n"
    ) as dst:
        for line in src:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            if not dimensions_seen:
                dimensions_seen = True
                continue
            old_column = int(parts[1])
            new_column = old_to_new.get(old_column)
            if new_column is None:
                continue
            dst.write(f"{parts[0]} {new_column} {parts[2]}\n")
            nnz += 1

    with destination.open("w", encoding="ascii", newline="\n") as dst, body.open(
        "r", encoding="ascii"
    ) as src:
        dst.write("%%MatrixMarket matrix coordinate real general\n%\n")
        dst.write(f"{n_genes} {len(old_to_new)} {nnz}\n")
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    body.unlink()
    return nnz


def subset_h5ad(
    source: Path,
    destination: Path,
    selected: list[tuple[int, list[str]]],
    n_genes: int,
) -> int:
    """Write selected CSR rows from an h5ad file as a genes-by-cells MTX."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required when --h5ad is used") from exc

    with h5py.File(source, "r") as h5:
        matrix = h5["X"]
        shape = tuple(int(value) for value in matrix.attrs["shape"])
        if shape[1] != n_genes:
            raise ValueError(f"h5ad has {shape[1]} genes, expected {n_genes}")
        indptr = matrix["indptr"]
        indices = matrix["indices"]
        data = matrix["data"]
        spans = [(int(indptr[old - 1]), int(indptr[old])) for old, _ in selected]
        nnz = sum(end - start for start, end in spans)

        with gzip.open(destination, "wt", encoding="ascii", newline="\n", compresslevel=6) as dst:
            dst.write("%%MatrixMarket matrix coordinate integer general\n%\n")
            dst.write(f"{n_genes} {len(selected)} {nnz}\n")
            for new_column, (start, end) in enumerate(spans, start=1):
                row_indices = indices[start:end]
                row_values = data[start:end]
                for gene_index, value in zip(row_indices, row_values):
                    dst.write(f"{int(gene_index) + 1} {new_column} {int(value)}\n")
    return nnz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--h5ad", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.source / "metadata.tsv"
    features = args.source / "features.tsv"
    barcodes = args.source / "barcodes.tsv"
    matrix = args.source / "matrix.mtx"

    header, selected, seen = select_cells(metadata, args.per_type, args.seed)
    old_to_new = {old: new for new, (old, _) in enumerate(selected, start=1)}
    selected_indices = set(old_to_new)

    out_metadata = args.output / "metadata.tsv"
    with out_metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(row for _, row in selected)

    shutil.copy2(features, args.output / "features.tsv")
    copy_selected_lines(barcodes, args.output / "barcodes.tsv", selected_indices)
    with features.open("r", encoding="utf-8") as stream:
        n_genes = sum(1 for _ in stream)
    compressed = args.output / "matrix.mtx.gz"
    if args.h5ad:
        nnz = subset_h5ad(args.h5ad, compressed, selected, n_genes)
    else:
        nnz = subset_matrix(matrix, args.output / "matrix.mtx", old_to_new, n_genes)
        with (args.output / "matrix.mtx").open("rb") as src, gzip.open(
            compressed, "wb", compresslevel=6
        ) as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        (args.output / "matrix.mtx").unlink()

    manifest = {
        "source": str(args.source),
        "seed": args.seed,
        "per_celltype_L1": args.per_type,
        "cells": len(selected),
        "genes": n_genes,
        "nonzero_values": nnz,
        "available_celltype_L1_counts": dict(sorted(seen.items())),
        "files": {},
    }
    for path in sorted(args.output.iterdir()):
        if path.name == "manifest.json":
            continue
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
