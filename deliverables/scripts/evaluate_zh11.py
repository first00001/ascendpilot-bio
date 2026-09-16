#!/usr/bin/env python3
"""Generate riceFM embeddings and evaluate ZH11 reference-label transfer."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def matrix_shape(path: Path):
    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line in stream:
            if not line.startswith("%"):
                return tuple(map(int, line.split()))
    raise ValueError("Matrix Market dimensions are missing")


def matrix_batches(path: Path, batch_size: int):
    n_genes, n_cells, _ = matrix_shape(path)
    with gzip.open(path, "rt", encoding="ascii") as stream:
        dimensions_seen = False
        start_cell = 0
        batch = np.zeros((min(batch_size, n_cells), n_genes), dtype=np.float32)
        for line in stream:
            if line.startswith("%"):
                continue
            if not dimensions_seen:
                dimensions_seen = True
                continue
            gene, cell, value = line.split()
            cell_index = int(cell) - 1
            while cell_index >= start_cell + len(batch):
                yield start_cell, batch
                start_cell += len(batch)
                remaining = n_cells - start_cell
                batch = np.zeros((min(batch_size, remaining), n_genes), dtype=np.float32)
            batch[cell_index - start_cell, int(gene) - 1] = float(value)
        if len(batch):
            yield start_cell, batch


def load_labels(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="	"))
    return rows, np.asarray([row["celltype_L1"] for row in rows])


def stratified_split(labels: np.ndarray, seed: int, test_ratio: float):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = max(1, min(len(indices) - 1, round(len(indices) * test_ratio)))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    return np.asarray(sorted(train)), np.asarray(sorted(test))


def cosine_knn(train_x, train_y, test_x, k: int):
    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    test_x = test_x / np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), 1e-12)
    predictions = []
    for start in range(0, len(test_x), 256):
        similarities = test_x[start : start + 256] @ train_x.T
        neighbors = np.argpartition(similarities, -k, axis=1)[:, -k:]
        for row, indices in zip(similarities, neighbors):
            votes: dict[str, float] = {}
            for index in indices:
                label = str(train_y[index])
                votes[label] = votes.get(label, 0.0) + float(max(row[index], 0.0))
            predictions.append(max(votes, key=lambda label: (votes[label], label)))
    return np.asarray(predictions)


def classification_metrics(truth: np.ndarray, predicted: np.ndarray):
    labels = sorted(set(truth) | set(predicted))
    per_class, confusion = {}, {}
    for label in labels:
        tp = int(np.sum((truth == label) & (predicted == label)))
        fp = int(np.sum((truth != label) & (predicted == label)))
        fn = int(np.sum((truth == label) & (predicted != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": int(np.sum(truth == label)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        confusion[label] = {
            target: int(np.sum((truth == label) & (predicted == target))) for target in labels
        }
    supports = np.asarray([per_class[label]["support"] for label in labels])
    f1s = np.asarray([per_class[label]["f1"] for label in labels])
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(f1s.mean()),
        "weighted_f1": float(np.average(f1s, weights=supports)),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from skill.ricefm_adapter import RiceFMRuntime

    started = time.perf_counter()
    n_genes, n_cells, nonzero = matrix_shape(args.data_dir / "matrix.mtx.gz")
    with (args.data_dir / "features.tsv").open("r", encoding="utf-8") as stream:
        genes = [line.rstrip("\n").split("\t")[1] for line in stream]
    rows, labels = load_labels(args.data_dir / "metadata.tsv")
    if (n_cells, n_genes) != (len(rows), len(genes)):
        raise ValueError(
            f"matrix shape {(n_cells, n_genes)} does not match metadata/features"
        )
    load_seconds = time.perf_counter() - started

    runtime = RiceFMRuntime(args.model_dir, args.repo_dir, args.device, args.seed)
    inference_started = time.perf_counter()
    embedding_batches = []
    for cell_offset, counts in matrix_batches(args.data_dir / "matrix.mtx.gz", args.batch_size):
        embedding_batches.append(
            runtime.cell_embeddings(counts, genes, args.batch_size, cell_offset)
        )
    embeddings = np.concatenate(embedding_batches)
    inference_seconds = time.perf_counter() - inference_started
    np.save(args.output_dir / "embeddings.npy", embeddings)

    train, test = stratified_split(labels, args.seed, args.test_ratio)
    predicted = cosine_knn(embeddings[train], labels[train], embeddings[test], args.k)
    metrics = classification_metrics(labels[test], predicted)
    metrics.update(
        {
            "dataset": "ZH11_riceFM_eval",
            "cells": n_cells,
            "genes": n_genes,
            "nonzero_values": nonzero,
            "celltype_L1_classes": len(set(labels)),
            "train_cells": len(train),
            "test_cells": len(test),
            "split": "deterministic stratified holdout",
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "classifier": f"cosine distance-weighted {args.k}-NN",
            "embedding_model": "riceFM pretraining checkpoint",
            "embedding_dimensions": int(embeddings.shape[1]),
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "cells_per_second": len(embeddings) / inference_seconds,
            "label_counts": dict(sorted(Counter(labels).items())),
            "limitation": (
                "The checkpoint may have seen overlapping ZH11 profiles during pretraining; "
                "this is a labeled-reference transfer evaluation, not an independent external test."
            ),
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["barcode", "true_celltype_L1", "predicted_celltype_L1"])
        for index, prediction in zip(test, predicted):
            writer.writerow([rows[index][""], labels[index], prediction])
    print(json.dumps({key: value for key, value in metrics.items() if key not in {
        "per_class", "confusion_matrix", "label_counts"
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
