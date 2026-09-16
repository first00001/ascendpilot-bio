#!/usr/bin/env python3
"""Build the review subset from official EMBL-EBI Single Cell Expression Atlas APIs."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path


ACCESSION = "E-ENAD-52"
GEO_ACCESSION = "GSE146035"
CLUSTERS = 28
ATLAS_REPORTED_CELLS = 28857
MAX_POINTS = 1120
UMAP_URL = (
    "https://www.ebi.ac.uk/gxa/sc/json/cell-plots/E-ENAD-52/clusters/k/28"
    "?plotMethod=UMAP&accessKey=&n_neighbors=20"
)
MARKERS_URL = (
    "https://www.ebi.ac.uk/gxa/sc/json/experiments/E-ENAD-52/"
    "marker-genes/clusters?k=28"
)
EXPERIMENT_URL = "https://www.ebi.ac.uk/gxa/sc/experiments/E-ENAD-52"
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146035"
OUTPUT = Path(__file__).with_name("e_enad_52_review_subset.json")


def fetch_json(url: str):
    request = urllib.request.Request(
        url, headers={"User-Agent": "AscendPilot-Bio public-data builder/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def evenly_spaced(items: list, count: int) -> list:
    if len(items) <= count:
        return items
    if count == 1:
        return [items[0]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def build_subset() -> dict:
    umap = fetch_json(UMAP_URL)
    marker_rows = fetch_json(MARKERS_URL)
    per_cluster = max(1, MAX_POINTS // max(1, len(umap["series"])))
    points = []
    samples = []
    cluster_sizes = {}
    for series in umap["series"]:
        cluster = str(series["name"]).removeprefix("Cluster ")
        cluster_sizes[cluster] = len(series["data"])
        chosen = evenly_spaced(series["data"], per_cluster)
        for x, y, barcode in chosen:
            points.append(
                {
                    "barcode": barcode,
                    "x": x,
                    "y": y,
                    "cluster": cluster,
                    "celltype_L1": f"Cluster {cluster}",
                    "celltype_L2": "Atlas unsupervised cluster",
                    "stage": "rice root tip",
                    "tissue": "seedling radicle",
                }
            )
        if chosen and len(samples) < 6:
            samples.append(
                {
                    "name": f"Public cell {chosen[0][2]} / Cluster {cluster}",
                    "expected_celltype_L1": f"Cluster {cluster}",
                    "barcode": chosen[0][2],
                    "cluster": cluster,
                    "genes": [f"ATLAS_CLUSTER_{cluster}"],
                    "counts": [[1.0]],
                }
            )

    markers_by_cluster: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in marker_rows:
        cluster = str(row.get("cellGroupValueWhereMarker", ""))
        gene = str(row.get("geneName", ""))
        if not cluster or not gene:
            continue
        current = markers_by_cluster[cluster].get(gene)
        candidate = {
            "gene": gene,
            "value_cpm": float(row.get("value", 0.0)),
            "p_value": float(row.get("pValue", 1.0)),
        }
        if current is None or candidate["value_cpm"] > current["value_cpm"]:
            markers_by_cluster[cluster][gene] = candidate

    top_markers = {}
    for cluster, genes in markers_by_cluster.items():
        top_markers[cluster] = sorted(
            genes.values(), key=lambda item: (-item["value_cpm"], item["gene"])
        )[:8]

    return {
        "schema_version": 1,
        "dataset": {
            "name": "Single-cell transcriptomic analysis of rice root tips",
            "atlas_accession": ACCESSION,
            "geo_accession": GEO_ACCESSION,
            "organism": "Oryza sativa",
            "source": "EMBL-EBI Single Cell Expression Atlas and NCBI GEO",
            "retrieved_on": date.today().isoformat(),
            "experiment_url": EXPERIMENT_URL,
            "geo_url": GEO_URL,
            "umap_api": UMAP_URL,
            "marker_api": MARKERS_URL,
            "cluster_count": CLUSTERS,
            "notice": (
                "Public Atlas clusters are unsupervised cluster identifiers, not curated "
                "biological cell-type labels."
            ),
        },
        "atlas_reported_cells": ATLAS_REPORTED_CELLS,
        "total_cells": sum(cluster_sizes.values()),
        "subset_cells": len(points),
        "cluster_sizes": cluster_sizes,
        "points": points,
        "samples": samples,
        "top_markers": top_markers,
    }


def main() -> None:
    payload = build_subset()
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    encoded = serialized.encode("utf-8")
    OUTPUT.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    print(f"Wrote {OUTPUT}")
    print(f"Cells: {payload['subset_cells']} / {payload['total_cells']}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
