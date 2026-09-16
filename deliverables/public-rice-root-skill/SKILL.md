---
name: public-rice-root-evidence
description: Inspect reproducible public rice root-tip single-cell clusters and marker evidence from EMBL-EBI E-ENAD-52 / NCBI GSE146035 without private data or model weights.
---

# Public Rice Root Evidence

Use this Skill when a reviewer needs a fully runnable, dependency-free public-data
path for the AscendPilot-Bio demonstration. It reads the checked-in review subset
generated from official EMBL-EBI Single Cell Expression Atlas APIs.

## Workflow

1. Run `python scripts/query_cluster.py summary` to verify the accession, public
   source, cell counts, cluster count, retrieval date, and evidence boundary.
2. Run `python scripts/query_cluster.py clusters` to list the 28 unsupervised
   Atlas clusters and their full-experiment cell counts.
3. Run `python scripts/query_cluster.py markers --cluster 1` to inspect the top
   marker-gene evidence for a cluster.
4. Run `python scripts/query_cluster.py cell --barcode BARCODE` to inspect a
   checked-in public UMAP point.
5. To reproduce the subset from the official APIs, run
   `python ../public_data/build_public_subset.py`.

## Evidence boundary

- Cluster IDs are unsupervised Atlas clusters, not curated biological cell types.
- Marker genes are evidence attached to the public cluster analysis; they do not
  prove cell identity or causality by themselves.
- This public Skill does not execute Qwen, riceFM, or Ascend NPU inference.
- Private ZH11 expression matrices, per-cell predictions, embeddings, weights,
  and checkpoints are intentionally excluded.
- Performance claims must cite the saved formal experiment records, not this
  lightweight review Skill.
