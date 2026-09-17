# Public review dataset

The offline review system uses a compact subset of the public rice root-tip
single-cell experiment `E-ENAD-52` / `GSE146035` instead of the private ZH11
assets used by the Ascend deployment.

The checked-in JSON contains stratified public UMAP points, Atlas cluster IDs,
and top marker-gene evidence for the 28-cluster analysis. Atlas cluster numbers
are unsupervised clusters and are not presented as curated biological cell-type
labels. The Atlas experiment page reports 28,857 cells, while the selected
28-cluster UMAP endpoint contains 28,856 clustered cells; both counts are recorded
explicitly. No private expression matrix, barcode, prediction, model weight, or
checkpoint is included.

Rebuild from the official APIs:

```bash
python deliverables/public_data/build_public_subset.py
```

Sources:

- EMBL-EBI Single Cell Expression Atlas experiment `E-ENAD-52`
- NCBI GEO series `GSE146035`
- Publication PMID `33824350`

The generated JSON records the exact API endpoints and retrieval date used for
provenance. Reviewers should use the upstream records for the authoritative
metadata and terms of use.

## riceFM public compatibility pilot

`gse232863_ricefm_anchor_pilot.json` contains 64 real cells from public GEO
sample `GSM8865415` (`GSE232863`, sample `E10_1`) and 25 gene anchors. The source
HDF5 contains 17,133 cells and has SHA-256
`e11a4890457c8537df73e2945c989b39749d4534fff6979c36147cb2c9e37a46`.

The mapping uses riceFM official tutorial anchors cross-referenced to Oryzabase
RAP IDs. It is intentionally limited to 25 genes and is not a whole-transcriptome
mapping. The corresponding Ascend execution record is stored at
`../results/ricefm-public-pilot/gse232863_ricefm_anchor_pilot_result.json`.
It proves that riceFM generated finite 256-dimensional embeddings; it does not
claim a validated cell-type prediction or classification head.
