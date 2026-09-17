# riceFM public anchor pilot

This directory records a real riceFM execution on an Ascend 910 using public
rice single-cell data from `GSE232863` / `GSM8865415` (`E10_1`).

The run used 64 public cells and a validated 25-gene compatibility subset. It
produced finite `[64, 256]` embeddings at 172.37 cells/s after model loading.
The JSON includes the device, source HDF5 hash, checkpoint path, embedding hash,
timing, and anchor detection summary.

This is execution and gene-ID compatibility evidence. It is not a
whole-transcriptome mapping, independent biological validation, or cell-type
classification result.

SHA-256:

```text
85ef8cc74f9b4189c18f8e04b98c7a53988ff6737b313ec4dded0aa550e0edf8  gse232863_ricefm_anchor_pilot_result.json
```
