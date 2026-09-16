# Qwen3.5-0.8B Formal Single-NPU Comparison

Run completed on 2026-09-15 at 17:15:27 Asia/Shanghai. The existing
Qwen3.5-4B service was left running on one physical NPU; both compared training
variants ran sequentially on the other physical NPU.

## Results

| Window | AscendC samples/s | Triton samples/s | AscendC throughput change | Mean loss abs diff | Max loss abs diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Steps 1-100 | 1.300702 | 1.232317 | +5.55% | 0.00126029 | 0.004655 |
| Steps 11-100 | 1.890313 | 2.745005 | -31.14% | 0.00113323 | 0.003936 |

The final-step absolute loss difference is `0.001393` in both windows. Report
both timing windows; the end-to-end result includes initialization and compile
effects and does not establish universal AscendC acceleration.

## Evidence

- `formal_single_100_comparison.json`: remote comparison output for steps
  1-100.
- `formal_single_steady_comparison.json`: remote comparison output for steps
  11-100.
- `formal_single_evidence_20260915.tar.gz`: verified remote evidence bundle.
- `remote-evidence/logs/ascendc_formal_single/`: complete AscendC 100-step log.
- `remote-evidence/logs/triton_formal_single/`: complete Triton 100-step log.
- `remote-evidence/results/formal_single_config_diff.txt`: confirms the expected
  GDN implementation and save-directory differences.
- `remote-evidence/results/formal_single_checkpoint_files.txt`: checkpoint file
  names and sizes.
- `remote-evidence/results/formal_single_checkpoint_sha256.txt`: hashes of the
  remote checkpoint metadata and shards. The multi-gigabyte checkpoint shards
  themselves remain on the remote host.
- `remote-evidence/logs/ascendc_formal/`: preserved failed two-NPU attempt. It
  contains no completed iterations and is excluded from comparison statistics.

Evidence archive SHA-256:

```text
60b7b0039c7418152fca63184737e9e491d3e9e84eec11eb920bdd511f03da40
```
