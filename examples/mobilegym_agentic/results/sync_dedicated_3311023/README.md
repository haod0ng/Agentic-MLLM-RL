# Sync-dedicated MobileGym performance bundle — run 3311023

This bundle contains the successful **PRM-turn + ORM** run of the local,
GPU-disaggregated synchronous pipeline. It is a public raw-performance package:
timeline spans, GPU sampler records, TransferQueue trace records, and
rollout-latency records are retained in `raw_performance/`; derived reports and
figures are in `analysis/`. JSON/JSONL telemetry is gzip-compressed so the
repository can enforce its normal per-file publication limit; inspect it with
`gzip -cd <file>.gz`.

The package deliberately does not include prompts, generated actions, images,
screenshots, browser/session data, or raw environment-process dumps. Those are
not necessary to reproduce the performance accounting and cannot be published
with a public benchmark artifact. Cluster identities, absolute paths, addresses,
model paths, GPU UUIDs, and call stacks have been removed. `manifest.json`
enumerates every exported artifact with a SHA-256 digest.

## Configuration and validity boundary

- Policy: Qwen3-VL-8B; PRM: Qwen2.5-VL-3B; ORM: Qwen3-4B.
- Synchronous dedicated deployment: 4 actor GPUs, 12 rollout GPUs, 4 PRM GPUs,
  and 4 ORM GPUs (24 GPUs total).
- PRM deployment: TP=1, DP=4; maximum client concurrency=32.
- 192 rollout samples, three optimizer steps; the measured ready-to-ready window
  covers steps 1 to 2 and is 687.45 s. The run's cross-host clock bound is
  6.46 ms.
- This is **not** a fresh PRM-terminal comparison: the terminal-mode run in the
  same campaign did not complete a valid end-to-end window, so this directory
  makes no terminal-mode performance claim.

## Primary measured results

| Quantity                          |                        Measurement |
| --------------------------------- | ---------------------------------: |
| Rollout generation union          |    52.40 s (7.62% of ready window) |
| Reward-request WIP union          |                  164.54 s (23.93%) |
| Transfer execution union          |                  114.93 s (16.72%) |
| Training plus optimizer union     |                  193.39 s (28.13%) |
| Weight update union               |                     3.82 s (0.56%) |
| Session admission/gate wait union | 413.12 s (60.09%, diagnostic wait) |
| Trainer `data_wait` union         | 226.93 s (33.01%, diagnostic wait) |
| Transfer-buffer wait union        |  76.08 s (11.07%, diagnostic wait) |

The selected active stages overlap only for rollout and reward (46.68 s).
Training, transfer, and weight update are observed serially in this window;
therefore their inclusive percentages must not be added to claim a causal
critical path. The `latency_overlap_exposure` report/figure gives the exact
active-set accounting and caveats.

For TransferQueue, the three 64-sample partitions carry 7.6–7.9 GB each. The
same-process manager fan-in / ZMQ round trip is 56.90–58.65 s per partition,
whereas measured serialization and storage-side deserialize/store are in the
millisecond range. Cross-host monotonic clocks are not subtracted; the report
uses the stated ordinal join for storage-unit rows.

The environment bubble classifier finds 192 sessions with usable evidence. A
session is labelled idle only when its root `bench_env` process is S/D/I with no
root CPU ticks while its policy request is in flight. The mean observed policy
wait is 311.15 s per evidenced session and 98.37% is classified idle. This is
an observational occupancy result, not proof that browser CPU was the causal
throughput limit.

## Reproduce or inspect

Regenerate a new bundle from an experiment directory with:

```bash
python3.11 examples/mobilegym_agentic/scripts/export_public_experiment_bundle.py \
  --experiment-dir /path/to/experiment \
  --output-dir examples/mobilegym_agentic/results/sync_dedicated_RUN_ID
```

The analysis scripts in `examples/mobilegym_agentic/scripts/` consume the
unredacted private run directory when regenerating figures. The public bundle
is intended for verification and independent inspection of the reported timing
data, not for replaying benchmark interactions.
