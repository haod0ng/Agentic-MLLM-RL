# MobileGym dual-Judge latency evaluation: evidence review

## Status

This document is a corrected review of historical measurements from a fully asynchronous MobileGym training setup.
The raw traces, trajectories, screenshots, model outputs, and cluster logs are not published. The observations below
are therefore descriptive and cannot be independently recomputed from this repository alone.

The implementation and analyzers are public; a completed, replicated, frozen-workload experiment is not. Do not cite
the historical runs as a causal performance result.

## System evaluated

The historical setup used a 24-GPU layout with one policy trainer, twelve rollout engines, and two four-GPU Judge
services. The training reward combined a state-check-assisted terminal outcome Judge and a multimodal trajectory
Judge. Two operational modes were observed:

- `terminal_once`: score the complete trajectory once after termination;
- `per_turn`: score completed response-to-observation interactions during rollout and average their scores at the
  trajectory boundary.

Both historical arms used `benchmark_mode=dual`. Consequently, they changed reward semantics and future policy
trajectories as well as scheduling. They were not a frozen, reward-preserving scheduler A/B test.

## Measurement layers

The analysis uses three different quantities. They must not be merged into one percentage attribution.

| Layer                    | Estimand                                                        | Valid interpretation                                                      |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Inclusive span occupancy | Union of each named span in a selected wall-clock window        | Descriptive activity; stages overlap and totals may exceed 100%           |
| Active-set partition     | Exact combination of observed spans at each instant             | Additive observed-time partition; `no_observed_span` is not hardware idle |
| Per-sample chain         | Consecutive timestamps from generation through transfer release | Arithmetic latency accounting for that chain; not a causal intervention   |

`scripts/chain_decomposition.py` ends at `transfer_release_end_at`. It does not include trainer consumption,
optimizer execution, weight synchronization, or the next serving-ready boundary. Its `H` value is therefore a
post-generation-to-transfer-release chain, not full end-to-end training latency. A zero residual follows from using
consecutive timestamp boundaries and proves closure only.

The primary end-to-end systems estimand should instead be ready-to-ready publication makespan under a fixed set of
publication IDs. Queue latency, stage occupancy, GPU samples, request timing, and the per-sample chain are explanatory
channels.

## What the historical data supports

The private single-run diagnostics support these limited observations:

1. In the observed `terminal_once` run, client-side admission queueing was a large part of the process-Judge branch.
2. Long reward, group-finalization, transfer-release, and trainer data-wait intervals co-occurred.
3. The observed `per_turn` smoke runs had shorter publication intervals than the observed `terminal_once` run.
4. The process Judge was much slower than the outcome Judge in that configuration.

These are useful hypotheses for experiment design. They do not establish that reward was the unique binding
bottleneck, that a particular percentage of end-to-end latency was caused by reward, or that `per_turn` would retain
the same direction and magnitude under independent repeats.

## Claims rejected by this review

### "Reward causes 78.6% of end-to-end latency"

The percentage combined wait, reward execution, and group-finalization segments inside a closed per-sample tail. It
does not represent a counterfactual E2E delta, and group-finalization can include straggler effects from other
samples. The defensible statement is that reward admission and transfer gating are candidates for controlled study.

### "The Judge was 88% idle because of a known three-part decomposition"

NVML sample fractions, client queue time, engine HTTP time, and SGLang request counters are different denominators.
They cannot be added into an idle-cause decomposition. A ratio derived from sampled GPU-hours and request count is
not per-request GPU compute time. Missing actor-side GPU telemetry remains `N/A` for the historical runs.

### "Arrival rate exceeded proven service capacity"

`concurrency / median service time` is not a stability-capacity proof for a bursty, multimodal service with variable
request sizes. It is a queueing heuristic. Increasing `max_concurrency` may increase contention and service time, so
it requires a controlled sweep rather than a configuration-only conclusion.

### "Network transfer is ruled out"

The average payload rate was small relative to nominal link bandwidth, but that does not exclude burst congestion,
serialization, base64 allocation, two-hop transport, or tail effects. The current proxy also reports text prompt
tokens rather than the model-specific visual-token expansion, so multimodal capacity conclusions based on that
counter alone are invalid.

### "Per-turn is conclusively faster"

The observed direction is encouraging but the historical runs used different online rewards, trajectories, and
request granularity; there were no independent paired replicates or confidence interval. Per-turn also moves work
into rollout and terminal join spans and excludes a normally unobserved final response from the process-Judge mean.

## Required causal design

Use reward-preserving frozen replay before an online total-system comparison.

1. Materialize a canonical trajectory corpus once, including media hashes, state checks, final outputs, sample IDs,
   and arrival schedule.
2. Replay the same corpus against `terminal_once` and `per_turn` services using immutable model revisions and the
   same engine, GPU allocation, sampling configuration, and request retry policy.
3. Run systems arms with `dual_shadow` or another fixed recorded training reward. Include rejected, retried,
   cancelled, and right-censored work.
4. Randomize arm order and run at least three independent deployment-level pairs. Use the deployment pair as the
   inference unit; publication rounds inside one deployment are autocorrelated observations.
5. Stop admission, drain pending work, flush timeline and GPU channels, then measure the same ready-to-ready window.
6. Report success rate, sample/group cardinality, workload identity, telemetry coverage, and GPU-seconds alongside
   makespan and throughput.

Suggested primary comparisons:

| Comparison                                         | Estimand                                                       |
| -------------------------------------------------- | -------------------------------------------------------------- |
| `dual_shadow terminal_once - recorded`             | Combined terminal context plus dual-Judge operational overhead |
| `dual_shadow per_turn - dual_shadow terminal_once` | Reward-preserving trigger/scheduling delta on frozen work      |
| `force concurrency c2 - c1` within one trigger     | Concurrency intervention, including changed service time       |
| online `dual per_turn - dual terminal_once`        | Joint reward-and-system effect; not a scheduler-only effect    |

## Analyzer contract

For a result to be valid, the analyzer must verify:

- identical benchmark invariant hash and immutable model revisions;
- exact sample/group cardinality for every measured publication;
- matching terminal trajectory and consumed-reward identity in paired systems arms;
- complete terminal status, including failures and cancellations;
- one fixed ready-to-ready publication sequence selected from the baseline;
- stable component/rank sets and synchronized clocks when events span hosts;
- explicit telemetry coverage, with missing channels reported as `N/A`, never zero.

The public analyzers implement many of these checks, but a report is only as valid as the collection boundary and
artifacts supplied to it.

## Reproducibility status

| Gate                                            | Status                                      |
| ----------------------------------------------- | ------------------------------------------- |
| Dual-Judge CPU/unit tests                       | available in repository                     |
| Direct real-model Judge test                    | opt-in; requires local GPUs and checkpoints |
| Full SessionShard to trainer GPU E2E            | not completed for this public snapshot      |
| Frozen, reward-preserving trigger comparison    | not completed                               |
| Independent paired replicates                   | not completed                               |
| Public anonymized artifact bundle and checksums | not included                                |

Until the final four gates are complete, the appropriate conclusion is:

> Historical single-run descriptive diagnostics suggest that reward admission and transfer gating are important
> candidates. No controlled causal comparison of trigger mode or concurrency has completed.
