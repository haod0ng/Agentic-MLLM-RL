# OSWorld rollout-only experiment, 2026-09-11

This directory publishes a sanitized aggregate record of a fixed-checkpoint
OSWorld rollout-only experiment. The experiment used one node, four TP1 policy
engines, and colocated OSWorld virtual machines. It exercised VM readiness,
native actions, native evaluation, release, fixed-trajectory replay, and two
live broker conditions.

## Result

The campaign **did not meet its acceptance criteria**. It is therefore not
evidence of a deployment-ready concurrency setting or of an inference
improvement over a synchronous RL deployment.

The native capacity checks passed at C=4, C=8, and C=16: each condition reached
the requested number of concurrent VMs, executed native `WAIT`, evaluated, and
released every VM. This establishes an observed capacity ceiling of C=16 for
this allocation only.

The real-policy action gate failed: all 32 strict action parses in the probe
were invalid and no broker action was executed. The remaining workload used
explicitly labelled controlled-WAIT trajectories. It retained model-generated
assistant content but replaced actions with a legal native `WAIT`; it cannot
substantiate real-policy serving performance.

Every replay matrix cell ran the same 128-request collection, but all failed
the 95% live-engine-load coverage requirement. The C=4/alpha=1 cell reached
94.986512%, which remains a failure without rounding. The replay outputs,
finish reasons, and cache signatures also differed across cells, so the raw
throughput values cannot be interpreted as a fixed-compute estimate of
environment-wait loss.

The C=4/delay=0 live condition completed two batches but missed the engine-load
coverage gate. The C=16/delay=5 condition exported two batches, then reached a
cell deadline before writing the driver-success marker; one injected delay was
5.277009 seconds, beyond the 5.25-second tolerance. The C=4/delay=5 and
C=16/delay=0 cells, fresh repeats, and intermediate conditions did not run.

## Fixed configuration

| Setting                | Value                                              |
| ---------------------- | -------------------------------------------------- |
| Model                  | Qwen3-VL-4B-Instruct fixed checkpoint              |
| Inference topology     | Four TP1 engines, round-robin routing              |
| GPU allocation         | One node, four GPUs                                |
| Static memory fraction | 0.4                                                |
| Sampling               | temperature=1, seed=42, maximum output=1024 tokens |
| Interaction limit      | Eight steps per trajectory                         |
| Prompt group size      | Four trajectories                                  |
| Workload label         | `repeated_pinned_task_workload`                    |

`fully_async` selected the fixed-model rollout execution path. It was not an
asynchronous RL experiment. No Actor, Critic, reference, Judge, optimizer, or
weight update was created in the rollout-only runtime.

## Published aggregate data

All files below contain aggregate measurements only. They exclude prompts,
responses, screenshots, action coordinates, credentials, VM identifiers, host
names, and private replay payloads.

- [Campaign summary JSON](data/campaign_summary.json) records scope, gates, configuration, and cost.
- [Capacity summary](data/capacity_summary.csv) records the C=4/8/16 VM checks.
- [Replay diagnostics](data/replay_diagnostics.csv) records the four replay cells and capacity reference. Every row is unvalidated.
- [Live diagnostics](data/live_diagnostics.csv) records the two executed live conditions. Every row is unvalidated.
- [Inter-turn gaps](data/inter_turn_gaps.csv) records the reconstructed action and residual-gap aggregates.
- [Allocation cost](data/allocation_cost.csv) records all allocation attempts without publishing scheduler identifiers.

The reported throughput denominator is four GPUs times the complete rollout
interaction window. That window includes request-free gaps and finite-batch
tails. Allocation cost includes startup, probes, failures, and cleanup.

## Interpretation limits

Engine load was sampled at a target 0.2-second interval and host state at one
second. Successful metric HTTP requests do not prove fresh scheduler state. In
this campaign, direct scheduler-load reads were often slower than the target
interval, so the verifier rejected coverage below 95% rather than treating
scrape success as coverage.

The host pressure interface was unavailable. CPU, I/O, cgroup throttling, and
OOM counters were retained, but missing PSI blocks a deployment recommendation.
One campaign also supplies no independent-run confidence interval.

The four allocation attempts consumed 5.707778 GPU-hours in total, below the
8 GPU-hour budget. Final cleanup found no remaining experiment-owned virtual
machines, broker manifests, or runtime roots. A later launcher fix separates
admission estimates from an admitted cell's execution deadline; it passed CPU
regression tests after the allocation ended and was not revalidated on GPUs in
this campaign.
