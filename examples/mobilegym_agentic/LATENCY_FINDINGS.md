# Sanitized latency investigation notes

This file replaces the original chronological lab diary. Internal paths, hosts, addresses, scheduler metadata, job
identifiers, and raw log excerpts are intentionally omitted. The formal evidence assessment is in
[EVALUATION_REPORT.md](EVALUATION_REPORT.md).

## Investigation outcome

Historical traces led to four useful engineering hypotheses:

1. terminal process-Judge requests arrived in bursts and spent substantial time behind admission control;
2. reward completion, group finalization, and TransferQueue release were frequently followed by trainer data-wait;
3. per-turn submission smoothed some request bursts in the observed runs, but also moved work into sidecars and a
   terminal join;
4. multimodal process-Judge service time was much larger than the text outcome-Judge service time, with model size,
   media count, preprocessing, and engine behavior all confounded.

The investigation also found that several attractive summaries were not statistically or causally justified:

- timestamp-chain closure was called causal attribution;
- dissimilar utilization and latency denominators were combined as an idle-time decomposition;
- one warmup convention was described while another was used for a headline table;
- operational `dual` runs were treated as if trajectories and training rewards were paired;
- smoke-run direction was reported without deployment-level repeats;
- average payload rate was used to rule out burst and tail effects.

Those claims are not retained in the public report.

## Instrumentation retained in the repository

- reward context, queue, client HTTP, server queue, tokenization, engine HTTP, retry, and branch timing;
- per-turn scheduling and terminal-barrier events;
- group finalization, TransferQueue release, trainer data-wait, optimizer, and weight-publication spans;
- best-effort NVML/SGLang GPU sampling with explicit coverage;
- workload, reward, service-config, and publication identity checks;
- fixed-window and independent-pair aggregation tools.

Instrumentation presence is not proof that every historical run contained every channel. Missing measurements must
remain `N/A`.

## Open questions

### Multimodal service time

Replay a frozen corpus while varying image count, pixel budget, and client concurrency. Add a zero-image control and
run the same text prompt on both Judge model families. Measure the model/processor-specific visual token expansion,
not only serialized prompt text.

### Concurrency

Sweep the role-level service limit while preserving request arrivals. Report both queue reduction and any increase
in service time, failure rate, memory use, and GPU-seconds. Do not infer capacity from `c / p50` alone.

### Transfer gating

Treat `num_iters_per_train_update` and the transfer flush threshold as training-semantic axes, not free systems
optimizations. A changed batch/normalization contract must be evaluated for reward and optimization equivalence.

### Trigger mode

First compare triggers with frozen trajectories and rewards. Only then run online `dual` experiments to measure the
joint effect on future policy, task length, reward distribution, and system throughput.

## Known system limitations discovered during bring-up

The historical cluster attempts exposed failure modes that are not resolved by the published evaluation:

- a restarted rollout service could leave the resident dataflow loop without useful forward progress;
- launcher-side validation could run before a complete publication boundary;
- health signaling did not demonstrate recovery from a killed SGLang child;
- full cleanup and stale completion-file behavior were not fault-injection tested.

These observations motivated stronger readiness and lifecycle gates. They are not presented as confirmed root causes
without a reproducible public artifact.

## Publication checklist for a future result

- anonymous run IDs and an offline private mapping;
- immutable code, model, container, data, and environment revisions;
- sanitized metric JSON plus manifest, analyzer command, and checksums;
- success/failure/retry/cancellation counts;
- workload and reward identity for paired arms;
- ready-to-ready windows after warmup with stop/drain/flush;
- at least three independent deployment-level pairs;
- telemetry coverage and `N/A` for absent channels;
- no prompts, model outputs, screenshots, session IDs, absolute paths, hostnames, IPs, GPU UUIDs, or scheduler IDs.
