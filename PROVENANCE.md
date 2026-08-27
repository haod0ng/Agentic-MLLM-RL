# Provenance and publication scope

## Upstream base

- Project: [redai-infra/Relax](https://github.com/redai-infra/Relax)
- Base commit: `d52cd0aca9b347a57fb435bda3ae2db8fc6706a4`
- License: Apache License 2.0

The full upstream source snapshot is retained. Feature changes are kept as a separate commit so `git diff` exposes
the derivative work directly.

## Main modification areas

- `relax/agentic/`: reward scheduling, context construction, transfer provenance, lifecycle accounting, and
  critical-path events;
- `relax/engine/rewards/`: dual local-Judge execution and multimodal projections;
- `relax/components/` and `relax/distributed/ray/`: dedicated Judge services, placement, admission control, and GPU
  telemetry;
- `relax/backends/`: trainer/publication instrumentation and compatibility gates used by the measured setup;
- `examples/agentic_dual_judge/`: configuration generators and latency analysis;
- `examples/mobilegym_agentic/`: MobileGym adapter, validation stages, launch templates, and corrected evaluation
  notes;
- `tests/`: CPU/unit tests plus an opt-in real-model GPU integration test.

The exact changed-file manifest is available from:

```bash
git diff --name-status <upstream-import-commit>..<feature-commit>
```

## Excluded publication material

The public snapshot intentionally excludes:

- raw MobileGym trajectories and screenshots;
- model checkpoints and generated training data;
- cluster logs, Slurm accounting data, reservations, accounts, internal IPs, and hostnames;
- user-specific absolute paths and container locations;
- credentials and environment secrets.

Historical latency numbers in `examples/mobilegym_agentic/EVALUATION_REPORT.md` are retained only as descriptive
evidence. The raw artifacts are not included, so they must not be treated as independently reproducible results from
this repository alone.

## Validation boundary

CPU/unit and static validation can run on a normal development host. Multi-node integration and performance claims
require the target Ray/SGLang/Megatron environment, dedicated GPUs, a resettable MobileGym deployment, admission
stop/drain/flush, and complete ready-to-ready publications. A code import, successful service deployment, or partial
smoke test is not evidence of full training readiness.
