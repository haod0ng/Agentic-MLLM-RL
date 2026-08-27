# MobileGym with two local Judge services

This example connects [MobileGym](https://github.com/Purewhiter/mobilegym) to Relax's agentic rollout and scores
each completed trajectory with two dedicated local model services:

- `judge_accuracy`: a terminal, state-check-assisted outcome Judge;
- `judge_multiturn_vlm`: a multimodal trajectory Judge, run either once at termination or once per completed
  response-to-observation interaction.

The integration is experimental. It assumes a private, single-tenant Ray cluster and has not been validated as a
multi-tenant service. MobileGym and its data are external dependencies and are not redistributed here; check their
upstream license before use.

## Data and reward flow

```text
MobileGym task manifest
        |
        v
app/agent.py -> bench_env.run -> Relax session endpoint -> policy SGLang engines
        |                                  |
        | results.jsonl                    | recorded messages and images
        v                                  v
metadata.mobilegym_outcome_evidence -> outcome Judge
recorded trajectory                  -> process Judge
                                           |
                                           v
             score = 0.8 * answer_accuracy + 0.2 * multi_turn_reasoning
```

`app/agent.py` deliberately returns `reward=None`. MobileGym's deterministic progress and goal checks are retained
as metadata, while the two model Judges produce the training reward. The outcome projection can include observable
`field`, `expected`, and `actual` state checks. It is therefore not an independent final-answer-only ORM.

In `per_turn` mode, only interactions with a committed observation are scored. The final assistant response normally
has no following observation and is not included in the process-Judge mean. The result is still one trajectory-level
scalar; Relax does not currently create turn-level or token-level advantages from it.

## Prepare MobileGym

Use a dedicated MobileGym environment following its upstream installation guide. The adapter requires:

| Variable              | Meaning                                                     |
| --------------------- | ----------------------------------------------------------- |
| `MOBILEGYM_PYTHON`    | Python executable with `bench_env` and Playwright installed |
| `MOBILEGYM_REPO_DIR`  | MobileGym checkout containing `bench_env`                   |
| `MOBILEGYM_ENV_URL`   | MobileGym gateway reachable from rollout workers            |
| `MOBILEGYM_RUNS_ROOT` | Per-session output directory                                |
| `MOBILEGYM_AGENT`     | MobileGym agent name; defaults to `generic_v2`              |
| `MOBILEGYM_MAX_STEPS` | Maximum environment steps; defaults to `8`                  |
| `MOBILEGYM_TIMEOUT_S` | Per-episode subprocess timeout; defaults to `1200`          |

Do not expose the Relax session endpoint, Ray dashboard, Ray Serve applications, or MobileGym gateway to an
untrusted network. The session identifier is used as the local bearer credential by `bench_env.run`.

Build a deterministic task manifest:

```bash
python examples/mobilegym_agentic/scripts/build_tasks_jsonl.py \
  --mobilegym-repo /path/to/mobilegym \
  --split train \
  --repeat 4 \
  --output /path/to/mobilegym_train.jsonl
```

Each row contains `metadata.task_id` and a stable `metadata.sample_seed`. The actual task instruction remains owned by
MobileGym.

## Configure the Judges

The two example configs differ only in `reasoning_trigger`:

- `judge_services_e2e_terminal_once.json`
- `judge_services_e2e_per_turn.json`

Their public model identifiers are examples. Pin an immutable model revision or replace each `model_path` with a
local checkpoint before a reproducible run. Treat `max_input_tokens` as a text-prompt guard for a multimodal Judge;
the current proxy-side count does not include the model-specific visual-token expansion. The authoritative SGLang
context limit must still be configured and monitored.

The `dual` mode changes the reward consumed by training. For a systems-only comparison, prepare paired `recorded` or
`dual_shadow` configs using `examples/agentic_dual_judge/prepare_benchmark_configs.py`; sample-local Judge rejections
fall back to recorded reward in shadow mode, but systemic and configuration failures still fail fast.

## Launching

`run_mobilegym_e2e.sh` is a target-cluster reference recipe, not a portable one-command launcher. It requires the
standard Relax cluster entrypoint plus explicit `MODEL_DIR`, `DATA_DIR`, `SAVE_DIR`, `EXP_DIR`, and MobileGym
variables. Review its resource map and model-parallel settings against the actual node topology before use.

The `g1`-`g5`, `submit_*`, and `check_*` files are preserved as historical bring-up and validation tools. They use
environment variables for site-specific paths but still encode a four-GPU-per-node GH200 experiment design. They are
not part of the supported public quick start and should not be copied to a different cluster without review.

## Evaluation tools

- `scripts/chain_decomposition.py` closes the timestamp chain from generation through TransferQueue release. Its zero
  residual is an arithmetic closure property, not causal attribution and not full training E2E.
- `scripts/stage_breakdown.py` reports overlapping span occupancy. Inclusive percentages can sum above 100%.
- `../agentic_dual_judge/analyze_latency.py` performs fixed-window checks, paired-workload validation, and
  ready-to-ready analysis when the required timeline channels are present.
- `../agentic_dual_judge/aggregate_latency_replicates.py` aggregates independent run pairs rather than treating
  correlated publication rounds as independent samples.

The corrected evidence assessment is in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).
[LATENCY_FINDINGS.md](LATENCY_FINDINGS.md) is a sanitized summary of the historical investigation, not a raw lab
diary. No raw trajectory, screenshot, cluster log, or model output is published in this repository.

## Validation

CPU/unit checks:

```bash
pytest -q \
  tests/agentic/test_mobilegym_inputs.py \
  tests/agentic/test_latency_analyzer.py \
  tests/agentic/test_latency_replicates.py \
  tests/agentic/test_ready_marker_analyzer.py
```

The opt-in GPU test validates direct calls to two real GenRM services, but does not cover the full lifecycle from
per-turn scheduling through group replacement, TransferQueue, and trainer consumption. A complete multi-node result
therefore remains `N/A` until that end-to-end gate is run in the target environment.
