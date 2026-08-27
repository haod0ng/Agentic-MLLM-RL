# Agentic MLLM RL on Relax

This repository is a research snapshot of [Relax](https://github.com/redai-infra/Relax) extended for agentic
multimodal reinforcement learning with two dedicated local reward-model services:

- an outcome Judge (`answer_accuracy`), and
- a multimodal process/trajectory Judge (`multi_turn_reasoning`).

The two services are independent Ray Serve deployments and can be placed on dedicated GPU groups. The process Judge
supports two execution modes:

- `terminal_once`: score the complete trajectory once after the episode;
- `per_turn`: score each completed response-to-observation interaction asynchronously, then average the interaction
  scores into one trajectory-level reward component.

`per_turn` is a scheduling and scoring-granularity variant. It does **not** currently produce turn-level or token-level
advantages, and the final assistant response has no following observation, so it is covered by the outcome Judge but
not by the per-turn process Judge.

## What is included

| Area                                                         | Location                                                        |
| ------------------------------------------------------------ | --------------------------------------------------------------- |
| Dual local-Judge runtime and projections                     | `relax/engine/rewards/`, `relax/agentic/session/`               |
| Ray Serve deployment, placement, and concurrency integration | `relax/components/`, `relax/distributed/ray/`                   |
| TransferQueue provenance and critical-path instrumentation   | `relax/agentic/`, `relax/utils/data/`, `relax/utils/metrics/`   |
| Reusable dual-Judge configs and latency analyzers            | `examples/agentic_dual_judge/`                                  |
| MobileGym adapter, launch templates, and evaluation tools    | `examples/mobilegym_agentic/`                                   |
| CPU/unit and opt-in GPU tests                                | `tests/agentic/`, `tests/engine/rewards/`, `tests/integration/` |

The complete Relax source tree is retained so the extension can be inspected and tested in its real integration
context. See [PROVENANCE.md](PROVENANCE.md) for the upstream base and publication boundary.

## Reward semantics

For `benchmark_mode=dual`, the training reward is:

```text
score = 0.8 * answer_accuracy + 0.2 * multi_turn_reasoning
```

The MobileGym adapter deliberately returns no environment reward, allowing the dual local Judges to become the
training reward source. Its outcome projection may include terminal state-check evidence (`field`, `expected`, and
`actual`). This should be described as a **state-check-assisted outcome Judge**, not as a final-answer-only Judge. If
the research question requires an independent learned ORM, ablate or remove those fields.

For systems-only latency comparisons, use `recorded` or `dual_shadow` modes so training consumes the same reward in
both arms. Comparing online `dual + terminal_once` against `dual + per_turn` changes both reward semantics and future
policy trajectories; that comparison is a joint operational effect, not a pure scheduling effect.

## Quick start

Install Relax as documented by the upstream project:

```bash
python -m pip install -r requirements.txt
python -m pip install \
  "transferqueue @ git+https://github.com/redai-infra/TransferQueue.git@58054a33834aadbcf76aacd6b1e32e25c030f2c9" \
  --no-deps
python -m pip install -e .
```

TransferQueue is a required public Git dependency but is not listed in the inherited `requirements.txt`. The command
above matches the compatibility check in `relax/core/controller.py`; use the project container for the complete
Ray/SGLang/Megatron runtime.

Run the CPU/unit validation for the extension:

```bash
pytest -q \
  tests/engine/rewards/test_dual_local_judge.py \
  tests/engine/rewards/test_reward_projection.py \
  tests/utils/test_judge_config.py \
  tests/agentic/session/test_reward_context.py \
  tests/agentic/pipeline/test_reward_dual_judge.py \
  tests/agentic/test_latency_analyzer.py
```

GPU integration tests are opt-in and require the target Ray/SGLang runtime and local model checkpoints:

```bash
pytest -q -m gpu tests/integration/test_agentic_dual_judge_real_models.py
```

Start with [the dual-Judge guide](examples/agentic_dual_judge/README.md). The
[MobileGym guide](examples/mobilegym_agentic/README.md) documents the external environment contract and cluster
variables without assuming a particular filesystem, account, or container image.

## Evaluation status

The included historical latency results are a diagnostic case study, not a causal performance claim. They support
the observations that the terminal process-Judge path had long client-side queueing, the reward/transfer path
coincided with substantial trainer data-wait, and historical `per_turn` smoke runs were faster. They do not prove:

- that reward was the unique binding bottleneck;
- a causal percentage attribution of end-to-end latency;
- the source of Judge GPU idle time;
- that increasing `max_concurrency` improves throughput; or
- the magnitude of a `per_turn` effect under replicated, frozen workloads.

See [the evaluation report](examples/mobilegym_agentic/EVALUATION_REPORT.md) for the corrected evidence levels and
required follow-up design. Raw trajectories, screenshots, checkpoints, and cluster logs are not included.

## License and attribution

This repository is distributed under the Apache License 2.0. It is derived from Relax commit
`d52cd0aca9b347a57fb435bda3ae2db8fc6706a4`. Original Relax notices are retained; see [NOTICE](NOTICE) and
[PROVENANCE.md](PROVENANCE.md) for modification and attribution details.
