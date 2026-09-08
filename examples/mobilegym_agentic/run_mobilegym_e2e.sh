#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# MobileGym x dual-judge sync-dedicated end-to-end pipeline. The G5 target
# uses six 4-GPU GH200 nodes; legacy non-G5 smoke paths retain their original
# model configuration.
#
# Resource layout (24 GPUs in G5):
#   actor:               4 GPUs (Qwen3-VL-8B policy, TP=4)
#   rollout:              12 GPUs (Qwen3-VL-8B, 12 SGLang engines x 1 GPU)
#   judge_accuracy:       4 GPUs (Qwen3-4B, terminal-only in both variants)
#   judge_multiturn_vlm:  4 GPUs (Qwen2.5-VL-3B, terminal_once OR per_turn --
#                          selected by JUDGE_SERVICES_CONFIG below)
#   advantages:           0 GPUs (CPU)
#
# rollout_batch_size(8) x n_samples_per_prompt(8) == global_batch_size(64)
# auto-enables true_on_policy_mode (arguments.py:2893-2900), which skips
# actor_fwd; --kl-loss-coef 0.00 means no KL term, so `reference` is not
# required either (arguments.py:2905-2911). Breaking either equality without
# adding the corresponding role back causes advantages to poll TransferQueue
# for a field nobody produces -- a silent hang, not an error.

set -ex
set -o pipefail

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

###############################################################################
#                                    DIRS                                     #
###############################################################################

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/mobilegym}"
REASONING_TRIGGER="${REASONING_TRIGGER:=terminal_once}"  # terminal_once | per_turn
# Independent replicates of a latency-benchmark pair need different seeds across pairs (same
# seed within a pair) -- see examples/mobilegym_agentic/LATENCY_FINDINGS.md section 7. Was
# hardcoded to 42 for both --rollout-seed/--seed; default unchanged so existing invocations
# are unaffected.
SEED="${SEED:=42}"
ROLLOUT_SHUFFLE="${ROLLOUT_SHUFFLE:=1}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ] || [ -z "${EXP_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, SAVE_DIR, and EXP_DIR must be set."
    exit 1
fi
required_models=("${POLICY_MODEL_NAME}" "Qwen3-4B")
if [ "${G5_FULL24_ONLY}" = "1" ]; then
    required_models+=("Qwen2.5-VL-3B-Instruct")
else
    required_models+=("Qwen2.5-VL-7B-Instruct")
fi
for required_model in "${required_models[@]}"; do
    if [ ! -d "${MODEL_DIR}/${required_model}" ]; then
        echo "ERROR: required model directory is missing: ${MODEL_DIR}/${required_model}" >&2
        exit 1
    fi
done
mkdir -p "${SAVE_DIR}" "${EXP_DIR}/timeline" "${EXP_DIR}/rollout_result" "${EXP_DIR}/gpu_samples" "${EXP_DIR}/latency_markers" "${EXP_DIR}/placement" "${EXP_DIR}/env_cpu" "${EXP_DIR}/transfer_trace"
G4_FULL12_ONLY="${G4_FULL12_ONLY:-0}"
G5_FULL24_ONLY="${G5_FULL24_ONLY:-0}"
IS_SYNC_DEDICATED="${IS_SYNC_DEDICATED:-0}"
NUM_DATA_STORAGE_UNITS="${NUM_DATA_STORAGE_UNITS:-8}"
G5_TOPOLOGY_PROFILE="${G5_TOPOLOGY_PROFILE:-balanced}"
case "${NUM_DATA_STORAGE_UNITS}" in
    ''|*[!0-9]*)
        echo "ERROR: NUM_DATA_STORAGE_UNITS must be a positive integer." >&2
        exit 1
        ;;
esac
if [ "${NUM_DATA_STORAGE_UNITS}" -lt 1 ]; then
    echo "ERROR: NUM_DATA_STORAGE_UNITS must be a positive integer." >&2
    exit 1
fi
if [ "${IS_SYNC_DEDICATED}" != "0" ] && [ "${IS_SYNC_DEDICATED}" != "1" ]; then
    echo "ERROR: IS_SYNC_DEDICATED must be 0 or 1." >&2
    exit 1
fi
if [ "${G5_FULL24_ONLY}" = "1" ] && [ "${IS_SYNC_DEDICATED}" != "1" ]; then
    echo "ERROR: G5_FULL24_ONLY=1 supports only IS_SYNC_DEDICATED=1." >&2
    exit 1
fi
if [ "${G5_FULL24_ONLY}" != "1" ] && [ "${IS_SYNC_DEDICATED}" = "1" ]; then
    echo "ERROR: IS_SYNC_DEDICATED=1 is supported only by the fixed G5_FULL24 topology." >&2
    exit 1
fi
if [ "${G5_FULL24_ONLY}" = "1" ] && [ "${G5_TOPOLOGY_PROFILE}" != "balanced" ] \
    && [ "${G5_TOPOLOGY_PROFILE}" != "prm_mc48" ] \
    && [ "${G5_TOPOLOGY_PROFILE}" != "prm_tp1dp4_mc32" ]; then
    echo "ERROR: G5_TOPOLOGY_PROFILE must be balanced, prm_mc48, or prm_tp1dp4_mc32." >&2
    exit 1
fi
if [ "${G5_FULL24_ONLY}" = "1" ]; then
    source "${MODEL_CONFIG_DIR}/qwen3-vl-8B.sh"
    POLICY_MODEL_NAME="Qwen3-VL-8B-Instruct"
    EXPERIMENT_MODEL_TAG="qwen3-vl-8B-prm3B"
else
    POLICY_MODEL_NAME="Qwen3-VL-4B-Instruct"
    EXPERIMENT_MODEL_TAG="qwen3-vl-4B"
fi
EXP_NAME="${EXPERIMENT_MODEL_TAG}-mobilegym-${REASONING_TRIGGER}-${TIMESTAMP}"
export RELAX_JUDGE_GPU_SAMPLE_DIR="${EXP_DIR}/gpu_samples"
export RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S="${RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S:-0.2}"
export RELAX_DUAL_JUDGE_MARKER_DIR="${EXP_DIR}/latency_markers"
export RELAX_PLACEMENT_MANIFEST_DIR="${EXP_DIR}/placement"
export RELAX_TQ_TRACE_DIR="${EXP_DIR}/transfer_trace"
export RELAX_ENV_CPU_TRACE_DIR="${EXP_DIR}/env_cpu"
export RELAX_ENV_CPU_SAMPLE_INTERVAL_S="${RELAX_ENV_CPU_SAMPLE_INTERVAL_S:-1.0}"
if [ "${G4_FULL12_ONLY}" = "1" ] || [ "${G5_FULL24_ONLY}" = "1" ]; then
    export RELAX_REQUIRE_WEIGHT_PUBLICATION=1
fi
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CUDNN_LIB_DIR,LD_LIBRARY_PATH,FLASHINFER_WORKSPACE_BASE,RELAX_DUAL_JUDGE_MARKER_DIR,RELAX_JUDGE_GPU_SAMPLE_DIR,RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S,RELAX_PLACEMENT_MANIFEST_DIR,RELAX_TQ_TRACE_DIR,RELAX_ENV_CPU_TRACE_DIR,RELAX_ENV_CPU_SAMPLE_INTERVAL_S,RELAX_PROPAGATE_ENV_VARS,RELAX_REQUIRE_WEIGHT_PUBLICATION"

# ``ray job submit`` starts the training driver in a fresh runtime environment;
# ordinary shell exports from this launcher are not inherited.  The driver must
# see both the sampler values and the propagation allow-list so post_process_env
# can forward them into GenRMEngine's explicitly-declared runtime_env.
if [ -z "${RUNTIME_ENV_JSON:-}" ]; then
    echo "ERROR: RUNTIME_ENV_JSON must be set by the Relax cluster entrypoint." >&2
    exit 1
fi
RUNTIME_ENV_JSON="$(
    python3 -c '
import json
import os
import sys

runtime_env = json.load(sys.stdin)
env_vars = runtime_env.setdefault("env_vars", {})
for name in (
    "CUDNN_LIB_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "RELAX_DUAL_JUDGE_MARKER_DIR",
    "RELAX_JUDGE_GPU_SAMPLE_DIR",
    "RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S",
    "RELAX_PLACEMENT_MANIFEST_DIR",
    "RELAX_TQ_TRACE_DIR",
    "RELAX_ENV_CPU_TRACE_DIR",
    "RELAX_ENV_CPU_SAMPLE_INTERVAL_S",
    "RELAX_PROPAGATE_ENV_VARS",
    "RELAX_REQUIRE_WEIGHT_PUBLICATION",
):
    if name in os.environ:
        env_vars[name] = os.environ[name]
json.dump(runtime_env, sys.stdout, separators=(",", ":"))
' <<<"${RUNTIME_ENV_JSON}"
)"
export RUNTIME_ENV_JSON

###############################################################################
#                            MOBILEGYM ENV WIRING                             #
###############################################################################
#
# MOBILEGYM_ENV_URL / MOBILEGYM_PYTHON / MOBILEGYM_REPO_DIR / MOBILEGYM_RUNS_ROOT
# must be set by the caller (see the sbatch submission script). MobileGym's
# own bench_env.run owns the whole multi-turn agent<->env loop once pointed
# at RELAX_BASE_URL -- see examples/mobilegym_agentic/app/agent.py.

if [ -z "${MOBILEGYM_ENV_URL:-}" ] || [ -z "${MOBILEGYM_PYTHON:-}" ] || [ -z "${MOBILEGYM_REPO_DIR:-}" ]; then
    echo "ERROR: MOBILEGYM_ENV_URL, MOBILEGYM_PYTHON, and MOBILEGYM_REPO_DIR must be set."
    exit 1
fi
MOBILEGYM_RUNS_ROOT="${MOBILEGYM_RUNS_ROOT:=${EXP_DIR}/mobilegym_runs}"
mkdir -p "${MOBILEGYM_RUNS_ROOT}"

###############################################################################
#                              JUDGE MODEL CONFIG                             #
###############################################################################
#
# DEBUG_ROLLOUT_ONLY=1 selects the 1-GPU-per-judge config so a single 4-GPU
# node (rollout[1,2] + judge_accuracy[1,1] + judge_multiturn_vlm[1,1]) can
# smoke-test env reachability, agent spawn, judge HTTP calls, and
# RewardContext construction before committing to the full 4-node/16-GPU
# allocation -- see L1 in examples/mobilegym_agentic/README.md.

DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
G3_DUAL8_ONLY="${G3_DUAL8_ONLY:-0}"
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ] && [ "${G3_DUAL8_ONLY}" != "1" ]; then
    # Must still honour REASONING_TRIGGER: a fixed debug config would silently
    # judge terminal_once while the caller believed it was measuring per_turn.
    JUDGE_SERVICES_CONFIG="${SCRIPT_DIR}/judge_services_e2e_debug_${REASONING_TRIGGER}.json"
elif [ "${G5_FULL24_ONLY}" = "1" ]; then
    # G5 uses a balanced 4/12/4/4 resource split. The PRM reserves all four
    # GPUs but runs one SGLang endpoint internally as TP2 x DP2.
    if [ "${G5_TOPOLOGY_PROFILE}" = "prm_mc48" ]; then
        JUDGE_SERVICES_CONFIG="${SCRIPT_DIR}/judge_services_e2e_g5_qwen3vl8_prm3b_${REASONING_TRIGGER}_tp1dp4_mc48.json"
    elif [ "${G5_TOPOLOGY_PROFILE}" = "prm_tp1dp4_mc32" ]; then
        JUDGE_SERVICES_CONFIG="${SCRIPT_DIR}/judge_services_e2e_g5_qwen3vl8_prm3b_${REASONING_TRIGGER}_tp1dp4_mc32.json"
    else
        JUDGE_SERVICES_CONFIG="${SCRIPT_DIR}/judge_services_e2e_g5_qwen3vl8_prm3b_${REASONING_TRIGGER}.json"
    fi
else
    JUDGE_SERVICES_CONFIG="${SCRIPT_DIR}/judge_services_e2e_${REASONING_TRIGGER}.json"
fi

# Escape hatch for latency A/B arms that need a config identical to the auto-selected one
# except for a specific field (e.g. max_concurrency) -- see
# examples/mobilegym_agentic/LATENCY_FINDINGS.md section 7. Still gated by the same
# REASONING_TRIGGER-mismatch guard as the rest of this block: the caller is responsible for
# generating the override file from the matching REASONING_TRIGGER base config.
if [ -n "${JUDGE_SERVICES_CONFIG_OVERRIDE:-}" ]; then
    JUDGE_SERVICES_CONFIG="${JUDGE_SERVICES_CONFIG_OVERRIDE}"
fi
if [ -n "${JUDGE_SERVICES_CONFIG}" ] && [ ! -f "${JUDGE_SERVICES_CONFIG}" ]; then
    echo "ERROR: no judge config at ${JUDGE_SERVICES_CONFIG}"
    exit 1
fi
# Expand ${USER} and other deployment-local variables once, after applying the
# override, before passing the JSON through argparse. This keeps profile files
# portable while making effective model paths explicit in the run's arguments.
JUDGE_SERVICES_JSON="$(JUDGE_SERVICES_CONFIG="${JUDGE_SERVICES_CONFIG}" python3 - <<'PY'
import json
import os
import sys


def expand(value):
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


with open(os.environ["JUDGE_SERVICES_CONFIG"], encoding="utf-8") as handle:
    json.dump(expand(json.load(handle)), sys.stdout, separators=(",", ":"))
PY
)"

###############################################################################
#                                  MODEL CONFIG                               #
###############################################################################

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${POLICY_MODEL_NAME}"
    --megatron-to-hf-mode bridge
    --save "${SAVE_DIR}/${POLICY_MODEL_NAME}-MobileGym-Checkpoint"
    --save-interval 100
    --max-actor-ckpt-to-keep 1
)
if [ "${G4_FULL12_ONLY}" = "1" ] || [ "${G5_FULL24_ONLY}" = "1" ]; then
    CKPT_ARGS=(
        --hf-checkpoint "${MODEL_DIR}/${POLICY_MODEL_NAME}"
        --megatron-to-hf-mode bridge
        --save "${EXP_DIR}/checkpoint"
        --save-interval 1
        --max-actor-ckpt-to-keep 1
    )
fi
if [ "${IS_SYNC_DEDICATED}" = "1" ]; then
    # Checkpoint I/O would otherwise be charged to the weight-update stage;
    # sync-dedicated timing runs intentionally isolate serving/update latency.
    CKPT_ARGS=(
        --hf-checkpoint "${MODEL_DIR}/${POLICY_MODEL_NAME}"
        --megatron-to-hf-mode bridge
    )
fi

###############################################################################
#                                  DATASET                                    #
###############################################################################

TRAIN_FILE="${DATA_DIR}/mobilegym_train.jsonl"
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: ${TRAIN_FILE} not found. Generate it first:" >&2
    echo "  ${MOBILEGYM_PYTHON} ${SCRIPT_DIR}/scripts/build_tasks_jsonl.py \\" >&2
    echo "    --mobilegym-repo ${MOBILEGYM_REPO_DIR} --split train --repeat 4 --output ${TRAIN_FILE}" >&2
    exit 1
fi

###############################################################################
#                               ROLLOUT CONFIG                                #
###############################################################################

if [ "${G4_FULL12_ONLY}" = "1" ]; then
    NUM_ROLLOUT="${NUM_ROLLOUT:=2}"
    if [ "${NUM_ROLLOUT}" -ne 2 ]; then
        echo "ERROR: G4_FULL12_ONLY=1 requires exactly two publication rounds." >&2
        exit 1
    fi
else
    NUM_ROLLOUT="${NUM_ROLLOUT:=3}"  # Phase 0 E2E smoke default; measurement runs override this.
fi
if [ "${G5_FULL24_ONLY}" = "1" ] && [ "${NUM_ROLLOUT}" -lt 3 ]; then
    echo "ERROR: G5_FULL24_ONLY=1 requires at least three rounds (one warmup plus at least two measured)." >&2
    exit 1
fi

# DEBUG_ROLLOUT_ONLY uses a much smaller batch (rollout[1,2] can only run a
# couple of concurrent SGLang requests usefully) -- this tier is exercising
# the MobileGym<->judge plumbing, not throughput.
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ] || [ "${G3_DUAL8_ONLY}" = "1" ] || [ "${G4_FULL12_ONLY}" = "1" ]; then
    ROLLOUT_BATCH_SIZE=2
    N_SAMPLES_PER_PROMPT=2
    GLOBAL_BATCH_SIZE=4
else
    ROLLOUT_BATCH_SIZE=8
    N_SAMPLES_PER_PROMPT=8
    GLOBAL_BATCH_SIZE=64
fi

ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
# Qwen3-VL can emit a long reasoning/tool-call turn even with thinking
# disabled.  The former hard-coded 512-token cap caused committed MobileGym
# samples to be marked truncated and silently routed to terminal fallback,
# contaminating the per-turn PRM measurement.  Keep this configurable so a
# benchmark can trade clean turn coverage against decode latency; 1024 is the
# conservative G5 candidate default and must be checked by the strict result
# checker on the next full run.
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_FILE}"
    --input-key input
    --metadata-key metadata
    --multimodal-keys '{"image":"images"}'
    --use-agentic-rollout
    --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
    --agent-cwd "${SCRIPT_DIR}"
    --agent-env \
        "MOBILEGYM_ENV_URL=${MOBILEGYM_ENV_URL}" \
        "MOBILEGYM_PYTHON=${MOBILEGYM_PYTHON}" \
        "MOBILEGYM_REPO_DIR=${MOBILEGYM_REPO_DIR}" \
        "MOBILEGYM_RUNS_ROOT=${MOBILEGYM_RUNS_ROOT}" \
        "MOBILEGYM_AGENT=${MOBILEGYM_AGENT:=generic_v2}" \
        "MOBILEGYM_MAX_STEPS=${MOBILEGYM_MAX_STEPS:=8}" \
        "MOBILEGYM_TIMEOUT_S=${MOBILEGYM_TIMEOUT_S:=1200}" \
        "MOBILEGYM_HISTORY_IMAGES=${MOBILEGYM_HISTORY_IMAGES:=1}" \
        "BROWSER_LD_LIBRARY_PATH=${BROWSER_LD_LIBRARY_PATH:-}"
    --agent-timeout 1800
    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
    --rollout-max-prompt-len 16384
    --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
    --rollout-max-context-len 32768
    --rollout-temperature ${ROLLOUT_TEMPERATURE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --rollout-seed ${SEED}
    --seed ${SEED}
    --reward-key score
)
case "${ROLLOUT_SHUFFLE}" in
    0) ;;
    1) ROLLOUT_ARGS+=(--rollout-shuffle) ;;
    *) echo "ERROR: ROLLOUT_SHUFFLE must be 0 or 1." >&2; exit 1 ;;
esac
ROLLOUT_ARGS+=(
    --rm-type dual-agentic-judge
    --judge-services-config "${JUDGE_SERVICES_JSON}"
)
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ] || [ "${G3_DUAL8_ONLY}" = "1" ]; then
    mkdir -p "${EXP_DIR}/debug_rollout"
    ROLLOUT_ARGS+=(
        --debug-rollout-only
        --save-debug-rollout-data "${EXP_DIR}/debug_rollout/{rollout_id}.pt"
    )
fi

###############################################################################
#                              ALGORITHM CONFIG                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
)

###############################################################################
#                              OPTIMIZER CONFIG                               #
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --rollout-engine-init-timeout 300
    --sglang-mem-fraction-static 0.6
)
ROLLOUT_ROUTER_MODE="${ROLLOUT_ROUTER_MODE:-round_robin}"
if [ "${G3_DUAL8_ONLY}" = "1" ] || [ "${G4_FULL12_ONLY}" = "1" ] || [ "${G5_FULL24_ONLY}" = "1" ]; then
    case "${ROLLOUT_ROUTER_MODE}" in
        round_robin)
            SGLANG_ARGS+=(--sglang-router-policy round_robin)
            ;;
        sticky)
            SGLANG_ARGS+=(
                --use-slime-router
                --slime-router-sticky
                --slime-router-sticky-idle-secs 1800
            )
            ;;
        *)
            echo "ERROR: ROLLOUT_ROUTER_MODE must be round_robin or sticky." >&2
            exit 1
            ;;
    esac
fi
echo "ROLLOUT_ROUTER_MODE=${ROLLOUT_ROUTER_MODE}"

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --use-metrics-service
    --timeline-dump-dir "${EXP_DIR}/timeline"
    --rollout-result-dir "${EXP_DIR}/rollout_result"
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################
#
# --transformer-impl local avoids a hard dependency on transformer_engine /
# apex (neither is imported by relax/backends/megatron/ directly -- see
# examples/mobilegym_agentic/README.md "Container" section). No forced
# --attention-backend: Megatron-Core's default (auto) falls back to unfused
# attention when flash-attn is unavailable rather than crashing, which is
# what this container has for Phase 0.

MEGATRON_ARGS=(
    --transformer-impl local
    # Qwen3-VL is multimodal, so RoPE is multi-axis (a list of tensors) and the
    # fused RoPE kernels do not apply; leaving fusion on makes training and
    # inference disagree on log-probs. _hf_validate_args (see
    # relax/backends/megatron/arguments.py:261) rejects the combination outright.
    # --debug-rollout-only sets skip_hf_validate, which is why the L1 tier never
    # hit this and the first training-enabled run did.
    --no-rope-fusion
    # gradient_accumulation_fusion defaults to True and is NOT auto-disabled by
    # --transformer-impl local (megatron/training/arguments.py:2620). Relax copies
    # it onto the bridge provider (model_provider.py:301), and ColumnParallelLinear
    # then hard-fails on the missing apex fused_weight_gradient_mlp_cuda extension,
    # which this container deliberately does not build.
    --no-gradient-accumulation-fusion
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --distributed-timeout-minutes 5
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --warm-hf-checkpoint-page-cache
    --use-dynamic-batch-size
    # arguments.py:2950 requires max_tokens_per_gpu * context_parallel_size >=
    # rollout_max_context_len, so that one over-long sample cannot form an
    # oversized micro-batch and OOM. rollout_max_context_len is 32768 above
    # (MobileGym trajectories carry screenshots, so contexts run long), and
    # context_parallel_size is 1, so this must be at least 32768.
    --max-tokens-per-gpu 32768
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
)

###############################################################################
#                              RESOURCE CONFIG                                #
###############################################################################

# debug_rollout_only short-circuits relax/core/registry.py:process_role() to
# ROLES_ROLLOUT_ONLY (just `rollout`) before fully_async is even consulted --
# no actor/advantages role is created, so they must not appear in --resource.
# Judges are unaffected: register_dual_judges() (relax/core/optional_roles.py)
# gates only on judge_services being set, independent of debug_rollout_only,
# so L1 still exercises real judge HTTP calls / RewardContext construction.
if [ "${G5_FULL24_ONLY}" = "1" ]; then
    # Balanced split: single-node 4-GPU actor (avoids the cross-node DP=2
    # actor topology that crashed the earlier 8-GPU-actor G5 layout with a
    # SIGSEGV inside train_one_step -- see g5-terminal-r5/r6/r7 slurm logs),
    # 4-GPU TP judge engines so ORM (Qwen3-4B) and PRM (Qwen2.5-VL) get equal
    # footprint regardless of per-trajectory call count (terminal-once vs
    # per-turn).
    RESOURCE_JSON='{"actor":[1,4],"rollout":[1,12],"judge_accuracy":[1,4],"judge_multiturn_vlm":[1,4]}'
    G5_EXPECTED_ROLLOUT_GPUS=12
    G5_EXPECTED_ORM_GPUS=4
    G5_EXPECTED_PRM_GPUS=4
elif [ "${G3_DUAL8_ONLY}" = "1" ]; then
    RESOURCE_JSON='{"rollout":[1,4],"judge_accuracy":[1,2],"judge_multiturn_vlm":[1,2]}'
elif [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
    RESOURCE_JSON='{"rollout":[1,2],"judge_accuracy":[1,1],"judge_multiturn_vlm":[1,1]}'
else
    # The second element is a TOTAL gpu count (see backends/megatron/arguments.py:325
    # `_, rollout_total_gpus = args.resource["rollout"]`). rollout must stay within
    # one node's worth of GPUs on this cluster: with 8 the engines are spread over
    # two GH200 nodes, and _allocate_rollout_engine_addr_and_ports_normal
    # (distributed/ray/rollout.py:3563) derives each block's dist_init_addr from the
    # *first* engine of that block, assuming Ray placed ranks in contiguous per-node
    # blocks. When that assumption does not hold, engines on one node are handed the
    # other node's rendezvous address and hang until the 600s TCPStore timeout.
    # rollout:[1,8] was carried over from an 8-GPU-per-node recipe
    # (examples/nemo_gym_agentic/recipes/r2e-gym/run-qwen35-9B-8xgpu-nemo-gym-r2e.sh);
    # GH200 nodes here have 4.
    RESOURCE_JSON='{"actor":[1,4],"rollout":[1,4],"advantages":[1,0],"judge_accuracy":[1,2],"judge_multiturn_vlm":[1,2]}'
fi

if [ "${G5_FULL24_ONLY}" = "1" ]; then
    # G5 is intentionally sync-dedicated only. Hash the same partition across
    # eight native TQ storage actors without changing batch/sampler semantics.
    RAY_RESOURCE_ARGS=(
        --resource "${RESOURCE_JSON}"
        --num-gpus-per-node 4
        --actor-num-nodes 1
        --actor-num-gpus-per-node 4
        --max-staleness 0
        --num-data-storage-units "${NUM_DATA_STORAGE_UNITS}"
        --no-offload-train
        --no-offload-rollout
        --is-sync-dedicated
        --weight-version-validation-timeout-s "${WEIGHT_VERSION_VALIDATION_TIMEOUT_S:-30}"
    )
else
    RAY_RESOURCE_ARGS=(
        --resource "${RESOURCE_JSON}"
        # GH200 allocations have four GPUs per node. Rollout rendezvous groups
        # are derived from this value; leaving the upstream default of eight
        # can assign a node-B engine a node-A rendezvous address.
        --num-gpus-per-node 4
        --actor-num-nodes 1
        --actor-num-gpus-per-node 4
        --max-staleness 1
        --num-data-storage-units 1
        --fully-async
    )
fi
if [ "${G3_DUAL8_ONLY}" != "1" ] && [ "${G4_FULL12_ONLY}" != "1" ] && [ "${G5_FULL24_ONLY}" != "1" ]; then
    RAY_RESOURCE_ARGS+=(--use-health-check)
fi

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

mkdir -p logs

if [ "${G3_DUAL8_ONLY}" = "1" ]; then
    DRIVER_LOG="${EXP_DIR}/g3_dual_driver.log"
elif [ "${G4_FULL12_ONLY}" = "1" ]; then
    DRIVER_LOG="${EXP_DIR}/g4_full12_driver.log"
elif [ "${G5_FULL24_ONLY}" = "1" ]; then
    DRIVER_LOG="${EXP_DIR}/g5_full24_driver.log"
else
    DRIVER_LOG="logs/${EXP_NAME}.log"
fi

# Ray's log-stream endpoint can close while the job is still RUNNING (the
# previous six-node run returned here immediately after rollout 0). An
# explicit submission id lets us distinguish that transport-level EOF from a
# terminal job state and keep the SPMD allocation alive until training and
# weight publication have actually finished.
RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-relax-${SLURM_JOB_ID:-manual}-${REASONING_TRIGGER}}"
set +e
ray job submit --submission-id "${RAY_JOB_SUBMISSION_ID}" ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    2>&1 | tee "${DRIVER_LOG}"
submit_rc=${PIPESTATUS[0]}
set -e
if [ "${submit_rc}" -ne 0 ]; then
    echo "ERROR: Ray job ${RAY_JOB_SUBMISSION_ID} submission/log stream failed with exit code ${submit_rc}." >&2
    exit "${submit_rc}"
fi

# The CLI normally waits for the job, but its log stream is not a reliable
# lifecycle signal under the large multimodal output volume. Poll the Job
# Submission API explicitly whenever the stream returned before a terminal
# state. This loop is bounded by the Slurm allocation and fails closed on a
# real FAILED/STOPPED state, so analyzers never consume a partial trajectory.
job_wait_timeout_s="${RAY_JOB_COMPLETION_TIMEOUT_S:-3600}"
job_wait_deadline=$((SECONDS + job_wait_timeout_s))
while true; do
    if ! job_status_output="$(ray job status "${RAY_JOB_SUBMISSION_ID}" 2>&1)"; then
        printf '%s\n' "${job_status_output}" | tee -a "${DRIVER_LOG}"
        if [ "${SECONDS}" -ge "${job_wait_deadline}" ]; then
            echo "ERROR: unable to query Ray job ${RAY_JOB_SUBMISSION_ID} before the ${job_wait_timeout_s}s deadline." >&2
            exit 1
        fi
        sleep 5
        continue
    fi
    printf '%s\n' "${job_status_output}" | tee -a "${DRIVER_LOG}"
    # Ray CLI emits either ``Status for job ...: SUCCEEDED`` or the terminal
    # summary ``Job '...' succeeded`` depending on the installed Ray version.
    # Treat both forms as lifecycle signals; otherwise a completed job leaves
    # the SPMD wrapper polling until Slurm kills an otherwise successful run.
    if printf '%s\n' "${job_status_output}" | grep -Eiq \
        "Status[^:]*: (SUCCEEDED|STOPPED|FAILED)|Job .* (succeeded|stopped|failed)"; then
        if printf '%s\n' "${job_status_output}" | grep -Eiq \
            "Status[^:]*: (STOPPED|FAILED)|Job .* (stopped|failed)"; then
            echo "ERROR: Ray job ${RAY_JOB_SUBMISSION_ID} reached a failure terminal state." >&2
            exit 1
        fi
        break
    fi
    if [ "${SECONDS}" -ge "${job_wait_deadline}" ]; then
        echo "ERROR: Ray job ${RAY_JOB_SUBMISSION_ID} did not reach a terminal state within ${job_wait_timeout_s}s." >&2
        exit 1
    fi
    sleep 5
done

if [ "${G3_DUAL8_ONLY}" = "1" ]; then
    python3 "${SCRIPT_DIR}/../agentic_dual_judge/analyze_latency.py" \
        --variant "${REASONING_TRIGGER}=${EXP_DIR}/rollout_result" \
        --direct \
        --gpu-samples "${REASONING_TRIGGER}=${EXP_DIR}/gpu_samples" \
        --expected-mode "${REASONING_TRIGGER}=dual" \
        --expected-reasoning-trigger "${REASONING_TRIGGER}=${REASONING_TRIGGER}" \
        --expected-groups-per-round 2 \
        --expected-samples-per-group 2 \
        --output "${EXP_DIR}/direct_report.json"
    python3 "${SCRIPT_DIR}/scripts/check_g3_dual8.py" \
        --exp-dir "${EXP_DIR}" \
        --trigger "${REASONING_TRIGGER}"
fi
if [ "${G4_FULL12_ONLY}" = "1" ]; then
    if [ -z "${RELAX_MULTI_HOST_CLOCK_MAX_OFFSET_MS:-}" ]; then
        echo "ERROR: G4 full12 requires an audited multi-host clock offset bound." >&2
        exit 1
    fi
    python3 "${SCRIPT_DIR}/../agentic_dual_judge/analyze_latency.py" \
        --variant "${REASONING_TRIGGER}=${EXP_DIR}" \
        --direct \
        --ready-markers "${REASONING_TRIGGER}=${EXP_DIR}/latency_markers/weight_serving_ready.jsonl" \
        --gpu-samples "${REASONING_TRIGGER}=${EXP_DIR}/gpu_samples" \
        --expected-mode "${REASONING_TRIGGER}=dual" \
        --expected-reasoning-trigger "${REASONING_TRIGGER}=${REASONING_TRIGGER}" \
        --expected-groups-per-round 2 \
        --expected-samples-per-group 2 \
        --warmup-steps 1 \
        --measure-updates 1 \
        --allow-synchronized-multi-host-clock \
        --multi-host-clock-max-offset-ms "${RELAX_MULTI_HOST_CLOCK_MAX_OFFSET_MS}" \
        --output "${EXP_DIR}/direct_report.json"
    python3 "${SCRIPT_DIR}/scripts/check_g4_full12.py" \
        --exp-dir "${EXP_DIR}" \
        --trigger "${REASONING_TRIGGER}" \
        --expected-steps "${NUM_ROLLOUT}" \
        --max-clock-offset-ms "${RELAX_G4_MAX_CLOCK_OFFSET_MS:-10}"
fi
if [ "${G5_FULL24_ONLY}" = "1" ]; then
    if [ -z "${RELAX_MULTI_HOST_CLOCK_MAX_OFFSET_MS:-}" ]; then
        echo "ERROR: G5 full24 requires an audited multi-host clock offset bound." >&2
        exit 1
    fi
    python3 "${SCRIPT_DIR}/../agentic_dual_judge/analyze_latency.py" \
        --variant "${REASONING_TRIGGER}=${EXP_DIR}" \
        --direct \
        --ready-markers "${REASONING_TRIGGER}=${EXP_DIR}/latency_markers/weight_serving_ready.jsonl" \
        --gpu-samples "${REASONING_TRIGGER}=${EXP_DIR}/gpu_samples" \
        --expected-mode "${REASONING_TRIGGER}=dual" \
        --expected-reasoning-trigger "${REASONING_TRIGGER}=${REASONING_TRIGGER}" \
        --expected-groups-per-round 8 \
        --expected-samples-per-group 8 \
        --warmup-steps 1 \
        --measure-updates "$((NUM_ROLLOUT - 1))" \
        --allow-synchronized-multi-host-clock \
        --multi-host-clock-max-offset-ms "${RELAX_MULTI_HOST_CLOCK_MAX_OFFSET_MS}" \
        --output "${EXP_DIR}/direct_report.json"
    python3 "${SCRIPT_DIR}/scripts/check_g5_full24.py" \
        --exp-dir "${EXP_DIR}" \
        --trigger "${REASONING_TRIGGER}" \
        --expected-steps "${NUM_ROLLOUT}" \
        --expected-storage-units "${NUM_DATA_STORAGE_UNITS}" \
        --expected-rollout-gpus "${G5_EXPECTED_ROLLOUT_GPUS}" \
        --expected-orm-gpus "${G5_EXPECTED_ORM_GPUS}" \
        --expected-prm-gpus "${G5_EXPECTED_PRM_GPUS}" \
        --max-clock-offset-ms "${RELAX_G5_MAX_CLOCK_OFFSET_MS:-10}"
fi
