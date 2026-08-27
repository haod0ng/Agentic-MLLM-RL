#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Slurm/EDF reference submission for the MobileGym dual-Judge pipeline.
# The historical topology used four GPUs per node. Site account, reservation,
# partition, image, filesystem, and gateway settings must be supplied by the
# caller; this file deliberately contains no cluster-specific identifiers.
#
# Usage (run from the SAME login node that is running the nginx gateway --
# MOBILEGYM_ENV_URL=https://$(hostname):4180 captures that node's hostname at
# submit time; compute nodes cannot reach the login node via "localhost", but
# its externally supplied address must be routable from every compute node):
#   sbatch --time=00:30:00 -p debug --nodes=1 \
#     --export=ALL,NUM_ROLLOUT=1,REASONING_TRIGGER=terminal_once,DEBUG_ROLLOUT_ONLY=1,MOBILEGYM_ENV_URL=https://$(hostname):4180 \
#     examples/mobilegym_agentic/submit_mobilegym_e2e.sh                 # L1 smoke, 1 node
#     # --nodes=1 (CLI) overrides the in-script `#SBATCH --nodes=4` default;
#     # DEBUG_ROLLOUT_ONLY=1 makes run_mobilegym_e2e.sh pass --debug-rollout-only
#     # and shrink --resource to rollout[1,2]+judge_accuracy[1,1]+judge_multiturn_vlm[1,1]
#     # (4 GPUs total, no actor/advantages role -- see run_mobilegym_e2e.sh).
#
#   sbatch --time=02:00:00 -p normal \
#     --export=ALL,REASONING_TRIGGER=terminal_once,NUM_ROLLOUT=3,MOBILEGYM_ENV_URL=https://$(hostname):4180 \
#     examples/mobilegym_agentic/submit_mobilegym_e2e.sh                 # L2, full 16 GPUs
#
#   sbatch --time=02:00:00 -p normal \
#     --export=ALL,REASONING_TRIGGER=per_turn,NUM_ROLLOUT=3,MOBILEGYM_ENV_URL=https://$(hostname):4180 \
#     examples/mobilegym_agentic/submit_mobilegym_e2e.sh                 # L3, full 16 GPUs

#SBATCH --job-name=mobilegym-e2e
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

echo "START TIME: $(date)"
echo "JOBID=$SLURM_JOB_ID NODELIST=$SLURM_NODELIST NNODES=$SLURM_JOB_NUM_NODES"

# Derived from the actual allocation, not hardcoded: spmd-multinode.sh's head
# node blocks until it sees exactly WORLD_SIZE devices join (device_count -eq
# NNODES, spmd-multinode.sh:126) -- a mismatch against the real node count
# (e.g. a 1-node L1 smoke request against a hardcoded WORLD_SIZE=4) hangs
# until the Slurm time limit kills the job rather than failing fast.
NUM_NODES="${SLURM_JOB_NUM_NODES}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"  # GH200 nodes: 4 GPUs/node (see #SBATCH --gpus-per-node above)

###############################################################################
#                                  KNOBS                                      #
###############################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_REPO_DIR="${RELAX_REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-${SCRATCH:-${TMPDIR:-/tmp}}/mobilegym_e2e}"
RELAX_ENV_ROOT="${RELAX_ENV_ROOT:-${WORK_ROOT}/g2_relax_env_te214_sm90_cuda13_v2}"
EDF_TOML="${EDF_TOML:-}"
if [ -z "${EDF_TOML}" ] || [ ! -f "${EDF_TOML}" ]; then
    echo "ERROR: EDF_TOML must name a readable container definition." >&2
    exit 2
fi
# G2-A deliberately starts only the production TP2 ORM and TP2 VLM services.
# It has no MobileGym browser/endpoint dependency and must occupy exactly one
# four-GPU node. Normal training remains the default path.
G2_JUDGE_TOPOLOGY_ONLY="${G2_JUDGE_TOPOLOGY_ONLY:-0}"
G3_ROLLOUT8_ONLY="${G3_ROLLOUT8_ONLY:-0}"
G3_ACTOR8_ONLY="${G3_ACTOR8_ONLY:-0}"
G3_DUAL8_ONLY="${G3_DUAL8_ONLY:-0}"
G4_ROLLOUT12_ONLY="${G4_ROLLOUT12_ONLY:-0}"
G4_FULL12_ONLY="${G4_FULL12_ONLY:-0}"
G5_FULL24_ONLY="${G5_FULL24_ONLY:-0}"
if [ -z "${RUN_SCRIPT:-}" ]; then
    if [ "${G2_JUDGE_TOPOLOGY_ONLY}" = "1" ]; then
        RUN_SCRIPT="examples/mobilegym_agentic/run_g2_judge_tp2_smoke.sh"
    elif [ "${G3_ROLLOUT8_ONLY}" = "1" ] || [ "${G4_ROLLOUT12_ONLY}" = "1" ]; then
        RUN_SCRIPT="examples/mobilegym_agentic/run_g3_rollout8_smoke.sh"
    elif [ "${G3_ACTOR8_ONLY}" = "1" ]; then
        RUN_SCRIPT="examples/mobilegym_agentic/run_g3_actor_tp4_dp2_smoke.sh"
    else
        RUN_SCRIPT="examples/mobilegym_agentic/run_mobilegym_e2e.sh"
    fi
fi
if [ "${G2_JUDGE_TOPOLOGY_ONLY}" = "1" ] && { [ "${NUM_NODES}" -ne 1 ] || [ "${GPUS_PER_NODE}" -ne 4 ]; }; then
    echo "ERROR: G2_JUDGE_TOPOLOGY_ONLY=1 requires exactly one node with four GPUs." >&2
    exit 1
fi

###############################################################################
#                    HOST LIBRARIES + FONTS FOR THE CONTAINER                 #
###############################################################################
# The EDF SGLang image is a compute-only image: it has no libnuma (sgl_kernel
# fails to load without it), none of the GLib/X11/GBM/NSS libraries Playwright's
# headless Chromium needs to launch, and neither a fontconfig config nor any
# fonts. That last one is not cosmetic: with no fontconfig, Chromium aborts
# itself the moment a page renders real text
# ("FATAL:...SkFontMgr_FontConfigInterface.cpp:163 Not implemented" -> SIGTRAP),
# which happens as soon as a MobileGym app opens. A CJK font is required too --
# the host ships Latin-only fonts while MobileGym's UI is Chinese, and the
# policy model reads these screenshots.
HOST_LIB_DIR="${HOST_LIB_DIR:-/usr/lib64}"
HOST_FONTCONFIG_DIR="${HOST_FONTCONFIG_DIR:-/etc/fonts}"
HOST_FONTS_DIR="${HOST_FONTS_DIR:-/usr/share/fonts}"
EXTRA_FONTS_DIR="${EXTRA_FONTS_DIR:-${WORK_ROOT}/fonts}"
required_host_paths=("${HOST_LIB_DIR}")
if [ "${G2_JUDGE_TOPOLOGY_ONLY}" != "1" ] && [ "${G3_ROLLOUT8_ONLY}" != "1" ] && [ "${G4_ROLLOUT12_ONLY}" != "1" ] && [ "${G3_ACTOR8_ONLY}" != "1" ]; then
    required_host_paths+=("${HOST_FONTCONFIG_DIR}" "${HOST_FONTS_DIR}" "${EXTRA_FONTS_DIR}")
fi
for required_host_path in "${required_host_paths[@]}"; do
    if [ ! -e "${required_host_path}" ]; then
        echo "ERROR: required host path for container mount is missing: ${required_host_path}" >&2
        echo "       (EXTRA_FONTS_DIR needs a CJK font, e.g. NotoSansCJKsc-Regular.otf)" >&2
        exit 1
    fi
done
CONTAINER_MOUNTS="${HOST_LIB_DIR}:/host_usr_lib64:ro"
if [ "${G2_JUDGE_TOPOLOGY_ONLY}" != "1" ] && [ "${G3_ACTOR8_ONLY}" != "1" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${HOST_FONTCONFIG_DIR}:/etc/fonts:ro"
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${HOST_FONTS_DIR}:/usr/share/fonts:ro"
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${EXTRA_FONTS_DIR}:/usr/local/share/fonts:ro"
fi

# sgl_kernel needs libnuma; keep the rest of the host libraries off the default
# search path so they cannot shadow the container's CUDA/torch stack. Only the
# browser subprocess gets the full host library directory, via
# BROWSER_LD_LIBRARY_PATH (applied by examples/mobilegym_agentic/app/agent.py).
export SGLANG_NUMA_LIBRARY="${SGLANG_NUMA_LIBRARY:-/host_usr_lib64/libnuma.so.1.0.0}"
export BROWSER_HOST_LIB_DIR="${BROWSER_HOST_LIB_DIR:-/host_usr_lib64}"

MOBILEGYM_REPO_DIR="${MOBILEGYM_REPO_DIR:-${WORK_ROOT}/mobilegym}"
# MobileGym's bench_env.run needs Playwright + Chromium and must run INSIDE the
# container, so it uses the same container-built venv as Relax rather than the
# host-side python3.11 venv (which is not importable in the EDF image).
MOBILEGYM_PYTHON="${MOBILEGYM_PYTHON:-${RELAX_ENV_ROOT}/relax_venv/bin/python}"
# No safe default here: this script body runs on an allocated compute node,
# where "localhost" is that compute node's own loopback, not the login node
# running the nginx gateway -- a silent localhost default would fail with a
# confusing "unreachable" error on every submission. Caller must pass the
# gateway address explicitly; it must be reachable from every rollout worker.
if [ "${G2_JUDGE_TOPOLOGY_ONLY}" != "1" ] && [ "${G3_ROLLOUT8_ONLY}" != "1" ] && [ "${G4_ROLLOUT12_ONLY}" != "1" ] && [ "${G3_ACTOR8_ONLY}" != "1" ] && [ -z "${MOBILEGYM_ENV_URL:-}" ]; then
    echo "ERROR: MOBILEGYM_ENV_URL must be set, e.g.:" >&2
    echo '  --export=ALL,MOBILEGYM_ENV_URL=https://'"$(hostname)"':4180,...' >&2
    exit 1
fi
if [ "${G2_JUDGE_TOPOLOGY_ONLY}" != "1" ] && [ "${G3_ROLLOUT8_ONLY}" != "1" ] && [ "${G4_ROLLOUT12_ONLY}" != "1" ] && [ "${G3_ACTOR8_ONLY}" != "1" ]; then
    export MOBILEGYM_ENV_URL
    export MOBILEGYM_REPO_DIR
    export MOBILEGYM_PYTHON
fi

export MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models}"
export DATA_DIR="${DATA_DIR:-${WORK_ROOT}/data}"
export SAVE_DIR="${SAVE_DIR:-${WORK_ROOT}/checkpoints}"
export EXP_DIR="${EXP_DIR:-${WORK_ROOT}/exp/${SLURM_JOB_ID}}"
# FlashInfer protects generated sampling modules with POSIX file locks.  Its
# default cache under a shared home can fail under the
# 24-GPU startup fan-out with ENOLCK.  Keep one cache per job on each node's
# local filesystem so engines on that node can safely share compiled modules,
# without carrying a warm JIT cache across experimental runs.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/relax-flashinfer/${SLURM_JOB_ID}}"
if [ "${G4_FULL12_ONLY}" = "1" ]; then
    export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
elif [ "${G5_FULL24_ONLY}" = "1" ]; then
    export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
else
    export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
fi
export REASONING_TRIGGER="${REASONING_TRIGGER:-terminal_once}"
export DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
export G3_ROLLOUT8_ONLY
export G3_ACTOR8_ONLY
export G3_DUAL8_ONLY
export G4_ROLLOUT12_ONLY
export G4_FULL12_ONLY
export G5_FULL24_ONLY
if [ "${G3_ROLLOUT8_ONLY}" = "1" ] || [ "${G3_ACTOR8_ONLY}" = "1" ] || [ "${G3_DUAL8_ONLY}" = "1" ]; then
    if [ "${NUM_NODES}" -ne 2 ] || [ "${GPUS_PER_NODE}" -ne 4 ]; then
        echo "ERROR: G3 rollout8/actor8 modes require exactly two nodes with four GPUs each." >&2
        exit 1
    fi
    export RAY_CLUSTER_READY_TIMEOUT_S="${RAY_CLUSTER_READY_TIMEOUT_S:-180}"
    export RAY_GCS_WAIT_ATTEMPTS="${RAY_GCS_WAIT_ATTEMPTS:-36}"
    export RAY_JOIN_ATTEMPTS="${RAY_JOIN_ATTEMPTS:-12}"
fi
if [ "${G4_ROLLOUT12_ONLY}" = "1" ]; then
    if [ "${NUM_NODES}" -ne 3 ] || [ "${GPUS_PER_NODE}" -ne 4 ]; then
        echo "ERROR: G4_ROLLOUT12_ONLY=1 requires exactly three nodes with four GPUs each." >&2
        exit 1
    fi
    export ROLLOUT_ENGINE_COUNT=12
    export ROLLOUT_SAMPLE_COUNT=64
    export RAY_CLUSTER_READY_TIMEOUT_S="${RAY_CLUSTER_READY_TIMEOUT_S:-180}"
    export RAY_GCS_WAIT_ATTEMPTS="${RAY_GCS_WAIT_ATTEMPTS:-36}"
    export RAY_JOIN_ATTEMPTS="${RAY_JOIN_ATTEMPTS:-12}"
fi
if [ "${G4_FULL12_ONLY}" = "1" ]; then
    if [ "${NUM_NODES}" -ne 3 ] || [ "${GPUS_PER_NODE}" -ne 4 ]; then
        echo "ERROR: G4_FULL12_ONLY=1 requires exactly three nodes with four GPUs each." >&2
        exit 1
    fi
    if [ "${NUM_ROLLOUT}" -ne 2 ]; then
        echo "ERROR: G4_FULL12_ONLY=1 requires exactly two publication rounds." >&2
        exit 1
    fi
    export RAY_CLUSTER_READY_TIMEOUT_S="${RAY_CLUSTER_READY_TIMEOUT_S:-180}"
    export RAY_GCS_WAIT_ATTEMPTS="${RAY_GCS_WAIT_ATTEMPTS:-36}"
    export RAY_JOIN_ATTEMPTS="${RAY_JOIN_ATTEMPTS:-12}"
fi
if [ "${G5_FULL24_ONLY}" = "1" ]; then
    if [ "${NUM_NODES}" -ne 6 ] || [ "${GPUS_PER_NODE}" -ne 4 ]; then
        echo "ERROR: G5_FULL24_ONLY=1 requires exactly six nodes with four GPUs each." >&2
        exit 1
    fi
    if [ "${NUM_ROLLOUT}" -lt 3 ]; then
        echo "ERROR: G5_FULL24_ONLY=1 requires at least three rounds (one warmup plus at least two measured)." >&2
        exit 1
    fi
    export RAY_CLUSTER_READY_TIMEOUT_S="${RAY_CLUSTER_READY_TIMEOUT_S:-180}"
    export RAY_GCS_WAIT_ATTEMPTS="${RAY_GCS_WAIT_ATTEMPTS:-36}"
    export RAY_JOIN_ATTEMPTS="${RAY_JOIN_ATTEMPTS:-12}"
fi
# sglang_cuda13.sqsh ships PyTorch 2.9.1 + CuDNN 9.13; SGLang's own startup
# guard (server_args.py check_torch_2_9_1_cudnn_compatibility, for a known
# nn.Conv3d perf bug: https://github.com/pytorch/pytorch/issues/168167) hard-
# fails engine init on that combination unconditionally, on every rank. This
# is a perf advisory, not a correctness issue -- safe to bypass for rollout
# smoke tests; revisit before perf-sensitive runs on this image.
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
echo "NUM_ROLLOUT=${NUM_ROLLOUT} REASONING_TRIGGER=${REASONING_TRIGGER} DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY} G3_ROLLOUT8_ONLY=${G3_ROLLOUT8_ONLY} G3_ACTOR8_ONLY=${G3_ACTOR8_ONLY} G3_DUAL8_ONLY=${G3_DUAL8_ONLY} G4_FULL12_ONLY=${G4_FULL12_ONLY} G5_FULL24_ONLY=${G5_FULL24_ONLY} G2_JUDGE_TOPOLOGY_ONLY=${G2_JUDGE_TOPOLOGY_ONLY}"

mkdir -p "${DATA_DIR}" "${SAVE_DIR}" "${EXP_DIR}" "${EXP_DIR}/flashinfer_workspace"
export RELAX_SPMD_COMPLETION_FILE="${RELAX_SPMD_COMPLETION_FILE:-${EXP_DIR}/spmd_completion}"
if [ -e "${RELAX_SPMD_COMPLETION_FILE}" ]; then
    echo "ERROR: SPMD completion marker already exists: ${RELAX_SPMD_COMPLETION_FILE}" >&2
    exit 1
fi

###############################################################################
#                        MOBILEGYM SIMULATOR REACHABILITY                     #
###############################################################################
# The simulator (nginx gateway) is started once, out of band, on the login
# node before submitting (see examples/mobilegym_agentic/README.md) -- it is
# not part of this allocation's lifecycle. Fail fast with a clear message
# rather than having every rollout episode fail one at a time.

if [ "${G2_JUDGE_TOPOLOGY_ONLY}" != "1" ] && [ "${G3_ROLLOUT8_ONLY}" != "1" ] && [ "${G4_ROLLOUT12_ONLY}" != "1" ] && [ "${G3_ACTOR8_ONLY}" != "1" ] && ! curl -sk -o /dev/null --max-time 10 "${MOBILEGYM_ENV_URL}"; then
    echo "ERROR: MobileGym simulator unreachable at ${MOBILEGYM_ENV_URL}." >&2
    echo "Start it first: see examples/mobilegym_agentic/README.md" >&2
    exit 1
fi

###############################################################################
#                      ONE-TIME RELAX VENV BOOTSTRAP (idempotent)             #
###############################################################################
# Runs once inside the container on the head node only; persists on
# shared work root so subsequent job submissions can reuse it.

NODELIST=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD_NODE="${NODELIST[0]}"
mkdir -p "${RELAX_ENV_ROOT}"

if [ "${G4_FULL12_ONLY}" = "1" ] || [ "${G5_FULL24_ONLY}" = "1" ]; then
    CLOCK_AUDIT_FILE="${EXP_DIR}/clock_sync_samples.jsonl"
    CLOCK_AUDIT_SUMMARY="${EXP_DIR}/clock_sync_audit.json"
    HOST_PYTHON="${HOST_PYTHON:-python3.11}"
    srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
        "${HOST_PYTHON}" "${RELAX_REPO_DIR}/examples/mobilegym_agentic/scripts/audit_cluster_clock.py" sample \
        >"${CLOCK_AUDIT_FILE}"
    "${HOST_PYTHON}" "${RELAX_REPO_DIR}/examples/mobilegym_agentic/scripts/audit_cluster_clock.py" summarize \
        "${CLOCK_AUDIT_FILE}" --expected-hosts "${NUM_NODES}" >"${CLOCK_AUDIT_SUMMARY}"
    export RELAX_MULTI_HOST_CLOCK_MAX_OFFSET_MS="$(
        "${HOST_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_pairwise_offset_ms"])' \
            "${CLOCK_AUDIT_SUMMARY}"
    )"
    echo "RELAX_MULTI_HOST_CLOCK_MAX_OFFSET_MS=${RELAX_MULTI_HOST_CLOCK_MAX_OFFSET_MS}"
fi

if [ "${G5_FULL24_ONLY}" = "1" ]; then
    GPU_INVENTORY_FILE="${EXP_DIR}/allocation_gpu_inventory.jsonl"
    srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
        "${HOST_PYTHON:-python3.11}" \
        "${RELAX_REPO_DIR}/examples/mobilegym_agentic/scripts/capture_gpu_inventory.py" \
        >"${GPU_INVENTORY_FILE}"
fi

srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" --environment="${EDF_TOML}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash "${RELAX_REPO_DIR}/examples/mobilegym_agentic/scripts/setup_relax_env.sh" \
    "${RELAX_ENV_ROOT}" "${RELAX_REPO_DIR}" "${MOBILEGYM_REPO_DIR}"

export MEGATRON_DIR="${RELAX_ENV_ROOT}/Megatron-LM"
if [ -n "${MEGATRON:-}" ] && [ "${MEGATRON}" != "${MEGATRON_DIR}" ]; then
    echo "ERROR: MEGATRON=${MEGATRON} conflicts with bootstrap root ${MEGATRON_DIR}." >&2
    exit 1
fi
export MEGATRON="${MEGATRON_DIR}"
export TRANSFER_QUEUE_DIR="${TRANSFER_QUEUE_DIR:-${RELAX_ENV_ROOT}/TransferQueue}"
export VENV_BIN="${RELAX_ENV_ROOT}/relax_venv/bin"
export CUDNN_LIB_DIR="${CUDNN_LIB_DIR:-${RELAX_ENV_ROOT}/relax_venv/lib/python3.12/site-packages/nvidia/cudnn/lib}"
if [ ! -d "${CUDNN_LIB_DIR}" ]; then
    echo "ERROR: expected venv cuDNN library directory is missing: ${CUDNN_LIB_DIR}" >&2
    exit 1
fi
export RUN_SCRIPT_PATH="${RELAX_REPO_DIR}/${RUN_SCRIPT}"
export RELAX="${RELAX_REPO_DIR}"

###############################################################################
#                          RESOLVE MASTER_ADDR (head node IP)                 #
###############################################################################

export MASTER_ADDR="$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')"
if [ -z "${MASTER_ADDR}" ]; then
    echo "ERROR: could not resolve an IP for head node ${HEAD_NODE}" >&2
    exit 1
fi
echo "MASTER_ADDR=${MASTER_ADDR} (head node: ${HEAD_NODE})"

###############################################################################
#                    LAUNCH: ONE TASK PER NODE, INSIDE THE CONTAINER          #
###############################################################################
# Every node runs the exact same command; scripts/entrypoint/spmd-multinode.sh
# decides head vs
# worker role by comparing POD_NAME (this node's own IP) against MASTER_ADDR.
# Only the head node's invocation actually execs RUN_SCRIPT (which does the
# `ray job submit` against its own local dashboard); workers block in
# spmd-multinode.sh after joining the Ray cluster.

# All variables the inner script needs (MASTER_ADDR, VENV_BIN, RELAX,
# RUN_SCRIPT_PATH, ...) are passed via --export and referenced as plain $VAR
# below -- deliberately no string interpolation into the quoted script body,
# which is easy to get subtly wrong with nested quoting.
srun --nodes="${NUM_NODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 --environment="${EDF_TOML}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --export=ALL,MASTER_ADDR,WORLD_SIZE="${NUM_NODES}",NUM_GPUS="${GPUS_PER_NODE}",NUM_GPUS_TOTAL="${GPUS_PER_NODE}",MEGATRON,MEGATRON_DIR,RELAX,VENV_BIN,CUDNN_LIB_DIR,FLASHINFER_WORKSPACE_BASE,RUN_SCRIPT_PATH,SGLANG_NUMA_LIBRARY,BROWSER_HOST_LIB_DIR,SGLANG_DISABLE_CUDNN_CHECK,TRANSFER_QUEUE_DIR,RAY_CLUSTER_READY_TIMEOUT_S,RAY_GCS_WAIT_ATTEMPTS,RAY_JOIN_ATTEMPTS,RELAX_SPMD_COMPLETION_FILE \
    bash -c '
        set -euo pipefail
        export HOST_IP="$(hostname -I | awk "{print \$1}")"
        export POD_NAME="${HOST_IP}"
        export PATH="${VENV_BIN}:${PATH}"
        # sgl_kernel dlopens libnuma; preload the one host object rather than
        # putting the whole host library directory on LD_LIBRARY_PATH, which
        # would let host libs shadow the containers CUDA/torch stack.
        if [ ! -r "${SGLANG_NUMA_LIBRARY}" ]; then
            echo "ERROR: SGLang NUMA library missing in container: ${SGLANG_NUMA_LIBRARY}" >&2
            exit 1
        fi
        export LD_PRELOAD="${SGLANG_NUMA_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
        # Transformer Engine was built against the venv cuDNN. Keep that
        # library ahead of the container copy for the Ray driver and workers.
        export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        # The FlashInfer shared-filesystem default can raise ENOLCK while several
        # engines JIT the same sampling module.  Exercise the same fcntl lock
        # primitive on every node before starting Ray, and fail closed.
        python3 "${RELAX}/examples/mobilegym_agentic/scripts/check_flashinfer_workspace.py" \
            --output "${EXP_DIR}/flashinfer_workspace/$(hostname).json"
        echo "[node $(hostname)] FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"
        # Ray is the container-wide install (no ray binary inside the venv), so
        # Ray workers run under the container interpreter and cannot see the
        # venv site-packages -- transfer_queue, relax and friends live there and
        # the TransferQueueController actor dies with ModuleNotFoundError.
        # spmd-multinode.sh forwards PYTHONPATH into the Ray runtime_env, so
        # exporting it here reaches head and workers alike.
        VENV_SITE="$("${VENV_BIN}/python" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")"
        export PYTHONPATH="${MEGATRON_DIR}:${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
        # TransferQueue is an editable install, and editable installs work via a
        # .pth file that only site.py executes for real site-packages dirs -- a
        # directory added through PYTHONPATH never has its .pth files processed.
        # So the venv path above is not enough; point at the source tree itself.
        if [ -d "${TRANSFER_QUEUE_DIR}/transfer_queue" ]; then
            export PYTHONPATH="${TRANSFER_QUEUE_DIR}:${PYTHONPATH}"
        fi
        # Only the Playwright/Chromium subprocess needs the host GLib/X11/GBM/NSS
        # libraries; app/agent.py applies this to that subprocess alone.
        export BROWSER_LD_LIBRARY_PATH="${BROWSER_HOST_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        echo "[node $(hostname)] HOST_IP=${HOST_IP} POD_NAME=${POD_NAME} MASTER_ADDR=${MASTER_ADDR}"
        cd "${RELAX}"
        exec bash scripts/entrypoint/spmd-multinode.sh "${RUN_SCRIPT_PATH}"
    '

echo "END TIME: $(date)"
