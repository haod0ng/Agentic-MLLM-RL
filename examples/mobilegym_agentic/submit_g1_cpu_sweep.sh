#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Submit with an explicitly chosen account/reservation, for example:
#   sbatch --account=<account> --reservation=<reservation> --partition=debug \
#     --export=ALL,RELAX_REPO_DIR=<Relax-repo>,MOBILEGYM_ENV_URL=https://<gateway-host>:4180,\
#MOBILEGYM_REPO_DIR=<repo>,MOBILEGYM_PYTHON=<python>,G1_OUTPUT_ROOT=<fresh-dir> \
#     examples/mobilegym_agentic/submit_g1_cpu_sweep.sh
#
# No GPU is requested.  G1 establishes browser/env capacity before the GPU
# topology validation stages, and should run on an otherwise quiet node.  The
# runner is entered through the same EDF image as the E2E path so that the
# pinned MobileGym/Playwright environment is available on compute nodes.

#SBATCH --job-name=mobilegym-g1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --no-requeue

set -euo pipefail

if [ -z "${RELAX_REPO_DIR:-}" ] || [ -z "${MOBILEGYM_ENV_URL:-}" ] || [ -z "${MOBILEGYM_REPO_DIR:-}" ] || [ -z "${MOBILEGYM_PYTHON:-}" ] || [ -z "${G1_OUTPUT_ROOT:-}" ]; then
    echo "ERROR: set RELAX_REPO_DIR, MOBILEGYM_ENV_URL, MOBILEGYM_REPO_DIR, MOBILEGYM_PYTHON, and a fresh G1_OUTPUT_ROOT." >&2
    exit 2
fi
RUNNER="${RELAX_REPO_DIR}/examples/mobilegym_agentic/run_g1_cpu_sweep.sh"
if [ ! -f "${RUNNER}" ]; then
    echo "ERROR: G1 runner is missing: ${RUNNER}" >&2
    exit 2
fi
if ! curl -sk --max-time 10 -o /dev/null "${MOBILEGYM_ENV_URL}"; then
    echo "ERROR: MobileGym environment is unreachable: ${MOBILEGYM_ENV_URL}" >&2
    exit 2
fi

echo "G1 job=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
: "${EDF_TOML:?set EDF_TOML to the Slurm container environment TOML}"
: "${RELAX_ENV_ROOT:?set RELAX_ENV_ROOT to the container runtime root}"
MOBILEGYM_PYTHON="${MOBILEGYM_PYTHON:-${RELAX_ENV_ROOT}/relax_venv/bin/python}"
HOST_LIB_DIR="${HOST_LIB_DIR:-/usr/lib64}"
HOST_FONTCONFIG_DIR="${HOST_FONTCONFIG_DIR:-/etc/fonts}"
HOST_FONTS_DIR="${HOST_FONTS_DIR:-/usr/share/fonts}"
: "${EXTRA_FONTS_DIR:?set EXTRA_FONTS_DIR to a directory containing required browser fonts}"
for required_path in "${EDF_TOML}" "${HOST_LIB_DIR}" "${HOST_FONTCONFIG_DIR}" "${HOST_FONTS_DIR}" "${EXTRA_FONTS_DIR}"; do
    if [ ! -e "${required_path}" ]; then
        echo "ERROR: required G1 container path is missing: ${required_path}" >&2
        exit 2
    fi
done
CONTAINER_MOUNTS="${HOST_LIB_DIR}:/host_usr_lib64:ro,${HOST_FONTCONFIG_DIR}:/etc/fonts:ro,${HOST_FONTS_DIR}:/usr/share/fonts:ro,${EXTRA_FONTS_DIR}:/usr/local/share/fonts:ro"
export MOBILEGYM_PYTHON
export BROWSER_HOST_LIB_DIR="${BROWSER_HOST_LIB_DIR:-/host_usr_lib64}"
export PYTHONPATH="${MOBILEGYM_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# G1 invokes Playwright directly (rather than through the agent wrapper that
# scopes this variable to Chromium). Keep container libraries first, then add
# the mounted host desktop libraries needed by the EDF image's Chromium.
export G1_RUNNER="${RUNNER}"
exec srun --nodes=1 --ntasks=1 --environment="${EDF_TOML}" --container-mounts="${CONTAINER_MOUNTS}" \
    bash -lc 'export LD_LIBRARY_PATH="/opt/conda/lib:/usr/local/nvidia/lib64:/host_usr_lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; exec bash "${G1_RUNNER}"'
