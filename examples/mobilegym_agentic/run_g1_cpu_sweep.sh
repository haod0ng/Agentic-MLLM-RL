#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Run this from an allocated CPU node (for example, an Slurm debug allocation).
# It intentionally makes no GPU or Ray request: G1 validates MobileGym browser
# concurrency before any Relax training topology is admitted.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MOBILEGYM_REPO_DIR="${MOBILEGYM_REPO_DIR:?set MOBILEGYM_REPO_DIR}"
MOBILEGYM_PYTHON="${MOBILEGYM_PYTHON:?set MOBILEGYM_PYTHON}"
MOBILEGYM_ENV_URL="${MOBILEGYM_ENV_URL:?set MOBILEGYM_ENV_URL}"
G1_OUTPUT_ROOT="${G1_OUTPUT_ROOT:?set G1_OUTPUT_ROOT to a fresh output directory}"

G1_CONCURRENCIES="${G1_CONCURRENCIES:-1,1,8,32,64}"
IFS=',' read -r -a concurrency_values <<< "${G1_CONCURRENCIES}"
if [ "${#concurrency_values[@]}" -eq 0 ]; then
    echo "ERROR: G1_CONCURRENCIES must contain at least one value." >&2
    exit 2
fi
CONCURRENCY_ARGS=()
for concurrency in "${concurrency_values[@]}"; do
    CONCURRENCY_ARGS+=(--concurrency "${concurrency}")
done

exec "${MOBILEGYM_PYTHON}" "${SCRIPT_DIR}/g1_cpu_sweep.py" \
    --mobilegym-repo "${MOBILEGYM_REPO_DIR}" \
    --mobilegym-python "${MOBILEGYM_PYTHON}" \
    --env-url "${MOBILEGYM_ENV_URL}" \
    --output-root "${G1_OUTPUT_ROOT}" \
    "${CONCURRENCY_ARGS[@]}"
