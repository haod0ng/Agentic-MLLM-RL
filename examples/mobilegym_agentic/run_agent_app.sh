#!/bin/bash

export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"
# MobileGym's screenshots are append-only trajectory evidence for the VLM
# judge.  Keep the entire interaction history unless a caller explicitly opts
# into a different experimental contract.
export MOBILEGYM_HISTORY_IMAGES="${MOBILEGYM_HISTORY_IMAGES:-1}"

# Managed-session launchers inherit the node environment, but ``python`` is not
# a portable executable name in the EDF image (some layers expose only
# ``python3``).  Resolve the interpreter explicitly so an interpreter lookup
# failure cannot look like a zero-length MobileGym run to the prepare pool.
AGENT_PYTHON="${RELAX_AGENT_PYTHON:-python3}"
if ! command -v "${AGENT_PYTHON}" >/dev/null 2>&1; then
    echo "ERROR: agent interpreter is unavailable: ${AGENT_PYTHON}" >&2
    exit 127
fi
exec "${AGENT_PYTHON}" -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
