#!/usr/bin/env bash
# sm86-dsa environment setup. Target: fresh GPU box (Vast.ai -devel image or
# bare metal with CUDA 12.x) productive in <= 10 minutes. Idempotent.
# Phase A (CPU-only) boxes: run with --cpu.
set -euo pipefail
cd "$(dirname "$0")"

CPU_ONLY=${1:-}

python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate
pip install -q -U pip

if [ "$CPU_ONLY" = "--cpu" ]; then
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
else
    nvidia-smi -L
    which nvcc >/dev/null || { echo "FATAL: nvcc not found — use a -devel image"; exit 1; }
    pip install -q torch --index-url https://download.pytorch.org/whl/cu126
    pip install -q triton
fi
pip install -q pytest numpy

# Phases B+ additionally need sglang (from source, nsa/tilelang DSA backend)
# and tilelang 0.1.11 + tvm-ffi, pinned per renning22's setup. Installed by a
# separate script once Phase B starts, to keep Phase A boxes light:
#   ./setup_sglang.sh   (TODO Phase B — pin exact commits here)

python -m pytest tests/ -q
echo "sm86-dsa environment ready."
