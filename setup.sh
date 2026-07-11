#!/usr/bin/env bash
# vllm-fp8kv environment setup. Target: fresh GPU box (Vast.ai -devel image or
# bare metal, CUDA 12.x, sm_80/sm_86) productive in <= 15 minutes. Idempotent.
#
# What it does:
#   1. venv + vLLM NIGHTLY wheel  (the 0.24.0 release does NOT work: PR #47629's
#      Python code calls into _C_stable_libtorch symbols only nightly ships)
#   2. clone vLLM, check out PR #47629 at the PINNED head below
#   3. overlay the PR's Python files onto the installed wheel — the PR is
#      Python-only for this path (its one .cu file is nvfp4 quant, unrelated)
#   4. apply our fp8 kernel patch to the OVERLAID site-packages copy — that is
#      the runtime the verify scripts import. Patching only the git clone is a
#      no-op, which an earlier version of the README got wrong.
#   5. run the harness
#
# Usage: ./setup.sh          GPU box: full install + verify
#        ./setup.sh --cpu    CPU box: deps only (no vLLM, no kernels)
set -euo pipefail
cd "$(dirname "$0")"

# PR #47629 head this patch was generated and measured against. A newer head may
# drift the anchors; the generator then fails loudly (ANCHOR NOT FOUND) rather
# than silently mispatching.
PR_HEAD=bbe2ab4d6cc01a634640d0b502ac67055d114219
CPU_ONLY=${1:-}

python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate
pip install -q -U pip

if [ "$CPU_ONLY" = "--cpu" ]; then
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q pytest numpy
    echo "vllm-fp8kv CPU env ready (static checks only)."
    exit 0
fi

nvidia-smi -L
which nvcc >/dev/null || { echo "FATAL: nvcc not found — use a -devel image"; exit 1; }

pip install -q -U vllm --extra-index-url https://wheels.vllm.ai/nightly
pip install -q pytest numpy

SITE=$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')

# 2. + 3. PR #47629 overlay onto the installed wheel
[ -d vllm-src ] || git clone --filter=blob:none https://github.com/vllm-project/vllm vllm-src
git -C vllm-src fetch -q origin pull/47629/head:pr47629 || true
git -C vllm-src checkout -q "$PR_HEAD"

for f in \
    model_executor/layers/sparse_attn_indexer.py \
    model_executor/models/deepseek_v2.py \
    platforms/cuda.py \
    v1/attention/backends/mla/indexer.py \
    v1/attention/backends/mla/triton_mla_sparse.py \
    v1/attention/backends/registry.py \
    v1/attention/ops/mqa_logits_triton.py \
    v1/attention/ops/triton_mla_sparse_kernel.py
do
    cp "vllm-src/vllm/$f" "$SITE/$f"
done
echo "overlaid PR #47629 (@ ${PR_HEAD:0:9}) onto $SITE"

# 4. our fp8 branch, applied to the RUNTIME copy. (make_fp8_kernel_patch.py
#    RE-DERIVES the patch and needs a git clone; here we just apply the shipped
#    diff to site-packages, which is what the verify scripts actually import.)
patch -p1 -d "$(dirname "$SITE")" < patches/triton_mla_sparse_fp8.patch
python -c "import py_compile; py_compile.compile('$SITE/v1/attention/ops/triton_mla_sparse_kernel.py', doraise=True)"
echo "applied fp8 kernel patch to the runtime"

# 5. verify
python -m pytest tests/ -q
python verify/run_all.py
python verify/contract.py
python verify/integration.py
echo "vllm-fp8kv environment ready."
