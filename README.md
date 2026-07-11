# vllm-fp8kv — fp8 KV cache for vLLM's Ampere DSA path

vLLM's Ampere sparse-MLA backend (`TRITON_MLA_SPARSE`, PR #47629) is
**bf16-KV-only**, because *"Triton fp8e4nv store … does not compile on SM80"*.
That doubles KV bytes per token (1152 B vs 656 B) and is the binding constraint
on 24 GB consumer cards.

This repo removes that constraint **without asking Triton for fp8 at all**: the
standard `fp8_ds_mla` pages are loaded as raw `uint8` and decoded in-register
with bit-math that is bit-exact against `torch.float8_e4m3fn`.

**Verified on RTX 3090:** cosine **0.999996** end-to-end (vLLM's own C++ cache
writer → vLLM's own index converter → our fp8 kernel), **1.756× KV pool**
(the format's ceiling). Full numbers, including where it is *not* yet fast
enough: **[RESULTS.md](RESULTS.md)**.

Same architecture vLLM already accepts for XPU
(`models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py`: *"dequantize FP8 KV cache
pages to BF16 on the fly, then reuse the BF16 sparse MLA attention kernel"*),
brought to CUDA sm_80/sm_86.

## Layout

```
patches/triton_mla_sparse_fp8.patch   the upstream-able diff (224 lines) —
                                      adds an IS_FP8 branch to upstream's own
                                      _sparse_mla_compute_tile, so fp8 inherits
                                      their split-KV, autotune and merge
patches/make_fp8_kernel_patch.py      regenerates it against a moving upstream
vllm_fp8kv/fp8_dequant.py             the fp8e4m3fn software decode (bit-exact)
vllm_fp8kv/fp8_ds_mla_sparse_decode.py  standalone kernel (reference impl / tests)
vllm_fp8kv/backend_patch.py           declares fp8_ds_mla on the backend + wires forward
verify/                               model-free harness: golden refs, contract
                                      tests vs vLLM ITSELF, integration, bench
PREFLIGHT.md  PORT_PLAN.md  RESULTS.md
```

## Reproduce

```bash
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
git clone https://github.com/vllm-project/vllm && cd vllm
git fetch origin pull/47629/head:pr47629 && git checkout pr47629
cp -r vllm/{v1,model_executor,platforms}/… "$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"/…   # PR is Python-only
python patches/make_fp8_kernel_patch.py .   # generates + applies the fp8 branch
python verify/run_all.py && python verify/contract.py && python verify/integration.py
```

## Provenance

renning22/glm-5.2-4090 → [nickus/dsa-3090](https://github.com/nickus/dsa-3090)
(the sm_86 DSA port; software fp8 dequant originates there) → this repo.
Apache-2.0 throughout; see `NOTICE`.
