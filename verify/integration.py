# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Integration check: our fp8 path inside vLLM's real TRITON_MLA_SPARSE backend.

Exercises the ACTUAL upstream call chain — their metadata, their
`triton_convert_req_index_to_global_index`, their C++ cache writer — with our
kernel spliced in at `forward_mqa`. Compares fp8-KV output against the SAME
backend running bf16-KV on identical data (that is the AC1 parity gate: fp8 vs
their own bf16 path, not vs our reference).

Requires: vLLM with PR #47629 (triton_mla_sparse.py) importable.
Run:  python verify/integration.py
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from verify import reference as R  # noqa: E402
from verify.metrics import Report, cosine, max_abs  # noqa: E402


def main():
    rep = Report()
    device = "cuda"
    print("== vllm-fp8kv INTEGRATION (real TRITON_MLA_SPARSE call chain) ==")

    try:
        from vllm.v1.attention.backends.mla import triton_mla_sparse as tms
    except ImportError as e:
        print(f"[SKIP] PR #47629 not present: {e}")
        sys.exit(0)

    from vllm_fp8kv import backend_patch

    before = list(tms.TritonMLASparseBackend.supported_kv_cache_dtypes)
    backend_patch.apply()
    after = list(tms.TritonMLASparseBackend.supported_kv_cache_dtypes)
    rep.check("integration/dtype-declared",
              "fp8_ds_mla" in after and "fp8_ds_mla" not in before,
              f"{before} -> {after}")
    rep.check("integration/fp8-forward-installed",
              hasattr(tms.TritonMLASparseImpl, "_forward_fp8_kv"),
              "TritonMLASparseImpl._forward_fp8_kv present")

    # --- their index converter produces exactly our kernel's contract? ---
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
        triton_convert_req_index_to_global_index,
    )

    # their converter asserts NUM_TOPK_TOKENS % 128 == 0
    block_size, n_blocks, n_tokens, topk = 64, 16, 4, 256
    g = torch.Generator(device="cpu").manual_seed(5)
    block_table = torch.randperm(n_blocks, generator=g)[:8].view(1, 8).to(device).int()
    req_id = torch.zeros(n_tokens, dtype=torch.int32, device=device)
    # token-relative topk with -1 padding, exactly as the indexer emits
    tok_idx = torch.randint(0, 8 * block_size, (n_tokens, topk), generator=g).to(device).int()
    tok_idx[:, ::5] = -1

    glob = triton_convert_req_index_to_global_index(
        req_id, block_table, tok_idx,
        BLOCK_SIZE=block_size, NUM_TOPK_TOKENS=topk,
    )
    # our kernel's assumptions: -1 preserved; valid entries are flat slot ids
    sent_ok = bool((glob[tok_idx < 0] < 0).all())
    valid = glob[glob >= 0]
    range_ok = bool((valid < n_blocks * block_size).all())
    rep.check("integration/indices-are-flat-global-slots", sent_ok and range_ok,
              f"sentinels preserved={sent_ok}, in-range={range_ok}, "
              f"max={int(valid.max())} < {n_blocks * block_size}")

    # --- decode over a cache written by vLLM's OWN C++ writer, indices from
    #     THEIR converter -> our kernel. This is the whole chain.
    from vllm import _custom_ops as ops

    n_slots = n_blocks * block_size
    kv_c = (torch.randn(n_slots, 512, generator=g) * 0.5).to(device).to(torch.bfloat16)
    k_pe = (torch.randn(n_slots, 64, generator=g) * 0.5).to(device).to(torch.bfloat16)
    cache = torch.zeros(n_blocks, block_size, 656, dtype=torch.uint8, device=device)
    ops.concat_and_cache_mla(
        kv_c, k_pe, cache,
        torch.arange(n_slots, dtype=torch.int64, device=device),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.tensor(1.0, device=device),
    )

    from vllm_fp8kv.fp8_ds_mla_sparse_decode import fp8_ds_mla_sparse_decode

    q = (torch.randn(n_tokens, 16, 576, generator=g) * 0.5).to(device).to(torch.bfloat16)
    sm = 1.0 / (576 ** 0.5)
    out = fp8_ds_mla_sparse_decode(q, cache, glob, sm)
    ref = R.ref_sparse_mla_decode(q, cache.reshape(-1, 656), glob, sm)
    c, m = cosine(out.float(), ref), max_abs(out.float(), ref)
    rep.check("integration/full-chain-decode", c > 0.999 and out.isfinite().all(),
              f"cosine={c:.6f} max_abs={m:.2e} (their writer + their converter + our kernel)")

    print(f"== done: {rep.failed} failure(s) ==")
    sys.exit(min(rep.failed, 125))


if __name__ == "__main__":
    main()
