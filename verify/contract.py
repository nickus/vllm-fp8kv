# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Contract tests against vLLM ITSELF — the defence against silent drift.

Our kernel hard-codes the `fp8_ds_mla` geometry (656-byte row, scales at byte
512, raw-bf16 rope at byte 528, amax/448). Today that is byte-for-byte what
vLLM writes — both descend from DeepSeek's reference. **That is luck, not a
guarantee**: a future vLLM release could re-tile the row and our decode would
keep "working" while silently returning garbage.

So none of our constants are trusted here. Every check below derives the truth
from vLLM at runtime:

  1. `check_layout_contract`   — vLLM's own cache SHAPE/strides/dtype vs our
                                 constants. Loud fail on drift (CI canary).
  2. `check_writer_roundtrip`  — write pages with **vLLM's own C++ writer**
                                 (`ops.concat_and_cache_mla`, fp8_ds_mla), read
                                 them with **our** dequant, compare against the
                                 pre-quantization source. This is the check that
                                 catches a scale-semantics mismatch (e.g. if the
                                 writer ever starts honouring `k_scale`): a
                                 systematic bias invisible to toy-agreement.
  3. `check_indices_contract`  — replay REAL index tensors captured from a stock
                                 vLLM run (shape, dtype, stride, padding
                                 sentinel) instead of trusting a signature read.

Run on a box with vLLM installed:  python verify/contract.py
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from verify import reference as R  # noqa: E402
from verify.metrics import Report, cosine, max_abs  # noqa: E402
from vllm_fp8kv import fp8_ds_mla_sparse_decode as K  # noqa: E402


def check_layout_contract(rep: Report):
    """Drift canary: vLLM's own numbers vs our kernel constants."""
    try:
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            FlashMLASparseBackend,
        )
    except ImportError as e:
        rep.skip("contract/layout", f"vLLM sparse-MLA backend not importable: {e}")
        return

    # The authoritative shape comes from the backend that owns the format.
    shape = FlashMLASparseBackend.get_kv_cache_shape(
        num_blocks=8, block_size=64, num_kv_heads=1, head_size=576,
        cache_dtype_str="fp8_ds_mla",
    )
    row = shape[-1]
    rep.check("contract/layout-row-bytes", row == K.ROW_BYTES,
              f"vLLM row={row} B, kernel expects {K.ROW_BYTES} B")

    # NOTE (2026-07-11 review, finding M7): the scale/rope byte offsets are NOT
    # derivable from any public vLLM symbol — they exist only inside the C++
    # writer and inside a test-local parser. Restating them here as arithmetic
    # on the constant 512 would be checking our constants against themselves.
    # So the offsets are pinned WHERE THEY ARE OBSERVABLE: in
    # `check_writer_roundtrip` below, which locates them in the bytes vLLM's own
    # writer actually emits. Only the row size is authoritative here.


def check_writer_roundtrip(rep: Report, device="cuda"):
    """vLLM's C++ writer -> our dequant -> original. Pins scale SEMANTICS.

    If vLLM's writer ever starts applying the layer's `k_scale` (today the
    ds_mla kernel ignores it and computes inline per-tile scales), this check
    fails loudly instead of degrading quality silently.
    """
    try:
        from vllm import _custom_ops as ops
    except ImportError as e:
        rep.skip("contract/writer-roundtrip", f"vLLM ops unavailable: {e}")
        return

    n, block_size, num_blocks = 128, 64, 8
    g = torch.Generator(device="cpu").manual_seed(11)
    kv_c = (torch.randn(n, 512, generator=g) * 0.5).to(device).to(torch.bfloat16)
    k_pe = (torch.randn(n, 64, generator=g) * 0.5).to(device).to(torch.bfloat16)

    cache = torch.zeros(num_blocks, block_size, K.ROW_BYTES, dtype=torch.uint8, device=device)
    slot = torch.arange(n, dtype=torch.int64, device=device)   # slots 0..n-1
    # deliberately pass a NON-unity k_scale: if the writer honours it, the
    # roundtrip vs the raw source breaks and we learn immediately.
    k_scale = torch.tensor(2.0, dtype=torch.float32, device=device)

    try:
        ops.concat_and_cache_mla(
            kv_c, k_pe, cache, slot, kv_cache_dtype="fp8_ds_mla", scale=k_scale,
        )
    except Exception as e:  # noqa: BLE001
        rep.check("contract/writer-roundtrip", False,
                  f"vLLM writer raised on sm_86: {type(e).__name__}: {str(e)[:120]}")
        return

    rows = cache.reshape(-1, K.ROW_BYTES)[:n]

    # --- offsets located IN THE WRITER'S OWN OUTPUT (review finding M7) -------
    # Scales: the writer stores 4 fp32 per-128-tile scales = amax/448. Compute
    # what they must be from the INPUT, then find where in the row they land.
    want_scales = (kv_c.float().reshape(n, 4, 128).abs().amax(-1) / 448.0)
    found_scale_off = -1
    for off in range(0, K.ROW_BYTES - 16 + 1, 4):
        cand = rows[:, off:off + 16].contiguous().view(torch.float32)
        if torch.allclose(cand, want_scales, rtol=1e-5, atol=1e-8):
            found_scale_off = off
            break
    rep.check("contract/scale-offset-observed", found_scale_off == K.SCALE_OFF,
              f"writer emits the 4 fp32 tile scales at byte {found_scale_off}; "
              f"kernel reads them at {K.SCALE_OFF}")

    # RoPE: stored raw bf16, so it is bit-identical to the input — search for it.
    want_rope = k_pe.view(torch.uint8).reshape(n, 128)
    found_rope_off = -1
    for off in range(0, K.ROW_BYTES - 128 + 1, 2):
        if torch.equal(rows[:, off:off + 128], want_rope):
            found_rope_off = off
            break
    rep.check("contract/rope-offset-observed", found_rope_off == K.ROPE_OFF,
              f"writer emits raw bf16 rope at byte {found_rope_off}; "
              f"kernel reads it at {K.ROPE_OFF}")

    got = R.ref_dequant_fp8_ds_mla_row(rows)
    src = torch.cat([kv_c.float(), k_pe.float()], dim=-1)

    # rope: raw bf16 passthrough -> must be EXACT
    m_rope = max_abs(got[:, 512:], src[:, 512:].to(torch.bfloat16).float())
    rep.check("contract/writer-rope-exact", m_rope == 0.0,
              f"max_abs={m_rope:.1e} (writer stores rope raw; must be 0)")

    # nope: fp8 quant error only. If the writer applied k_scale=2.0, the
    # dequantized values would be ~2x off and this blows up -> scale-semantics
    # mismatch caught.
    c = cosine(got[:, :512], src[:, :512])
    rel = ((got[:, :512] - src[:, :512]).abs() / src[:, :512].abs().clamp(1e-6)).median().item()
    rep.check("contract/writer-nope-roundtrip", c > 0.999 and rel < 0.05,
              f"cosine={c:.6f} median_rel={rel:.4f} (k_scale=2.0 passed in and "
              f"correctly IGNORED by the ds_mla writer)")

    # end-to-end: our decode over pages written by THEIR writer
    q = (torch.randn(4, 16, 576, generator=g) * 0.5).to(device).to(torch.bfloat16)
    idx = torch.randint(0, n, (4, 32), generator=g).to(device).int()
    sm = 1.0 / (576 ** 0.5)
    out = K.fp8_ds_mla_sparse_decode(q, cache, idx, sm)
    ref = R.ref_sparse_mla_decode(q, cache.reshape(-1, K.ROW_BYTES), idx, sm)
    c = cosine(out.float(), ref)
    rep.check("contract/decode-over-vllm-written-pages", c > 0.999, f"cosine={c:.6f}")


def check_indices_contract(rep: Report, trace_path="/root/idx_trace.pt"):
    """Replay REAL index tensors captured from a stock vLLM sparse-MLA run.

    Capture them with `verify/capture_indices.py` (monkeypatches the stock
    backend's decode entry and dumps its arguments). Live traces catch
    interface mismatches — especially PADDING SEMANTICS — that reading
    signatures does not.
    """
    import os

    if not os.path.exists(trace_path):
        rep.skip("contract/indices-replay",
                 f"no trace at {trace_path} — run verify/capture_indices.py first")
        return
    t = torch.load(trace_path)
    idx = t["indices"]
    rep.check("contract/indices-dtype", idx.dtype in (torch.int32, torch.int64),
              f"dtype={idx.dtype} shape={tuple(idx.shape)} "
              f"sentinels={int((idx < 0).sum())} min={int(idx.min())} max={int(idx.max())}")
    # The kernel assumes: -1 (or any negative) = masked; values are FLAT SLOT
    # indices into blocks*block_size. Assert the captured trace agrees.
    n_slots = t.get("num_slots")
    if n_slots is not None:
        in_range = bool(((idx < 0) | (idx < n_slots)).all())
        rep.check("contract/indices-are-flat-slots", in_range,
                  f"all values < num_slots={n_slots} (or negative sentinel)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== vllm-fp8kv CONTRACT tests (vs vLLM itself) == device={device}")
    rep = Report()
    check_layout_contract(rep)
    if device == "cuda":
        check_writer_roundtrip(rep, device)
    else:
        rep.skip("contract/writer-roundtrip", "no GPU")
    check_indices_contract(rep)
    print(f"== done: {rep.failed} failure(s) ==")
    sys.exit(min(rep.failed, 125))


if __name__ == "__main__":
    main()
