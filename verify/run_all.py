# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""One-command verification: `python verify/run_all.py`.

Model-free. Exercises the fp8_ds_mla decode kernel against fp32 golden
references over the shape matrix + the traps (int32 slot overflow, masked/-1
slots, LEADING-masked chunks -> NaN poisoning, empty rows, the real 3D paged
cache shape, LSE).

Exit code = number of failed checks.
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from verify import reference as R  # noqa: E402
from verify.metrics import Report, cosine, max_abs  # noqa: E402


def _rand_kv(n_slots, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(n_slots, 576, generator=g) * 0.5
    return R.quantize_to_fp8_ds_mla(kv.to(device)), kv.to(device)


def section_dequant(rep, device):
    """AC1 building block: fp8e4m3fn -> fp16 decode, all 256 bytes."""
    from vllm_fp8kv.fp8_dequant import build_lut, dequant_bitmath_torch

    b = torch.arange(256, dtype=torch.uint8)
    ref = b.view(torch.float8_e4m3fn).to(torch.float16)

    def bitexact(got):
        g, r = got.cpu(), ref
        if not torch.equal(g.isnan(), r.isnan()):
            return False
        return bool((((g.view(torch.int16) != r.view(torch.int16)) & ~r.isnan()).sum() == 0).item())

    rep.check("dequant/bitmath-256", bitexact(dequant_bitmath_torch(b)), "bit-exact (NaN-ness on 0x7F/0xFF)")
    rep.check("dequant/lut-256", bitexact(build_lut(torch.float16, device="cpu")[b.long()]), "bit-exact")

    if device == "cuda":
        from vllm_fp8kv.fp8_dequant import dequant_triton_launch
        out_bm, out_lut = dequant_triton_launch(b.cuda(), build_lut(torch.float16, device="cuda"))
        rep.check("dequant/triton-bitmath-256", bitexact(out_bm), "bit-exact on device")
        rep.check("dequant/triton-lut-256", bitexact(out_lut), "bit-exact on device")


def section_layout(rep, device):
    """The packer and the INDEPENDENT parser must agree — a symmetric bug can't hide."""
    packed, truth = _rand_kv(1024, seed=3, device=device)
    got = R.ref_dequant_fp8_ds_mla_row(packed)

    # RoPE is stored RAW bf16 — must be bit-exact, not merely close (trap #2)
    m_rope = max_abs(got[:, 512:], truth[:, 512:].to(torch.bfloat16).float())
    rep.check("layout/rope-raw-bf16-exact", m_rope == 0.0, f"max_abs={m_rope:.1e} (must be 0)")

    # NoPE: fp8 quant error only — e4m3 has 3 mantissa bits => rel err <= 2^-4
    rel = ((got[:, :512] - truth[:, :512]).abs() / truth[:, :512].abs().clamp(1e-6)).median().item()
    rep.check("layout/nope-fp8-quant-err", rel < 0.04, f"median_rel={rel:.4f}")

    # scales must land where vLLM's parser looks: fp32 view at element 128..132
    sc = packed.reshape(-1, 656)[:, 512:528].contiguous().view(torch.float32)
    rep.check("layout/scales-positive-finite", bool((sc > 0).all() and sc.isfinite().all()),
              f"4 tile scales/row, min={sc.min():.2e}")


def section_decode(rep, device):
    if device != "cuda":
        rep.skip("decode/*", "no GPU")
        return
    from vllm_fp8kv.fp8_ds_mla_sparse_decode import fp8_ds_mla_sparse_decode

    sm = 1.0 / (576 ** 0.5)
    # matrix: (seq_q, h_q, n_slots, topk)
    for seq_q, h_q, n_slots, topk in [
        (1, 16, 512, 64),        # single-token decode, small
        (4, 16, 4096, 256),      # batch
        (1, 128, 8192, 512),     # many heads (GLM: 128 q-heads before TP)
        (8, 16, 65536, 2048),    # GLM index_topk=2048 at 64K-slot pool
    ]:
        packed, _ = _rand_kv(n_slots, seed=seq_q + h_q, device=device)
        g = torch.Generator(device="cpu").manual_seed(topk)
        q = (torch.randn(seq_q, h_q, 576, generator=g) * 0.5).to(device).to(torch.bfloat16)
        # random slots, with -1 sentinels sprinkled in (masked slots trap)
        idx = torch.randint(0, n_slots, (seq_q, topk), generator=g).to(device).int()
        idx[:, ::7] = -1
        idx[0, : topk // 2] = -1   # a row that selects only half

        out = fp8_ds_mla_sparse_decode(q, packed, idx, sm)
        ref = R.ref_sparse_mla_decode(q, packed, idx, sm)
        m, c = max_abs(out.float(), ref), cosine(out.float(), ref)
        ok = c > 0.999 and m < 5e-2
        rep.check(f"decode/fp8-ds-mla-q{seq_q}h{h_q}s{n_slots}k{topk}", ok,
                  f"cosine={c:.6f} max_abs={m:.2e}")

    # trap: a row that selects NOTHING must yield zeros, not NaN
    packed, _ = _rand_kv(256, seed=99, device=device)
    q = torch.randn(2, 16, 576, device=device, dtype=torch.bfloat16) * 0.5
    idx = torch.full((2, 32), -1, dtype=torch.int32, device=device)
    out = fp8_ds_mla_sparse_decode(q, packed, idx, sm)
    rep.check("decode/empty-selection-is-zero", bool(out.isfinite().all() and (out == 0).all()),
              "all-masked rows -> zeros, no NaN")

    # trap: a LEADING run of -1 (fully-masked chunks before any valid one) is
    # what makes exp2(-inf - -inf) = NaN and poisons acc permanently. Upstream
    # hits this (DCP emits interleaved -1s); the naive online-softmax does NOT
    # survive it. Separate check because a NaN here is invisible in cosine.
    packed, _ = _rand_kv(512, seed=13, device=device)
    q = torch.randn(1, 16, 576, device=device, dtype=torch.bfloat16) * 0.5
    idx = torch.full((1, 64), -1, dtype=torch.int32, device=device)
    idx[0, 48:] = torch.arange(16, dtype=torch.int32, device=device)   # valid only at the END
    out = fp8_ds_mla_sparse_decode(q, packed, idx, sm)
    ref = R.ref_sparse_mla_decode(q, packed, idx, sm)
    finite = bool(out.isfinite().all())
    c = cosine(out.float(), ref) if finite else 0.0
    rep.check("decode/leading-masked-chunks-no-nan", finite and c > 0.999,
              f"finite={finite} cosine={c:.6f} (48 leading -1s then valid slots)")

    # the real cache is 3D [num_blocks, block_size, 656] — exercise the reshape
    # path, not just the flat view the other tests pass
    packed3d = packed.reshape(8, 64, 656)
    idx = torch.randint(0, 512, (1, 32), dtype=torch.int32, device=device)
    out3 = fp8_ds_mla_sparse_decode(q, packed3d, idx, sm)
    ref3 = R.ref_sparse_mla_decode(q, packed, idx, sm)
    c = cosine(out3.float(), ref3)
    rep.check("decode/paged-3d-cache-shape", c > 0.999, f"cosine={c:.6f} ([8,64,656] pool)")

    # LSE: never exercised before — the WRITE_LSE branch did not even compile
    out4, lse = fp8_ds_mla_sparse_decode(q, packed, idx, sm, return_lse=True)
    kv = R.ref_dequant_fp8_ds_mla_row(packed).float()
    sel = idx[0][(idx[0] >= 0)].long()
    scores = (q[0].float() @ kv[sel].T) * sm
    lse_ref = torch.logsumexp(scores, dim=-1)
    m = max_abs(lse[0], lse_ref)
    rep.check("decode/lse-matches-logsumexp", m < 5e-2, f"max_abs={m:.2e} (natural-log LSE)")


def section_bigpool(rep, device):
    """int32 slot-offset overflow: slot*656 exceeds 2^31 past ~3.27M slots.
    dsa-3090 shipped this bug once (illegal access at capacity scale)."""
    if device != "cuda":
        rep.skip("bigpool/int64-offsets", "no GPU")
        return
    from vllm_fp8kv.fp8_ds_mla_sparse_decode import fp8_ds_mla_sparse_decode

    # the earlier sections leave several GB resident; this pool is 2.3 GB and
    # must be allocated on a clean allocator or it OOMs for the wrong reason.
    torch.cuda.empty_cache()
    n_slots = 3_500_000                       # 3.5M * 656 = 2.30e9 > 2^31
    free, _ = torch.cuda.mem_get_info()
    if free < 3.5e9:
        rep.skip("bigpool/int64-offsets", f"needs ~2.4GB free, have {free/1e9:.1f}GB")
        return
    kv = torch.zeros(n_slots, 656, dtype=torch.uint8, device=device)
    small, _ = _rand_kv(2, seed=7, device=device)
    kv[0] = small.reshape(-1, 656)[0]
    kv[-1] = small.reshape(-1, 656)[1]        # the row past the int32 boundary

    q = torch.randn(1, 16, 576, device=device, dtype=torch.bfloat16) * 0.5
    idx = torch.tensor([[0, n_slots - 1]], dtype=torch.int32, device=device)
    out = fp8_ds_mla_sparse_decode(q, kv, idx, 1.0 / (576 ** 0.5))
    ref = R.ref_sparse_mla_decode(q, kv, idx, 1.0 / (576 ** 0.5))
    c = cosine(out.float(), ref)
    del kv
    torch.cuda.empty_cache()
    rep.check("bigpool/int64-offsets-3.5M-slots", c > 0.999, f"cosine={c:.6f} (int32 would fault/corrupt)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== vllm-fp8kv verification == device={device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"   {p.name} sm_{p.major}{p.minor}  smem/block={p.shared_memory_per_block_optin/1024:.0f}KB")
    rep = Report()
    section_dequant(rep, device)
    section_layout(rep, device)
    section_decode(rep, device)
    if device == "cuda":
        torch.cuda.empty_cache()
    section_bigpool(rep, device)
    print(f"== done: {rep.failed} failure(s) ==")
    sys.exit(min(rep.failed, 125))


if __name__ == "__main__":
    main()
