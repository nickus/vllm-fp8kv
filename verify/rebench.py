#!/usr/bin/env python3
"""Honest re-bench: fp8 vs bf16 INSIDE upstream's patched sparse-MLA kernel.

Replaces the retracted numbers. Every timing here comes from
`triton_mla_sparse_attention` — upstream's own dispatcher, its split-KV kernel,
its LSE merge — with only the KV dtype differing. No scratch kernel, no
hand-rolled reduction.

It also tests the RFC's open hypothesis: are upstream's autotune CONFIG LISTS
(_FINAL: BLOCK_N=16 only; _SPLIT: BLOCK_N=32 only) simply bf16-shaped, so that
fp8's 656 B rows + a scale load want bigger blocks / more warps?

Correctness is gated on EVERY timed configuration (cosine AND max_abs) — the
missing max_abs gate is exactly how an 8x-wrong kernel passed last time.
"""
import argparse
import itertools
import sys

import torch
import triton

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from verify.metrics import cosine, max_abs                       # noqa: E402
from verify.reference import (                                    # noqa: E402
    quantize_to_fp8_ds_mla,
    ref_sparse_mla_decode,
)

import vllm.v1.attention.ops.triton_mla_sparse_kernel as K       # noqa: E402

DEV = "cuda"
SM = 1.0 / (576 ** 0.5)


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3      # microseconds


def make(bs, n_slots, topk, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv = (torch.randn(n_slots, 576, generator=g) * 0.5).to(DEV)
    packed = quantize_to_fp8_ds_mla(kv).contiguous()             # [n, 656] uint8
    kv_bf16 = kv.to(torch.bfloat16).reshape(n_slots, 1, 576).contiguous()
    q = (torch.randn(bs, 128, 576, generator=g) * 0.5).to(DEV).to(torch.bfloat16)
    idx = torch.randint(0, n_slots, (bs, 1, topk), generator=g).to(DEV).int()
    return q, packed, kv_bf16, idx


def call(q, kv, idx):
    out = K.triton_mla_sparse_attention(q, kv, idx, SM)
    return out[0] if isinstance(out, tuple) else out


def correctness(q, packed, idx):
    got = call(q, packed.reshape(-1, 1, 656), idx).float()
    ref = ref_sparse_mla_decode(q, packed, idx[:, 0, :], SM)
    return cosine(got, ref), max_abs(got, ref)


def bench_matrix(shapes):
    print(f"\n{'ctx':>7} {'bs':>4} {'topk':>5} | {'fp8 us':>8} {'bf16 us':>8} | "
          f"{'fp8/bf16':>8} | correctness (cos / max_abs)")
    print("-" * 82)
    rows = []
    for n_slots, bs, topk in shapes:
        q, packed, kv_bf16, idx = make(bs, n_slots, topk)
        cos, mad = correctness(q, packed, idx)
        ok = cos > 0.999 and mad < 5e-2
        t8 = timeit(lambda: call(q, packed.reshape(-1, 1, 656), idx))
        t16 = timeit(lambda: call(q, kv_bf16, idx))
        rows.append((n_slots, bs, topk, t8, t16, t16 / t8, cos, mad, ok))
        flag = "" if ok else "   <-- WRONG, timing meaningless"
        print(f"{n_slots:>7} {bs:>4} {topk:>5} | {t8:>8.1f} {t16:>8.1f} | "
              f"{t16/t8:>7.2f}x | {cos:.6f} / {mad:.2e}{flag}")
        del q, packed, kv_bf16, idx
        torch.cuda.empty_cache()
    return rows


def sweep_configs(bs=32, n_slots=65536, topk=2048):
    """Is fp8 slow because upstream's config lists are bf16-shaped?

    Upstream ships BLOCK_N=16 (final) / 32 (split) only. Force each config and
    see whether a bigger block or more warps rescues fp8 -- and whether bf16
    would benefit from the same, which would mean it is not a dtype story at all.
    """
    print(f"\n== config sweep @ bs={bs} topk={topk} pool={n_slots} ==")
    q, packed, kv_bf16, idx = make(bs, n_slots, topk, seed=1)
    kv8 = packed.reshape(-1, 1, 656)

    print(f"{'BLOCK_N':>8} {'warps':>6} {'stages':>7} | {'fp8 us':>8} {'bf16 us':>8} | "
          f"{'ratio':>6} | fp8 correctness")
    print("-" * 76)
    best = {}
    for bn, nw, ns in itertools.product((16, 32, 64, 128), (4, 8), (2, 3)):
        cfgs = [triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)]
        # override BOTH autotuned entry points with this single config
        old_final = K._sparse_mla_kernel_final.configs
        old_split = K._sparse_mla_kernel_split.configs
        K._sparse_mla_kernel_final.configs = cfgs
        K._sparse_mla_kernel_final.cache.clear()
        K._sparse_mla_kernel_split.configs = cfgs
        K._sparse_mla_kernel_split.cache.clear()
        try:
            got = call(q, kv8, idx).float()
            ref = ref_sparse_mla_decode(q, packed, idx[:, 0, :], SM)
            cos, mad = cosine(got, ref), max_abs(got, ref)
            ok = cos > 0.999 and mad < 5e-2
            t8 = timeit(lambda: call(q, kv8, idx), iters=30)
            t16 = timeit(lambda: call(q, kv_bf16, idx), iters=30)
            status = f"{cos:.6f} / {mad:.1e}" + ("" if ok else "  WRONG")
            print(f"{bn:>8} {nw:>6} {ns:>7} | {t8:>8.1f} {t16:>8.1f} | "
                  f"{t16/t8:>5.2f}x | {status}")
            if ok:
                best[(bn, nw, ns)] = (t8, t16)
        except Exception as e:                                   # noqa: BLE001
            print(f"{bn:>8} {nw:>6} {ns:>7} | {'-':>8} {'-':>8} | "
                  f"{'-':>6} | {type(e).__name__}: {str(e)[:40]}")
        finally:
            K._sparse_mla_kernel_final.configs = old_final
            K._sparse_mla_kernel_split.configs = old_split
            K._sparse_mla_kernel_final.cache.clear()
            K._sparse_mla_kernel_split.cache.clear()

    if best:
        bf8 = min(best.items(), key=lambda kv: kv[1][0])
        bf16 = min(best.items(), key=lambda kv: kv[1][1])
        print(f"\n  best fp8 : BLOCK_N={bf8[0][0]} warps={bf8[0][1]} stages={bf8[0][2]}"
              f" -> {bf8[1][0]:.1f} us")
        print(f"  best bf16: BLOCK_N={bf16[0][0]} warps={bf16[0][1]} stages={bf16[0][2]}"
              f" -> {bf16[1][1]:.1f} us")
        print(f"  fp8 (best cfg) / bf16 (best cfg) = "
              f"{bf16[1][1] / bf8[1][0]:.2f}x")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"== re-bench on {p.name} sm_{p.major}{p.minor} ==")
    src = __import__("inspect").getsource(K.triton_mla_sparse_attention)
    assert "is_fp8 = kv.dtype" in src, "RUNTIME KERNEL IS NOT PATCHED"
    print("   runtime kernel: PATCHED (is_fp8 dispatch present)")

    # AC3's real axis is CONTEXT LENGTH, which the old bench never swept.
    # (pool_slots, batch, topk)
    print("\n### AC3 as written: context-length sweep (topk=2048, bs=1) ###")
    bench_matrix([
        (8_192, 1, 2048),        # 8K
        (16_384, 1, 2048),
        (65_536, 1, 2048),       # 64K
        (131_072, 1, 2048),      # 128K
    ])

    print("\n### batch sweep @ 64K pool (what the retracted table measured) ###")
    bench_matrix([
        (65_536, 1, 2048),
        (65_536, 8, 2048),
        (65_536, 32, 2048),
        (65_536, 64, 2048),
    ])

    if a.sweep:
        sweep_configs()


if __name__ == "__main__":
    main()
