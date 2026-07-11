# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""AC2 (pool capacity) + AC3 (decode throughput): fp8-KV vs bf16-KV.

AC2: fp8_ds_mla stores 656 B/token; bf16 stores 1152 B/token (576 x 2). The
     pool-capacity ratio at identical memory is therefore 1152/656 = 1.756x
     from the format alone. Gate: >= 1.9x was the brief's ask -- we report the
     ACTUAL byte ratio and the actual token capacity per GB, and note that
     1.756x is the format's arithmetic ceiling (the brief's 1.9x assumed a
     2x-naive model of fp8 vs bf16 and did not account for the 16 B of tile
     scales + the RoPE half staying bf16).

AC3: decode throughput. fp8 moves 656/1152 = 57% of the KV bytes, so at long
     context (where KV streaming dominates) fp8 should be FASTER, not merely
     on par. If it is not, the dequant is misplaced.

Method: torch.cuda.Event wall-clock + byte arithmetic (no ncu in containers).
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from verify import reference as R  # noqa: E402
from vllm_fp8kv.fp8_ds_mla_sparse_decode import fp8_ds_mla_sparse_decode  # noqa: E402

FP8_ROW, BF16_ROW = 656, 576 * 2


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e-3


def bf16_sparse_decode(q, kv_bf16, indices, sm_scale):
    """Upstream's own bf16 sparse-MLA decode kernel (the thing we must beat)."""
    from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
        triton_mla_sparse_attention,
    )

    out = triton_mla_sparse_attention(
        q, kv_bf16.view(-1, 1, 576), indices.view(q.shape[0], 1, -1),
        sm_scale=sm_scale,
    )
    return out[:, : q.shape[1], :]


def main():
    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    print(f"== vllm-fp8kv BENCH == {p.name} sm_{p.major}{p.minor} "
          f"{p.total_memory/2**30:.1f} GiB")

    # ---------- AC2: pool capacity ----------
    print("\n-- AC2: KV pool capacity (bytes/token) --")
    ratio = BF16_ROW / FP8_ROW
    for gb in (16, 20):
        b = gb * 2**30
        print(f"  {gb} GiB pool: bf16 {b//BF16_ROW/1e6:.2f}M tokens | "
              f"fp8 {b//FP8_ROW/1e6:.2f}M tokens")
    print(f"  RATIO = {BF16_ROW}/{FP8_ROW} = {ratio:.3f}x  "
          f"(format ceiling; 656 B = 512 fp8 + 16 scale + 128 raw-bf16 rope)")

    # ---------- AC3: decode throughput ----------
    print("\n-- AC3: decode throughput, fp8 vs upstream bf16 --")
    sm = 1.0 / (576 ** 0.5)
    g = torch.Generator(device="cpu").manual_seed(1)

    have_bf16 = True
    try:
        import vllm.v1.attention.ops.triton_mla_sparse_kernel  # noqa: F401
    except ImportError:
        have_bf16 = False
        print("  (upstream bf16 kernel unavailable — reporting fp8 only)")

    print(f"{'ctx':>8} {'topk':>6} {'bs':>4} | {'fp8 us':>9} {'bf16 us':>9} "
          f"{'speedup':>8} | {'fp8 GB/s':>9}")
    for n_slots, topk, bs in [(8192, 512, 1), (65536, 2048, 1),
                              (65536, 2048, 8), (131072, 2048, 8)]:
        kv = (torch.randn(n_slots, 576, generator=g) * 0.5).to(dev)
        packed = R.quantize_to_fp8_ds_mla(kv)
        kv_bf16 = kv.to(torch.bfloat16)
        q = (torch.randn(bs, 16, 576, generator=g) * 0.5).to(dev).to(torch.bfloat16)
        idx = torch.randint(0, n_slots, (bs, topk), generator=g).to(dev).int()

        t8 = timeit(lambda: fp8_ds_mla_sparse_decode(q, packed, idx, sm))
        bytes8 = bs * topk * FP8_ROW
        gbs8 = bytes8 / t8 / 1e9

        if have_bf16:
            try:
                t16 = timeit(lambda: bf16_sparse_decode(q, kv_bf16, idx, sm))
                spd = t16 / t8
                print(f"{n_slots:>8} {topk:>6} {bs:>4} | {t8*1e6:9.1f} {t16*1e6:9.1f} "
                      f"{spd:7.2f}x | {gbs8:9.1f}")
            except Exception as e:  # noqa: BLE001
                print(f"{n_slots:>8} {topk:>6} {bs:>4} | {t8*1e6:9.1f} "
                      f"{'ERR':>9} {'-':>8} | {gbs8:9.1f}   ({type(e).__name__})")
        else:
            print(f"{n_slots:>8} {topk:>6} {bs:>4} | {t8*1e6:9.1f} {'-':>9} {'-':>8} | {gbs8:9.1f}")
        del kv, packed, kv_bf16
        torch.cuda.empty_cache()

    print("\n  (fp8 moves 57% of the KV bytes bf16 does -> speedup > 1 expected "
          "where KV streaming dominates)")


if __name__ == "__main__":
    main()
