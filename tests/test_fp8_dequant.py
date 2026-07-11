# Copyright 2026 Nick / sm86-dsa contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""AC1: fp8e4m3fn -> fp16 decode must match torch's cast bit-exactly for all
256 byte values — normals, denormals, +-0, max finite (+-448), NaN (0x7F/0xFF).
"""

import pytest
import torch

from vllm_fp8kv.fp8_dequant import build_lut, dequant_bitmath_torch

ALL_BYTES = torch.arange(256, dtype=torch.uint8)


def _ref(dtype):
    return ALL_BYTES.view(torch.float8_e4m3fn).to(dtype)


def assert_bit_exact(got: torch.Tensor, ref: torch.Tensor):
    """Bit-exact on all non-NaN values; NaN positions must be NaN on both sides.

    NaN *payload* bits are excluded on purpose: torch's own fp8e4m3fn->fp16
    cast emits 0x7F80/0xFF80 (sign-preserved) on CPU but 0x7FFF (canonical,
    sign-stripped) on CUDA — there is no single bit pattern to match, and any
    NaN propagates identically in arithmetic. Our bit-math emits the CPU-style
    encoding; a GPU-built LUT contains the CUDA-style one.
    """
    ibits = torch.int16 if ref.dtype in (torch.float16, torch.bfloat16) else torch.int32
    gf, rf = got.cpu(), ref.cpu()
    nan_ok = torch.equal(gf.isnan(), rf.isnan())
    assert nan_ok, "NaN positions differ"
    g, r = gf.view(ibits), rf.view(ibits)
    bad = ((g != r) & ~rf.isnan()).nonzero().flatten()
    assert bad.numel() == 0, (
        f"{bad.numel()} mismatches; first at byte 0x{bad[0].item():02X}: "
        f"got 0x{g[bad[0]].item() & 0xFFFF:04X}, want 0x{r[bad[0]].item() & 0xFFFF:04X}"
    )


def test_bitmath_torch_all256_fp16():
    assert_bit_exact(dequant_bitmath_torch(ALL_BYTES), _ref(torch.float16))


def test_bitmath_torch_all256_bf16():
    assert_bit_exact(dequant_bitmath_torch(ALL_BYTES, torch.bfloat16), _ref(torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lut_all256_cpu(dtype):
    assert_bit_exact(build_lut(dtype, device="cpu")[ALL_BYTES.long()], _ref(dtype))


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@needs_cuda
def test_triton_both_variants_all256():
    from vllm_fp8kv.fp8_dequant import dequant_triton_launch

    u8 = ALL_BYTES.cuda()
    lut = build_lut(torch.float16, device="cuda")
    out_bm, out_lut = dequant_triton_launch(u8, lut)
    ref = _ref(torch.float16)
    assert_bit_exact(out_bm, ref)   # bit-math incl. denormal-under-multiply behavior
    assert_bit_exact(out_lut, ref)  # LUT gather


@needs_cuda
def test_triton_lut_bf16_all256():
    from vllm_fp8kv.fp8_dequant import dequant_triton_launch

    u8 = ALL_BYTES.cuda()
    lut = build_lut(torch.bfloat16, device="cuda")
    _, out_lut = dequant_triton_launch(u8, lut)
    assert_bit_exact(out_lut, _ref(torch.bfloat16))


@needs_cuda
def test_triton_bitmath_large_random():
    """Denormal/NaN bytes scattered through a big buffer — catches any
    block/masking bug the 256-value sweep can't see."""
    from vllm_fp8kv.fp8_dequant import dequant_triton_launch

    g = torch.Generator(device="cpu").manual_seed(0)
    u8 = torch.randint(0, 256, (1 << 20,), generator=g, dtype=torch.uint8).cuda()
    lut = build_lut(torch.float16, device="cuda")
    out_bm, out_lut = dequant_triton_launch(u8, lut)
    ref = u8.view(torch.float8_e4m3fn).to(torch.float16)
    assert_bit_exact(out_bm, ref)
    assert_bit_exact(out_lut, ref)
