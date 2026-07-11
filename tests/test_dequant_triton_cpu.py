# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The DEVICE-side fp8 decoders (bit-math and LUT), run on CPU.

`tests/test_fp8_dequant.py` covers the torch reference and gates the Triton
variants on a GPU. Under the interpreter they run on CPU too — and these decode
paths are pure integer/fp16 work, which the interpreter models exactly. So the
kernel-side decode is CI-verifiable, not just GPU-verifiable.
"""

import torch

from tests.test_fp8_dequant import ALL_BYTES, assert_bit_exact
from vllm_fp8kv.fp8_dequant import build_lut, dequant_triton_launch


def _ref(dtype):
    return ALL_BYTES.view(torch.float8_e4m3fn).to(dtype)


def test_triton_bitmath_and_lut_all_256_bytes():
    lut = build_lut(torch.float16, device="cpu")
    out_bm, out_lut = dequant_triton_launch(ALL_BYTES, lut)

    assert_bit_exact(out_bm, _ref(torch.float16))    # 7-op bit-math, in-register
    assert_bit_exact(out_lut, _ref(torch.float16))   # 256-entry table gather


def test_triton_lut_bf16_table():
    lut = build_lut(torch.bfloat16, device="cpu")
    _, out_lut = dequant_triton_launch(ALL_BYTES, lut)
    assert_bit_exact(out_lut, _ref(torch.bfloat16))


def test_triton_decode_masks_the_tail_of_a_ragged_buffer():
    """n is not a multiple of BLOCK=256: the tail must be masked, not read OOB."""
    g = torch.Generator(device="cpu").manual_seed(2)
    u8 = torch.randint(0, 256, (300,), generator=g, dtype=torch.uint8)
    lut = build_lut(torch.float16, device="cpu")

    out_bm, out_lut = dequant_triton_launch(u8, lut)
    ref = u8.view(torch.float8_e4m3fn).to(torch.float16)
    assert out_bm.shape == (300,)
    assert_bit_exact(out_bm, ref)
    assert_bit_exact(out_lut, ref)


def test_denormals_and_signed_zero_survive_the_bitmath():
    """The subtle half of the decode: fp8 denormals land as fp16 denormals and
    must rescale exactly under the x256 multiply; -0.0 must stay -0.0."""
    lut = build_lut(torch.float16, device="cpu")
    denormals = torch.tensor([0x01, 0x02, 0x07, 0x80, 0x81, 0x87], dtype=torch.uint8)

    out_bm, _ = dequant_triton_launch(denormals, lut)
    ref = denormals.view(torch.float8_e4m3fn).to(torch.float16)
    assert_bit_exact(out_bm, ref)
    assert torch.signbit(out_bm[3]), "-0.0 lost its sign"
