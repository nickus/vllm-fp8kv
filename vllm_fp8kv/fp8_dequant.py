# Copyright 2026 Nick / sm86-dsa contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from the DSA port in renning22/glm-5.2-4090 (Apache-2.0).
"""fp8e4m3fn -> fp16/bf16 software decode for GPUs without hardware FP8 (sm_86).

The KV / indexer caches stay raw fp8e4m3fn *bytes* (byte-per-element layout
untouched); kernels load them as uint8 and decode in-register. Two variants:

**Bit-math** — the fp8 payload dropped into the fp16 bit layout is off by a
fixed power of two, for normals and denormals alike:

    fp16( (s<<15) | ((b & 0x7F) << 7) ) * 2^8  ==  fp8e4m3fn(b)   [b not NaN]

The fp8 exponent (4 bits, bias 7) lands in the low 4 bits of the fp16 exponent
field (5 bits, bias 15), so the value comes out scaled by 2^(7-15)*2^(10-... )
= 2^-8 uniformly — one exact fp16 multiply by 256 fixes it. fp8 denormals land
as fp16 denormals and rescale exactly under the same multiply (fp16 arithmetic
has full denormal support in CUDA hardware). fp8e4m3fn has no inf; 0x7F/0xFF
are NaN and need an explicit select — torch's cast yields sign-preserved
0x7F80/0xFF80, which we reproduce for bit-exactness.

**LUT** — a 256-entry table built once with torch's own cast, gathered
in-kernel. Immune to any arithmetic quirk by construction; costs a (usually
L1-resident) extra load.

Both are verified bit-exact against `torch.float8_e4m3fn -> float16` over all
256 byte values (AC1), including denormals, +-0 and both NaN encodings.
"""

import torch

FP16_NAN_MAG = 0x7F80  # magnitude bits of the NaN torch produces for 0x7F/0xFF


def dequant_bitmath_torch(u8: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Pure-PyTorch reference of the bit-math decode. Works on CPU and GPU.

    `u8` is a uint8 tensor of raw fp8e4m3fn bytes; returns `dtype` (fp16/bf16).
    """
    assert u8.dtype == torch.uint8
    b = u8.to(torch.int16)
    sgn = (b & 0x80) << 8
    h = (sgn | ((b & 0x7F) << 7)).view(torch.float16) * 256.0
    nan = (sgn | FP16_NAN_MAG).view(torch.float16)
    out = torch.where((b & 0x7F) == 0x7F, nan, h)
    return out if dtype == torch.float16 else out.to(dtype)


def build_lut(dtype: torch.dtype = torch.float16, device="cuda") -> torch.Tensor:
    """256-entry decode table: lut[b] = fp8e4m3fn(b) as `dtype`.

    Built with torch's own software cast, so it is exact by construction.
    Keep one per device/dtype alive for the lifetime of the process and pass
    its pointer to kernels using `dequant_lut_triton`.
    """
    return (
        torch.arange(256, dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .to(dtype)
        .contiguous()
    )


try:
    import triton  # noqa: F401
    import triton.language as tl

    @triton.jit
    def dequant_bitmath_triton(u8):
        """fp8e4m3fn bytes (any int tensor holding 0..255) -> fp16, in-register.

        Bit-exact vs torch's cast for all 256 values incl. NaN (0x7F/0xFF).
        """
        b = u8.to(tl.uint16)
        sgn = (b & 0x80) << 8
        mag = (b & 0x7F) << 7
        h = (sgn | mag).to(tl.float16, bitcast=True) * 256.0
        nan = (sgn | 0x7F80).to(tl.float16, bitcast=True)
        return tl.where((b & 0x7F) == 0x7F, nan, h.to(tl.float16))

    @triton.jit
    def dequant_lut_triton(u8, lut_ptr):
        """fp8e4m3fn bytes -> lut dtype via 256-entry table gather."""
        return tl.load(lut_ptr + u8.to(tl.int32))

    @triton.jit
    def _dequant_test_kernel(in_ptr, out_bm_ptr, out_lut_ptr, lut_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        b = tl.load(in_ptr + offs, mask=m, other=0)
        tl.store(out_bm_ptr + offs, dequant_bitmath_triton(b), mask=m)
        tl.store(out_lut_ptr + offs, dequant_lut_triton(b, lut_ptr), mask=m)

    def dequant_triton_launch(u8: torch.Tensor, lut: torch.Tensor):
        """Test/bench driver: run both device-side decoders over `u8`."""
        n = u8.numel()
        out_bm = torch.empty(n, dtype=torch.float16, device=u8.device)
        out_lut = torch.empty(n, dtype=lut.dtype, device=u8.device)
        grid = (triton.cdiv(n, 256),)
        _dequant_test_kernel[grid](u8.reshape(-1), out_bm, out_lut, lut, n, BLOCK=256)
        return out_bm, out_lut

except ImportError:  # CPU-only box (Phase A): torch reference + LUT builder still usable
    pass
