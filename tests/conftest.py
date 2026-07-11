# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Test config: run the Triton kernels on CPU when there is no GPU.

`TRITON_INTERPRET=1` executes @triton.jit bodies in Python/numpy, so the kernel
MATH is testable without hardware (only performance needs a GPU). It must be set
before triton is imported, hence conftest.

Interpreter limitation, discovered the hard way and worth knowing: it stores
bf16 as raw uint16 and only converts on explicit bf16<->fp32 casts
(triton/runtime/interpreter.py:188). `tl.dot` on bf16 operands therefore
computes on garbage. Kernel tests here consequently drive the decode with fp32
q (the kernel accepts fp32 precisely so it can be checked without a GPU); the
bf16 production path is covered by the GPU-gated tests.
"""

import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def device():
    return DEVICE


@pytest.fixture
def kv_pool():
    """Build a small fp8_ds_mla pool + its pre-quantization source.

    Returns (packed_u8 [n_slots, 656], src_f32 [n_slots, 576]).
    """
    from verify.reference import quantize_to_fp8_ds_mla

    def _build(n_slots=64, seed=0, scale=0.5):
        g = torch.Generator(device="cpu").manual_seed(seed)
        src = (torch.randn(n_slots, 576, generator=g) * scale).to(DEVICE)
        packed = quantize_to_fp8_ds_mla(src).contiguous()
        return packed, src.float()

    return _build
