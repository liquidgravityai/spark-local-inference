# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared mHC RMSNorm preserving Torch CUDA reduction order.

The four independent per-lane accumulators and descending reduction follow
the FP32 mean kernel in Torch 2.14.1. The optional contraction preserves each
FP32 product and sum before converting the collapsed value to BF16.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _load_collapsed(Source, p0, p1, p2, p3, row, column, H: tl.constexpr,
                    COLLAPSE: tl.constexpr):
    if COLLAPSE:
        a = tl.load(Source + (row * 4 + 0) * H + column).to(tl.float32)
        b = tl.load(Source + (row * 4 + 1) * H + column).to(tl.float32)
        c = tl.load(Source + (row * 4 + 2) * H + column).to(tl.float32)
        d = tl.load(Source + (row * 4 + 3) * H + column).to(tl.float32)
        value = ((a * p0 + b * p1) + c * p2) + d * p3
        return value.to(tl.bfloat16).to(tl.float32)
    else:
        return tl.load(Source + row * H + column).to(tl.float32)


@triton.jit
def _normalize(Source, p0, p1, p2, p3, Weight, Output,
               H: tl.constexpr, W: tl.constexpr, EPS: tl.constexpr,
               COLLAPSE: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, W)
    component = tl.arange(0, 4)
    accumulators = tl.full((W, 4), 0, tl.float32)
    # Keep each of ATen's four accumulation chains independent while loading contiguous vectors.
    for start in range(triton.cdiv(H, 4 * W)):
        column = 4 * lane[:, None] + component[None, :] + start * 4 * W
        valid = column < H
        safe = tl.where(valid, column, 0)
        values = _load_collapsed(Source, p0, p1, p2, p3, row, safe, H, COLLAPSE)
        accumulators = accumulators + tl.where(valid, values * values, 0.)
    even, odd = tl.split(tl.reshape(accumulators, (W, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    partial = ((a0 + a1) + a2) + a3
    # Spell out the entire tree so Triton cannot choose a different inter-warp order.
    for level in tl.static_range(0, 9):
        offset = W >> (level + 1)
        if offset > 0:
            other = tl.gather(partial, tl.minimum(lane + offset, W - 1), axis=0)
            partial = tl.where(lane < offset, partial + other, 0.)
    total = tl.sum(partial, axis=0)
    mean = total * (1.0 / H)
    denominator = mean + EPS
    inverse = libdevice.rsqrt(denominator)
    column = tl.arange(0, triton.next_power_of_2(H))
    valid = column < H
    safe = tl.where(valid, column, 0)
    value = _load_collapsed(Source, p0, p1, p2, p3, row, safe, H, COLLAPSE)
    weight = tl.load(Weight + safe).to(tl.float32)
    result = (value * inverse) * weight
    tl.store(Output + row * H + column, result, valid)


@triton.jit
def _aten_norm(Source, Coefficients, Weight, Output,
               H: tl.constexpr, W: tl.constexpr, EPS: tl.constexpr,
               COLLAPSE: tl.constexpr):
    row = tl.program_id(0)
    if COLLAPSE:
        p0 = tl.load(Coefficients + row * 4)
        p1 = tl.load(Coefficients + row * 4 + 1)
        p2 = tl.load(Coefficients + row * 4 + 2)
        p3 = tl.load(Coefficients + row * 4 + 3)
    else:
        p0 = 0.
        p1 = 0.
        p2 = 0.
        p3 = 0.
    _normalize(Source, p0, p1, p2, p3, Weight, Output, H, W, EPS, COLLAPSE)


@triton.jit
def _projected_norm(Source, Dot, Rstd, Scale, Bias, Weight, Output,
                    H: tl.constexpr, W: tl.constexpr, EPS: tl.constexpr, HC_EPS: tl.constexpr):
    row = tl.program_id(0)
    inverse = tl.load(Rstd + row)
    scale = tl.load(Scale)
    z0 = (tl.load(Dot + row * 24) * inverse) * scale + tl.load(Bias)
    z1 = (tl.load(Dot + row * 24 + 1) * inverse) * scale + tl.load(Bias + 1)
    z2 = (tl.load(Dot + row * 24 + 2) * inverse) * scale + tl.load(Bias + 2)
    z3 = (tl.load(Dot + row * 24 + 3) * inverse) * scale + tl.load(Bias + 3)
    p0 = tl.div_rn(1., 1. + libdevice.exp(-z0)) + HC_EPS
    p1 = tl.div_rn(1., 1. + libdevice.exp(-z1)) + HC_EPS
    p2 = tl.div_rn(1., 1. + libdevice.exp(-z2)) + HC_EPS
    p3 = tl.div_rn(1., 1. + libdevice.exp(-z3)) + HC_EPS
    _normalize(Source, p0, p1, p2, p3, Weight, Output, H, W, EPS, True)


def _validate(source, weight, output, coefficients, collapse):
    from b12x.norm.mhc._alias import overlaps

    if output.ndim != 2:
        raise ValueError("SM103 mHC normalization requires a BF16 output matrix")
    rows, hidden = output.shape
    shape = (rows, 4, hidden) if collapse else (rows, hidden)
    if (hidden not in (4096, 5120, 7168) or rows < 1 or source.shape != shape
            or weight.shape != (hidden,) or output.dtype != torch.bfloat16
            or source.dtype != torch.bfloat16 or weight.dtype not in (torch.bfloat16, torch.float32)
            or source.device.type != "cuda"
            or any(not value.is_contiguous() or value.device != source.device
                   for value in (source, weight, output))
            or (coefficients is not None and (coefficients.shape != (rows, 4)
                or coefficients.dtype != torch.float32 or coefficients.device != source.device
                or not coefficients.is_contiguous()))):
        raise ValueError("SM103 mHC normalization requires contiguous BF16 activations and BF16/FP32 weights on one CUDA device")
    if any(overlaps(output, value) for value in (source, weight, coefficients) if value is not None):
        raise ValueError("SM103 mHC normalization output must not overlap its inputs")
    return rows, hidden


def prepare(source, weight, output, *, eps, coefficients=None):
    """Compile fixed-shape normalization with optional four-stream contraction."""
    from ignition.preparation.triton import prepare_triton

    rows, hidden = _validate(source, weight, output, coefficients, coefficients is not None)
    block_height = min(1 << (rows.bit_length() - 1), 16)
    width = 512 // block_height
    return prepare_triton(_aten_norm, (source, coefficients, weight, output),
        (rows, 1, 1), H=hidden, W=width, EPS=float(eps),
        COLLAPSE=coefficients is not None,
        num_warps=4, enable_fp_fusion=False)


def prepare_projected(source, weight, output, dot, rstd, scale, bias, *, eps, hc_eps):
    """Fuse exact FP32 pre coefficients into ordered collapse and normalization."""
    from b12x.norm.mhc._alias import overlaps
    from ignition.preparation.triton import prepare_triton

    rows, hidden = _validate(source, weight, output, None, True)
    for value, shape in ((dot, (rows, 24)), (rstd, (rows,)), (scale, (3,)), (bias, (24,))):
        if (value.shape != shape or value.dtype != torch.float32 or value.device != source.device
                or not value.is_contiguous()):
            raise ValueError("SM103 mHC projected normalization requires contiguous FP32 projection statistics and weights")
        if overlaps(output, value):
            raise ValueError("SM103 mHC normalization output must not overlap its inputs")
    width = 512 // min(1 << (rows.bit_length() - 1), 16)
    return prepare_triton(_projected_norm, (source, dot, rstd, scale, bias, weight, output),
        (rows, 1, 1), H=hidden, W=width, EPS=float(eps), HC_EPS=float(hc_eps),
        num_warps=4, enable_fp_fusion=False)
