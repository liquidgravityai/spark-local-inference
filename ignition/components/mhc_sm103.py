# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 mHC composition with ordered residual mixing and FP32 coefficients.

Residual products accumulate before the branch addition and BF16 store. Large
ordinary calls project through BF16 high/low weight halves with FP32 accumulation.
Small ordinary calls use an explicitly precise cuBLAS handle and ATen-ordered
input statistics to preserve collapse rounding.
FlashInfer forms the post and combination coefficients, and weighted RMSNorm
consumes the collapsed BF16 input.
"""

from __future__ import annotations

import ctypes
import itertools
import math
import weakref

import torch
from torch.utils.weak import WeakIdKeyDictionary
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


SPLITS = 4
_BLOCK_T = 128
_BLOCK_K = 64
# Chained Blackwell dots need a single stage to preserve fixed-input replay.
_CONFIG = triton.Config({}, num_warps=8, num_stages=1)
# Larger ordinary calls use tensor cores; small calls preserve FP32 projection rounding.
MIN_ROWS = 256
_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_POST_SCRATCH: dict[tuple, torch.Tensor] = {}
_NORM_SCRATCH: dict[tuple, torch.Tensor] = {}
_SPLIT_WEIGHTS = WeakIdKeyDictionary()
_EXACT_SCRATCH: dict[tuple, tuple[torch.Tensor, ...]] = {}
_CUBLAS_OWNERS = weakref.WeakValueDictionary()
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


# FlashInfer's mHC launches build on first use, so compiled forwards see the composition as one operator
# that reads its inputs and writes the residual, mixes and normalized input.
@torch.library.custom_op("flare::sm103_composed_mhc", mutates_args=("residual", "post", "comb", "normalized"))
def _composed(inputs: list[torch.Tensor], residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
              normalized: torch.Tensor, handle: int) -> None:
    _OWNERS[handle].run(*inputs)


@_composed.register_fake
def _fake_composed(inputs, residual, post, comb, normalized, handle) -> None:
    pass


@triton.jit
def _ordered_post(Output, Source, Residual, Post, Comb, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    h = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = tl.arange(0, 4)
    valid = h < HIDDEN
    source = tl.load(Source + row * HIDDEN + h, valid, other=0).to(tl.float32)
    r0 = tl.load(Residual + (row * 4 + 0) * HIDDEN + h, valid, other=0).to(tl.float32)
    r1 = tl.load(Residual + (row * 4 + 1) * HIDDEN + h, valid, other=0).to(tl.float32)
    r2 = tl.load(Residual + (row * 4 + 2) * HIDDEN + h, valid, other=0).to(tl.float32)
    r3 = tl.load(Residual + (row * 4 + 3) * HIDDEN + h, valid, other=0).to(tl.float32)
    c0 = tl.load(Comb + row * 16 + j)
    c1 = tl.load(Comb + row * 16 + 4 + j)
    c2 = tl.load(Comb + row * 16 + 8 + j)
    c3 = tl.load(Comb + row * 16 + 12 + j)
    # Round each product and residual sum before adding the branch, then store BF16.
    mixed = c0[:, None] * r0[None, :] + c1[:, None] * r1[None, :]
    mixed = mixed + c2[:, None] * r2[None, :]
    mixed = mixed + c3[:, None] * r3[None, :]
    branch = tl.load(Post + row * 4 + j)[:, None] * source[None, :]
    tl.store(Output + (row * 4 + j[:, None]) * HIDDEN + h[None, :],
             mixed + branch, valid[None, :])


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _project(Residual, FnHigh, FnLow, Dot, Sqrsum, rows, K: tl.constexpr, SPLITS: tl.constexpr,
             BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, HAS_LOW: tl.constexpr = True):
    tile, split = tl.program_id(0), tl.program_id(1)
    tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    live = tokens < rows
    mixes = tl.arange(0, 32)
    dot = tl.zeros((BLOCK_T, 32), tl.float32)
    squares = tl.zeros((BLOCK_T,), tl.float32)
    span = K // SPLITS
    for start in range(split * span, (split + 1) * span, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        values = tl.load(Residual + tokens.to(tl.int64)[:, None] * K + k[None, :], mask=live[:, None], other=0.0)
        dot += tl.dot(values, tl.trans(tl.load(FnHigh + mixes[:, None] * K + k[None, :])))
        if HAS_LOW:
            dot += tl.dot(values, tl.trans(tl.load(FnLow + mixes[:, None] * K + k[None, :])))
        wide = values.to(tl.float32)
        squares += tl.sum(wide * wide, 1)
    tl.store(Dot + (split * rows + tokens.to(tl.int64))[:, None] * 24 + mixes[None, :], dot,
             mask=live[:, None] & (mixes[None, :] < 24))
    tl.store(Sqrsum + split * rows + tokens, squares, mask=live)


def _split_weights(fn):
    """BF16 high and low halves of the FP32 projection, padded to 32 mixes; shared by every call on ``fn``."""
    if fn not in _SPLIT_WEIGHTS:
        padded = torch.zeros((32, fn.shape[1]), dtype=torch.float32, device=fn.device)
        padded[:24] = fn
        high = padded.to(torch.bfloat16)
        _SPLIT_WEIGHTS[fn] = (high, (padded - high.float()).to(torch.bfloat16))
    return _SPLIT_WEIGHTS[fn]


class OrderedMHCPost:
    """Prepared residual mixing with products summed before the branch addition."""

    def __init__(self, source, residual, post, comb, output):
        from ignition.preparation.triton import prepare_triton

        if (residual.ndim != 3 or residual.shape[0] <= 0 or residual.shape[1] != 4
                or residual.shape[2] not in (4096, 5120, 7168)):
            raise ValueError("SM103 mHC residual mixing requires [tokens, 4, hidden] with a supported hidden size")
        rows, _, hidden = residual.shape
        if (source.shape != (rows, hidden) or output.shape != residual.shape
                or post.shape != (rows, 4) or comb.shape != (rows, 4, 4)
                or any(value.dtype != torch.bfloat16 for value in (source, residual, output))
                or any(value.dtype != torch.float32 for value in (post, comb))
                or residual.device.type != "cuda"
                or any(value.device != residual.device or not value.is_contiguous()
                       for value in (source, residual, post, comb, output))):
            raise ValueError("SM103 mHC residual mixing requires contiguous BF16 activations and FP32 mixes on one CUDA device")
        begin, end = output.data_ptr(), output.data_ptr() + output.nbytes
        if any(begin < value.data_ptr() + value.nbytes and value.data_ptr() < end
               for value in (source, residual, post, comb)):
            raise ValueError("SM103 mHC residual mixing requires output storage disjoint from its inputs")
        self.output = output
        self.launch = prepare_triton(_ordered_post, (output, source, residual, post, comb),
            (triton.cdiv(hidden, 256), rows, 1), HIDDEN=hidden, BLOCK=256,
            num_warps=4, enable_fp_fusion=False)

    def __call__(self, source, residual, post, comb):
        from ignition.preparation.triton import launch_triton

        launch_triton(self.launch.kernel, self.launch.compiled,
            (self.output, source, residual, post, comb), self.launch.grid, **self.launch.options)
        return self.output


def mixed_residual_scratch(residual):
    """Share the serial post/pre intermediate without aliasing b12x pre's output."""
    key = (residual.device, *residual.shape)
    if key not in _POST_SCRATCH:
        _POST_SCRATCH[key] = torch.empty_like(residual)
    return _POST_SCRATCH[key]


def collapsed_scratch(residual):
    """Share serial BF16 collapse storage disjoint from normalized outputs."""
    rows, _, hidden = residual.shape
    key = (residual.device, rows, hidden)
    if key not in _NORM_SCRATCH:
        _NORM_SCRATCH[key] = torch.empty((rows, hidden), device=residual.device, dtype=torch.bfloat16)
    return _NORM_SCRATCH[key]


class ComposedMHC:
    """Post/pre or pre into ``scratch`` (residual, post, comb, normalized) for one prefill capacity.

    Layers run serially, so every call of one capacity on a device shares its logits, square sums
    and collapsed input.
    """

    def __init__(self, operation, weights, scratch, *, rms_eps, hc_eps, sinkhorn_iters, norm_eps):
        from flashinfer.mhc import _mhc_pre_big_fuse_impl
        from ignition.components.mhc_norm_sm103 import prepare as prepare_norm

        self.fuse = _mhc_pre_big_fuse_impl

        rows, _, hidden = scratch.residual.shape
        if operation not in ("pre", "post_pre") or weights.fn.shape != (24, 4 * hidden):
            raise ValueError("composed mHC serves pre and post/pre with 24 mixes")
        self.operation, self.weights, self.scratch = operation, weights, scratch
        if operation == "post_pre":
            self.mix_grid = (triton.cdiv(hidden, 512), rows, 1)
            self.mix = _ordered_post.warmup(scratch.residual, scratch.normalized,
                scratch.residual, scratch.post, scratch.comb, HIDDEN=hidden, BLOCK=512,
                grid=self.mix_grid, num_warps=4, enable_fp_fusion=False)
        self.eps = (float(rms_eps), float(hc_eps), int(sinkhorn_iters), float(norm_eps))
        self.high, self.low = _split_weights(weights.fn)
        self.has_low = bool(torch.any(self.low != 0).item())
        # An exactly zero low half needs no second dot; that single-dot loop can pipeline.
        self.projection_stages = _CONFIG.num_stages if self.has_low else 3
        key = (scratch.residual.device, rows, hidden)
        if key not in _SCRATCH:
            device = scratch.residual.device
            _SCRATCH[key] = (torch.empty((SPLITS, rows, 24), dtype=torch.float32, device=device),
                             torch.empty((SPLITS, rows), dtype=torch.float32, device=device),
                             torch.empty((rows, hidden), dtype=torch.bfloat16, device=device))
        self.dot, self.squares, self.collapsed = _SCRATCH[key]
        self.norm = prepare_norm(self.collapsed, weights.norm, scratch.normalized, eps=norm_eps)
        flat = scratch.residual.view(rows, -1)
        self.args = (flat, self.high, self.low, self.dot, self.squares, rows)
        self.options = {"K": 4 * hidden, "SPLITS": SPLITS, "BLOCK_T": _BLOCK_T,
                        "BLOCK_K": _BLOCK_K, "HAS_LOW": self.has_low}
        self.grid = (triton.cdiv(rows, _BLOCK_T), SPLITS, 1)
        self.compiled = _project.fn.warmup(*self.args, grid=self.grid, num_warps=_CONFIG.num_warps,
                                           num_stages=self.projection_stages, **self.options)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, *inputs):
        s = self.scratch
        _composed(list(inputs), s.residual, s.post, s.comb, s.normalized, self.handle)
        return s.normalized

    def run(self, *inputs):
        s, w = self.scratch, self.weights
        rms_eps, hc_eps, iterations, norm_eps = self.eps
        if self.operation == "post_pre":
            source, residual, post, comb = inputs
            self.mix[self.mix_grid](s.residual, source, residual, post, comb)
        else:
            s.residual.copy_(inputs[0])
        self.compiled[self.grid](*self.args)
        self.fuse(s.post, s.comb, self.collapsed, self.dot, self.squares, s.residual, w.scale, w.base,
                               s.residual.shape[1] * s.residual.shape[2], rms_eps, hc_eps, hc_eps, 2.0, iterations,
                               SPLITS, 0)
        self.norm()


@triton.jit
def _fp32_statistics(Source, Cast, Squares, Rstd, K: tl.constexpr,
                     X: tl.constexpr, Y: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, 512)
    channel = tl.arange(0, 4)
    accum = tl.full((512, 4), 0, tl.float32)
    # ATen's vectorized mean keeps four independent sums per lane, then reduces X before Y.
    for chunk in range(K // 2048):
        columns = index[:, None] * 4 + channel[None, :] + chunk * 2048
        values = tl.load(Source + row * K + columns).to(tl.float32)
        tl.store(Cast + row * K + columns, values)
        accum = accum + values * values
    even, odd = tl.split(tl.reshape(accum, (512, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    values = tl.reshape(((a0 + a1) + a2) + a3, (Y, X))
    x = tl.arange(0, X)
    for level in tl.static_range(9):
        offset = X >> (level + 1)
        if offset > 0:
            other = tl.gather(values,
                tl.broadcast_to(tl.minimum(x[None, :] + offset, X - 1), (Y, X)), axis=1)
            values = tl.where(x[None, :] < offset, values + other, 0.)
    partial = tl.sum(values, axis=1)
    y = tl.arange(0, Y)
    for level in tl.static_range(4):
        offset = Y >> (level + 1)
        if offset > 0:
            other = tl.gather(partial, tl.minimum(y + offset, Y - 1), axis=0)
            partial = tl.where(y < offset, partial + other, 0.)
    total = tl.sum(partial, axis=0)
    tl.store(Squares + row, total)
    tl.store(Rstd + row, libdevice.rsqrt(total * (1. / K) + EPS))



class _FP32Blas:
    """One explicitly precise cuBLAS handle and workspace for serial device preparation."""

    def __init__(self, device):
        from cuda.pathfinder import load_nvidia_dynamic_lib

        self.device = device
        self.library = ctypes.CDLL(load_nvidia_dynamic_lib("cublas").abs_path)
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        signatures = {
            "cublasCreate_v2": [ctypes.POINTER(pointer)],
            "cublasDestroy_v2": [pointer],
            "cublasSetMathMode": [pointer, integer],
            "cublasSetStream_v2": [pointer, pointer],
            "cublasSetWorkspace_v2": [pointer, pointer, ctypes.c_size_t],
            "cublasGemmEx": [pointer, *([integer] * 5), pointer, pointer, integer, integer,
                             pointer, integer, integer, pointer, pointer, integer, integer,
                             integer, integer],
        }
        for name, signature in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = signature, integer
        self.handle = pointer()
        with torch.cuda.device(device):
            self.call("cublasCreate_v2", ctypes.byref(self.handle))
            self.finalizer = weakref.finalize(self, self.library.cublasDestroy_v2, self.handle)
            self.call("cublasSetMathMode", self.handle, 2)  # CUBLAS_PEDANTIC_MATH
            self.workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.alpha, self.beta = ctypes.c_float(1), ctypes.c_float(0)

    def call(self, name, *arguments):
        status = getattr(self.library, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed with cuBLAS status {status}")

    def project(self, source, weight, output):
        rows, width = source.shape
        mixes = weight.shape[0]
        self.call("cublasSetStream_v2", self.handle, torch.cuda.current_stream(self.device).cuda_stream)
        # Setting the stream resets cuBLAS workspace; bind caller-owned storage afterward.
        self.call("cublasSetWorkspace_v2", self.handle, self.workspace.data_ptr(), self.workspace.nbytes)
        self.call("cublasGemmEx", self.handle, 1, 0, mixes, rows, width,
                  ctypes.byref(self.alpha), weight.data_ptr(), 0, width,
                  source.data_ptr(), 0, width, ctypes.byref(self.beta), output.data_ptr(),
                  0, mixes, 69, -1)  # CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT


class ExactMHC:
    """FP32 projection and ATen-ordered input statistics for small ordinary mHC calls."""

    def __init__(self, operation, inputs, weights, scratch, *, rms_eps, hc_eps, sinkhorn_iters, norm_eps):
        from b12x.norm.mhc import _impl
        from flashinfer.mhc import _mhc_pre_big_fuse_impl
        from ignition.components.mhc_norm_sm103 import prepare_projected
        from ignition.preparation.triton import prepare_triton

        if type(sinkhorn_iters) is not int or sinkhorn_iters != 20:
            raise ValueError("mHC production kernels require 20 Sinkhorn iterations")
        if rms_eps not in (1e-20, 1e-6, 1e-5) or hc_eps != 1e-6:
            raise ValueError("mHC epsilon values differ from the production numerical contract")
        if not math.isfinite(norm_eps) or norm_eps < 0:
            raise ValueError("normalization requires a finite nonnegative epsilon")
        residual = inputs[0 if operation == "pre" else 1]
        rows, hidden, _ = _impl._validate_pre_inputs(
            residual, weights.fn, weights.scale, weights.base, expanded=True)
        _impl._validate_mhc_binding_views(partials=None, y=scratch.normalized,
            post=scratch.post, comb=scratch.comb, out=scratch.residual,
            tokens=rows, hidden_size=hidden, split_k=1, dtype=torch.bfloat16, device=residual.device)
        self.operation, self.weights, self.scratch = operation, weights, scratch
        self.eps = float(rms_eps), float(hc_eps), int(sinkhorn_iters)
        self.fuse = _mhc_pre_big_fuse_impl
        if operation == "post_pre":
            self.post = OrderedMHCPost(*inputs, scratch.residual)
        key = scratch.residual.device, rows, hidden
        if key not in _EXACT_SCRATCH:
            def empty(shape, dtype=torch.float32):
                return torch.empty(shape, dtype=dtype, device=scratch.residual.device)
            _EXACT_SCRATCH[key] = (empty((rows, 4 * hidden)), empty((rows, 24)),
                                  empty((rows,)), empty((rows,)),
                                  empty((rows, hidden), torch.bfloat16))
        self.cast, self.dot, self.squares, self.rstd, self.collapsed = _EXACT_SCRATCH[key]
        self.blas = _CUBLAS_OWNERS.get(scratch.residual.device)
        if self.blas is None:
            self.blas = _FP32Blas(scratch.residual.device)
            _CUBLAS_OWNERS[scratch.residual.device] = self.blas
        y = min(1 << (rows.bit_length() - 1), 16)
        self.statistics = prepare_triton(_fp32_statistics,
            (scratch.residual, self.cast, self.squares, self.rstd), (rows, 1, 1),
            K=4 * hidden, X=512 // y, Y=y, EPS=float(rms_eps),
            num_warps=8 if rows == 1 and hidden == 4096 else 16, enable_fp_fusion=False)
        self.norm = prepare_projected(scratch.residual, weights.norm, scratch.normalized,
            self.dot, self.rstd, weights.scale, weights.base, eps=norm_eps, hc_eps=hc_eps)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, *inputs):
        scratch = self.scratch
        _composed(list(inputs), scratch.residual, scratch.post, scratch.comb, scratch.normalized, self.handle)
        return scratch.normalized

    def run(self, *inputs):
        scratch, weights = self.scratch, self.weights
        rms_eps, hc_eps, iterations = self.eps
        if self.operation == "post_pre":
            self.post(*inputs)
        else:
            scratch.residual.copy_(inputs[0])
        self.statistics()
        self.blas.project(self.cast, weights.fn, self.dot)
        self.fuse(scratch.post, scratch.comb, self.collapsed, self.dot, self.squares,
                  scratch.residual, weights.scale, weights.base, self.cast.shape[1],
                  rms_eps, hc_eps, hc_eps, 2.0, iterations, 1, 0)
        self.norm()
