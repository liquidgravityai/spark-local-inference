# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Songlin Yang, Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed ordinary KDA recurrence with FP32 persistent state.

The recurrence follows FLA's fused recurrent delta update as adapted by SGLang's
packed KDA decode. Query offsets and device live counts use Ignition's compressed
ordinary-decode metadata; an empty request never aliases the next packed token.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache

from .native_operator import native_contract, native_plan


@triton.jit
def _packed_decode(QKV, Gate, Beta, ALog, Bias, State, Starts, Indices, Sequences, Tokens, Output,
                   QKV_STRIDE: tl.constexpr, GATE_STRIDE: tl.constexpr, BETA_STRIDE: tl.constexpr,
                   STATE_STRIDE: tl.constexpr, HEADS: tl.constexpr, BV: tl.constexpr):
    value_tile, request, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if request >= tl.load(Sequences):
        return
    token = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    if token >= end or token >= tl.load(Tokens):
        return
    key = tl.arange(0, 128)
    value = value_tile * BV + tl.arange(0, BV)
    out = Output + (token * HEADS + head) * 128 + value
    slot = tl.load(Indices + request).to(tl.int64)
    if slot <= 0:
        tl.store(out, tl.full((BV,), 0, tl.float32))
        return
    q = tl.load(QKV + token * QKV_STRIDE + head * 128 + key).to(tl.float32)
    k = tl.load(QKV + token * QKV_STRIDE + (HEADS + head) * 128 + key).to(tl.float32)
    v = tl.load(QKV + token * QKV_STRIDE + (2 * HEADS + head) * 128 + value).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    q = q * 0.08838834764831845
    a = tl.load(Gate + token * GATE_STRIDE + head * 128 + key).to(tl.float32)
    bias = tl.load(Bias + head * 128 + key).to(tl.float32)
    decay = -5.0 * tl.sigmoid(tl.exp(tl.load(ALog + head).to(tl.float32)) * (a + bias))
    beta = tl.sigmoid(tl.load(Beta + token * BETA_STRIDE + head).to(tl.float32))
    pointer = State + slot * STATE_STRIDE + head * 128 * 128 + value[:, None] * 128 + key[None, :]
    h = tl.load(pointer).to(tl.float32)
    h *= tl.exp(decay)[None, :]
    v -= tl.sum(h * k[None, :], 1)
    v *= beta
    h += v[:, None] * k[None, :]
    result = tl.sum(h * q[None, :], 1)
    tl.store(out, result.to(tl.bfloat16))
    tl.store(pointer, h)


OPERANDS = ("mixed_qkv", "raw_g", "raw_beta", "A_log", "dt_bias", "recurrent_state",
            "query_start_loc", "state_indices", "num_seqs", "num_tokens", "raw_output")


@dataclass(frozen=True)
class Query:
    operands: tuple
    heads: int
    sequences: int


CONTRACT = native_contract(component_id="flare.kda_packed_decode", query_type=Query,
                           tactics=lambda query: (32, 16))


def _launch_arguments(tensors, query, tile):
    qkv, gate, beta, _, _, state = tensors[:6]
    options = dict(QKV_STRIDE=qkv.stride(0), GATE_STRIDE=gate.stride(0), BETA_STRIDE=beta.stride(0),
                   STATE_STRIDE=state.stride(0), HEADS=query.heads, BV=tile, num_warps=1, num_stages=3)
    return options, (128 // tile, query.sequences, query.heads)


@program_cache(scope="preparation")
def compile_decode(payload, tile, ordinal):
    query = Query(**dict(payload))
    tensors = tuple(torch.empty_strided(shape, stride, dtype=getattr(torch, dtype), device="meta")
                    for shape, stride, dtype in query.operands)
    options, grid = _launch_arguments(tensors, query, tile)
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(_packed_decode, grid, *tensors, **options)


@dataclass(frozen=True)
class State:
    query: Query
    tile: int
    program: object

    def run(self, tensors):
        options, grid = _launch_arguments(tensors, self.query, self.tile)
        self.program[grid](*tensors, *(value for key, value in options.items()
                                     if key not in ("num_warps", "num_stages")))


def recurrence_plan(arguments):
    """Declare the raw recurrence; callers compose and race its BF16 output norm."""
    tensors = tuple(arguments[name] for name in OPERANDS)
    qkv, gate, beta, a_log, bias, state, starts, indices, sequences, tokens, output = tensors
    heads = state.shape[1]
    device = qkv.device
    if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 3)
            or heads not in (16, 32, 64) or state.shape[2:] != (128, 128)
            or state.dtype != torch.float32 or state.stride()[1:] != (16384, 128, 1)
            or indices.ndim != 2 or indices.shape[1] != 1 or not indices.is_contiguous()
            or qkv.shape != (output.shape[0], 3 * heads * 128)
            or gate.shape != (output.shape[0], heads, 128)
            or beta.shape != (output.shape[0], heads) or output.shape[1:] != (heads, 128)
            or a_log.shape != (heads,) or bias.shape != (heads, 128)
            or any(t.dtype != torch.bfloat16 for t in (qkv, gate, beta, output))
            or any(t.dtype != torch.float32 for t in (a_log, bias))
            or any(t.device != device for t in tensors)
            or starts.shape != (indices.shape[0] + 1,)
            or sequences.shape != (1,) or tokens.shape != (1,)
            or any(t.dtype != torch.int32 for t in (starts, sequences, tokens))
            or indices.dtype != torch.int64
            or not all(t.is_contiguous() for t in (a_log, bias, output, starts, sequences, tokens))
            or qkv.stride(1) != 1 or gate.stride()[1:] != (128, 1) or beta.stride(1) != 1):
        raise ValueError("packed KDA decode requires SM103 ordinary packed BF16 inputs, FP32 state and parameters")
    query = Query(tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype).removeprefix("torch."))
                        for t in tensors), heads, indices.shape[0])
    payload = CONTRACT.encode_query(query)

    def materialize(tile):
        program = compile_decode(payload, tile, device.index)
        load_programs(program)
        return attach_programs(State(query, tile, program), program)

    return native_plan(contract=CONTRACT, query=query, device=device, scratch_bytes=lambda tile: 0,
        materialize=materialize, compile_jobs=lambda tile, detected: (CompileJob.create(
            "ignition.components.kda_decode_sm103:compile_decode", payload, tile, detected.ordinal),))
