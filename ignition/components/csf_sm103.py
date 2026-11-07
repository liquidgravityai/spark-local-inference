# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare compact SM103 expert scales for direct trtllm-gen expansion."""

import torch


def prepare_trtllm_csf_decoder(scales13, scales2, out13, out2):
    """Encode row-permuted E4M3 bytes and prepare fixed-output expansion.

    Inputs are logical row-major uint8 scale bytes, already in the FC1 and FC2
    row order required by trtllm-gen. The encoder and payload conversion run
    during preparation; the decoder owns compact data and compiled programs.
    Output buffers must remain alive and disjoint until their consumers finish.
    """
    from b12x._lib.quant._csf_prepare import prepare_csf_payload
    from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane
    from b12x._lib.quant.csf_encode import encode_csf_source
    from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
    from flashinfer.fp4_quantization import block_scale_interleave

    scales = (scales13, scales2)
    if any(value.dtype != torch.uint8 or value.ndim != 3 or value.device.type != "cuda"
           or not value.is_contiguous() or value.shape[0] == 0 or value.shape[1] == 0 or value.shape[1] % 128
           or value.shape[2] == 0 or value.shape[2] % 4 for value in scales):
        raise ValueError("SM103 compact scales require contiguous CUDA uint8 [experts, 128-aligned rows, 4-aligned columns]")
    if scales13.device != scales2.device or scales13.shape[0] != scales2.shape[0]:
        raise ValueError("SM103 compact scales require one device and equal expert counts")
    # Encoding reads both sources before decoding writes either output. Reject
    # overlap before allocating or launching any preparation kernels.
    for source in scales:
        source_begin, source_end = source.data_ptr(), source.data_ptr() + source.nbytes
        for output in (out13, out2):
            if output.device == source.device:
                begin, end = output.data_ptr(), output.data_ptr() + output.nbytes
                if source_begin < end and begin < source_end:
                    raise ValueError("SM103 compact scale outputs must not overlap source storage")
    planes = tuple(Nvfp4CompactBlockPlane(prepare_csf_payload(
        encode_csf_source(value, quantization="nvfp4"), destination=0)) for value in scales)
    decoder = Nvfp4CsfDecoder.prepare(*planes, out13, out2)
    decoder.decode_all(out13, out2)
    for source, output in zip(scales, (out13, out2), strict=True):
        for start in range(0, source.shape[0], 8):
            part = source[start:start + 8]
            expected = block_scale_interleave(part).reshape_as(part)
            actual = output[start:start + part.shape[0]].view(torch.uint8)
            if not torch.equal(actual, expected):
                raise ValueError("SM103 compact scale expansion differs from native scale bytes")
    return decoder


def _expansion_arguments(decoder, routes, schedule):
    from b12x._lib.quant.csf_blocks_tuning import (CsfExpansionConfig, CsfExpansionQuery,
                                                  _validate_config, _validate_query)

    groups, shared, bitmap, direct = schedule
    query = CsfExpansionQuery(decoder.first.geometry[:7] + decoder.first.geometry[9:],
                              decoder.second.geometry[:7] + decoder.second.geometry[9:],
                              routes, "int32")
    config = CsfExpansionConfig(groups, shared, bitmap, direct)
    _validate_query(query, None)
    _validate_config(query, config, None)
    return ({"first": query.first, "second": query.second, "routes": query.routes,
             "ids_dtype": query.ids_dtype}, config.__dict__)


def decoder_compile_job(decoder, routes, schedule, ordinal):
    from b12x._lib.compile_pool import CompileJob

    query, config = _expansion_arguments(decoder, routes, schedule)
    return CompileJob.create("b12x._lib.quant.csf_blocks_tuning:compile_expansion",
                             query, config, ordinal)


def prepare_tuned_decoder(decoder, routes, schedule):
    from dataclasses import replace
    from b12x._lib.quant.csf_blocks_tuning import compile_expansion

    query, config = _expansion_arguments(decoder, routes, schedule)
    ordinal = decoder.first.fixed.device.index
    programs, routing, _ = compile_expansion(query, config, ordinal)
    groups, shared, bitmap, direct = schedule
    return replace(decoder,
                   first=replace(decoder.first, groups_per_task=groups, shared_exceptions=shared),
                   second=replace(decoder.second, groups_per_task=groups, shared_exceptions=shared),
                   programs=programs, routing_programs=routing,
                   bitmap_routes=bitmap, direct_routes=direct)
