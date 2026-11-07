"""Capture qualification is opt-in and independent of model/mHC correctness."""

import os
import pytest

from ignition.cli import _parser


def test_parallax_preparation_flag_is_explicit():
    args = _parser().parse_args(["export", "glm-5.3-flash", "--no-mtp", "--parallax-routing"])
    assert args.parallax_routing and args.no_mtp


def test_capture_metadata_is_read_only_and_unknown_graph_input_rejected():
    from ignition.preparation.execution import GraphSignature
    with pytest.raises(ValueError, match="reserved"):
        GraphSignature("decode", 1, 1, ("parallax_routes_v1",))


def test_traced_forward_shares_operators_without_instrumenting_plain_forward():
    from types import SimpleNamespace as NS
    from ignition.recipes.glm53.generation import PreparedGeneration
    from ignition.recipes.glm53.stack import PreparedDecoder
    from ignition.recipes.glm53.weights import MoEWeights

    calls = []

    class Trace:
        def counter(self, layer, ids, valid):
            calls.append((layer, ids, valid))
            return object()

    decoder = PreparedDecoder.__new__(PreparedDecoder)
    decoder.routing_trace = None
    decoder.metadata = NS(recurrent=NS(num_tokens=object()))
    decoder.state, decoder.scratch, decoder.embedding = object(), object(), object()
    decoder.weights = NS(layers=[NS(mlp=object()), NS(mlp=MoEWeights.__new__(MoEWeights))])
    decoder.layers = [NS(attention=object(), mlp=object()),
        NS(attention=object(), mlp=NS(routing_counter=None, scratch=NS(router=NS(ids=object())),
                                     prepared_operator=object()))]
    plain = PreparedGeneration.__new__(PreparedGeneration)
    plain.decoder, plain.batch, plain.completion = decoder, object(), object()
    trace = Trace()
    traced = plain.with_routing_trace(trace)
    assert traced is not plain and traced.decoder is not decoder
    assert traced.batch is plain.batch and traced.completion is plain.completion
    for name in ("metadata", "state", "scratch", "embedding", "weights"):
        assert getattr(traced.decoder, name) is getattr(decoder, name)
    assert traced.decoder.layers[0] is decoder.layers[0]
    original, instrumented = decoder.layers[1], traced.decoder.layers[1]
    assert instrumented is not original and instrumented.mlp is not original.mlp
    assert instrumented.attention is original.attention
    assert instrumented.mlp.prepared_operator is original.mlp.prepared_operator
    assert instrumented.mlp.scratch is original.mlp.scratch
    assert original.mlp.routing_counter is None and decoder.routing_trace is None
    assert instrumented.mlp.routing_counter is not None and traced.decoder.routing_trace is trace
    assert calls == [(1, original.mlp.scratch.router.ids, decoder.metadata.recurrent.num_tokens)]
    with pytest.raises(ValueError, match="untraced"):
        traced.with_routing_trace(trace)

    from ignition.recipes.glm53.speculative import PreparedPrefill, PreparedRound

    prefill = PreparedPrefill.__new__(PreparedPrefill)
    prefill.generation, prefill.absorb, prefill.feedback = plain, object(), object()
    mtp_prefill = prefill.with_routing_trace(trace)
    assert mtp_prefill.generation is not plain
    assert mtp_prefill.absorb is prefill.absorb and mtp_prefill.feedback is prefill.feedback
    assert prefill.generation.decoder.routing_trace is None

    operation = PreparedRound.__new__(PreparedRound)
    operation.decoder = decoder
    for name in ("owner", "operations", "draft", "absorb", "metadata", "kda_commit"):
        setattr(operation, name, object())
    mtp_round = operation.with_routing_trace(trace)
    assert mtp_round.decoder is not decoder
    assert mtp_round.decoder.routing_trace is trace and decoder.routing_trace is None
    for name in ("owner", "operations", "draft", "absorb", "metadata", "kda_commit"):
        assert getattr(mtp_round, name) is getattr(operation, name)


@pytest.mark.parametrize("layers,experts,tokens,top_k", [(45, 288, 32, 8), (45, 288, 8192, 8), (3, 5, 4, 2)])
def test_route_geometry_and_valid_token_metadata(layers, experts, tokens, top_k):
    import torch
    from ignition.components.routing_trace import validate_output, validate_counter

    output = torch.empty((2, layers, 4 + experts), dtype=torch.int64, device="meta")
    frame = torch.empty(1, dtype=torch.int32, device="meta")
    sequence = torch.empty(1, dtype=torch.int64, device="meta")
    ids = torch.empty((tokens, top_k), dtype=torch.int32, device="meta")
    valid = torch.empty(1, dtype=torch.int32, device="meta")
    validate_output(output, frame, sequence)
    validate_counter(output, layers - 1, ids, valid)
    for bad in (valid.to(torch.int64), valid.reshape(()), valid.expand(2), torch.empty(1, dtype=torch.int32)):
        with pytest.raises(ValueError, match="valid-token"):
            validate_counter(output, layers - 1, ids, bad)
    with pytest.raises(ValueError, match="canonical layer"):
        validate_counter(output, layers, ids, valid)
    with pytest.raises(ValueError, match="contiguous"):
        validate_counter(output, 0, ids[:, ::2], valid)
    with pytest.raises(ValueError, match="bounded"):
        validate_output(output[:1], frame, sequence)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicitly selected idle test GPU")
@pytest.mark.parametrize("compiled", [False, True])
def test_captured_route_counts_preserve_ids_and_alternating_frame_lifetimes(compiled):
    import torch
    from ignition.components.routing_trace import RoutingTrace

    device = "cuda:0"
    output = torch.full((2, 3, 4 + 5), -77, device=device, dtype=torch.int64)
    frame = torch.zeros(1, device=device, dtype=torch.int32)
    sequence = torch.ones(1, device=device, dtype=torch.int64)
    ids = torch.tensor([[0, 1], [1, 4], [3, 2], [4, 4]], device=device, dtype=torch.int32)
    valid = torch.tensor([2], device=device, dtype=torch.int32)
    original = ids.clone()
    trace = RoutingTrace(output, frame, sequence)
    counter = trace.counter(2, ids, valid)
    def forward():
        trace.begin()
        counter()
        return output

    execute = torch.compile(forward, fullgraph=True, options={"triton.cudagraphs": False}) if compiled else forward
    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    for serial in range(1, 7):
        selected = serial % 2
        frame.fill_(selected)
        sequence.fill_(serial)
        before = output[1 - selected].clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ids, original)
        assert torch.equal(output[1 - selected], before)
        row = output[selected, 2].cpu().tolist()
        assert row == [serial, 2, 2, 0, 1, 2, 0, 0, 1]
        assert output[selected, 0].cpu().tolist() == [serial, 0, 0, 0, 0, 0, 0, 0, 0]
    valid.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert output[selected, 2].cpu().tolist() == [serial, 0, 2, 0, 0, 0, 0, 0, 0]
    valid.fill_(1)
    ids[0, 0] = 99
    graph.replay()
    torch.cuda.synchronize()
    assert output[selected, 2, 3].item() == 1  # Invalid route is observable, never an out-of-bounds write.


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicitly selected idle test GPU")
@pytest.mark.parametrize("capacity,valid_rows", [(32, 31), (192, 173), (8192, 8179)])
def test_compiled_histogram_uses_valid_rows_at_target_decode_and_prefill_sizes(capacity, valid_rows):
    import torch
    from ignition.components.routing_trace import RoutingTrace

    output = torch.zeros((2, 45, 292), dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.tensor([17], dtype=torch.int64, device="cuda")
    valid = torch.tensor([valid_rows], dtype=torch.int32, device="cuda")
    host_ids = torch.arange(capacity * 8, dtype=torch.int32).reshape(capacity, 8) % 288
    host_ids[valid_rows:] = 999
    ids = host_ids.cuda()
    trace = RoutingTrace(output, frame, sequence)
    counter = trace.counter(44, ids, valid)

    def forward():
        trace.begin()
        counter()

    execute = torch.compile(forward, fullgraph=True, options={"triton.cudagraphs": False})
    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.bincount(host_ids[:valid_rows].flatten().long(), minlength=288)
    assert output[0, 44, :4].cpu().tolist() == [17, valid_rows, 8, 0]
    assert torch.equal(output[0, 44, 4:].cpu(), expected)
    assert torch.equal(ids.cpu(), host_ids)
    assert torch.count_nonzero(output[0, :44, 1:]).item() == 0
    assert torch.count_nonzero(output[1]).item() == 0
