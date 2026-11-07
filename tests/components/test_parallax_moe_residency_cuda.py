"""Compare captured SM103 MoE outputs while native storage changes weight tiers.

This tests Parallax's noninterference with prepared kernels, not model accuracy.
The checkpoint's expanded scales and calibration vectors remain in HBM.
"""
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned idle GB300 and FLARE_RUN_CUDA_TESTS=1",
)


@pytest.fixture(scope="module")
def experts():
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.moe_sm103 import TrtllmNvfp4Experts
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.glm53.weights import GlmWeightLoader

    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
    assert torch.cuda.get_device_capability(0) == (10, 3)
    snapshot = os.environ.get("FLARE_GLM53_SNAPSHOT")
    assert snapshot, "select the pinned checkpoint explicitly"
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    prefix = "model.language_model.layers.3.mlp.experts."
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                         prefixes=(prefix,), expert_activation="a4") as loader:
        value = loader.load_experts(3)
    assert isinstance(value, TrtllmNvfp4Experts)
    assert value.num_experts == 288 and not value.has_csf_blocks
    assert value.input_global_scale.numel() == 288
    assert value.input_global_scale.unique().numel() > 1
    return value


def _mapped_weights(experts):
    from flare_heliopause import ExpertResidencyTestStorage

    fields = ("w13", "w2")
    regions, sizes = [], []
    offset = 0
    for part, field in enumerate(fields):
        tensor = getattr(experts, field)
        assert tensor.is_contiguous() and tensor.dtype == torch.uint8
        size = tensor[0].numel() * tensor.element_size()
        sizes.append(size)
        for expert in range(experts.num_experts):
            regions.append((3, expert, part, offset + expert * size,
                            tensor.data_ptr() + expert * size, size))
        offset += tensor.numel() * tensor.element_size()
    slots = (experts.num_experts + 1) * len(fields)
    storage = ExpertResidencyTestStorage(experts, regions, slots, slots * max(sizes), offset)
    views = {}
    for part, field in enumerate(fields):
        original = getattr(experts, field)
        address = storage.address(3, 0, part)
        for expert in range(experts.num_experts):
            assert storage.address(3, expert, part) == address + expert * sizes[part]
        raw = torch._C._construct_storage_from_data_pointer(address, original.device, original.numel())
        view = torch.empty(0, dtype=original.dtype, device=original.device).set_(
            raw, 0, original.shape, original.stride())
        view._parallax_owner = storage
        views[field] = view
    return storage, replace(experts, **views)


@pytest.mark.parametrize("tokens", [1, 32, 257, 8192])
def test_prepared_moe_replays_exactly_across_residency(experts, tokens):
    from b12x.preparation import PreparationSession
    from ignition.components.moe import PreparedRoutedExperts, RouterScratch
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    storage, mapped = _mapped_weights(experts)
    count = experts.num_experts
    all_ids = [(3, i) for i in range(count)]
    storage.exchange(all_ids, [])
    for field in ("w13", "w2"):
        assert torch.equal(getattr(experts, field), getattr(mapped, field))
    source = torch.zeros((tokens, 4096), dtype=torch.bfloat16, device="cuda")
    routes = RouterScratch.allocate(tokens, source.device, experts=count, top_k=8)
    ids = ((torch.arange(tokens)[:, None] * 7 + torch.arange(8)[None, :]) % count).to(torch.int32)
    routes.ids.copy_(ids)
    routes.weights.fill_(2.5 / 8)
    output = torch.empty_like(source)
    workspace = ScratchArena("cuda:0")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        prepared = PreparedRoutedExperts(source, mapped, routes.ids, routes.weights, output,
            session=session, name=f"parallax_storage_moe:{tokens}", workspace=workspace)
        session.freeze()
        workspace.finalize()
        workspace.freeze()
        run = compile_forward(lambda: prepared())
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        source.copy_(torch.randn((tokens, 4096), generator=torch.Generator().manual_seed(941 + tokens)))

        def replay():
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            result = output.cpu()
            assert torch.isfinite(result).all()
            return result

        baseline = replay()
        assert baseline.abs().max() > 0, "the captured expert computation must produce nonzero output"
        assert torch.equal(replay(), baseline), "all-HBM repeatability must hold before tier comparisons"
        phases = []
        phases.append(("all_grace", storage.exchange([], all_ids)))
        assert torch.equal(replay(), baseline), "Grace execution changed the captured output"
        hot = all_ids[:8]
        phases.append(("mixed", storage.exchange(hot, [])))
        assert torch.equal(replay(), baseline), "mixed residency changed the captured output"
        for i in range(3):
            incoming = all_ids[8 * (i + 1):8 * (i + 2)]
            phases.append((f"exchange_{i}", storage.exchange(incoming, hot)))
            hot = incoming
            assert torch.equal(replay(), baseline), "expert exchange changed the captured output"
        phases.append(("all_hbm", storage.exchange([e for e in all_ids if e not in hot], [])))
        assert torch.equal(replay(), baseline), "restoring HBM changed the captured output"
        print(json.dumps({"tokens": tokens, "experts": count, "top_k": 8,
            "implementation": prepared.binding.implementation,
            "exact_output_parity": True, "transaction_wall_ns": dict(phases)}))
        graph.reset()
        storage.close()
