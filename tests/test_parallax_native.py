"""Exercise routing readback through the native executor with synthetic state."""

import json
import os
import time
from unittest import mock

import pytest


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                   reason="requires an explicitly assigned CUDA device")
@pytest.mark.parametrize("width", [1, 4, 6])
def test_native_routing_readback_preserves_two_frame_results_and_state(width):
    import torch
    import flare_heliopause as native
    from helios.tests.test_execution import Resources

    def run(enabled):
        owner = Resources(native, verification_width=width, routing_trace=enabled)
        ids = owner.route_ids.clone() if enabled else None
        executor = native.GraphExecutor(owner, owner.manifest)
        results, samples = [], []
        allocated = torch.cuda.memory_allocated()
        try:
            with mock.patch.object(torch.cuda.CUDAGraph, "replay",
                                   side_effect=AssertionError("Python replay callback")):
                for cycle in range(4):
                    if enabled and cycle == 3:
                        # Corrupt telemetry input only after both native frames have retired.
                        owner.route_ids[0, 0] = 999
                        torch.cuda.synchronize()
                    first = executor.submit(0, [])
                    second = executor.submit(1, [])
                    assert first == cycle * 2 + 1 and second == first + 1
                    for sequence, tokens in ((first, 2), (second, 4)):
                        deadline = time.monotonic() + 10
                        event = None
                        while event is None and time.monotonic() < deadline:
                            event = executor.poll()
                            if event is None:
                                time.sleep(.001)
                        assert event is not None and event.fence == sequence
                        results.append(event.records)
                        if not enabled:
                            assert event.routing_json is None
                            continue
                        sample = json.loads(event.routing_json)
                        samples.append(sample)
                        assert sample["valid"] == (cycle != 3)
                        if cycle == 3:
                            continue
                        assert sample["submission"] == sequence
                        assert sample["available_ns"] >= sample["demand_lower_bound_ns"]
                        assert [layer["layer"] for layer in sample["layers"]] == list(range(3, 45))
                        for layer in sample["layers"]:
                            assert layer["tokens"] == tokens * width and layer["top_k"] == 8
                            expected = torch.bincount(ids[:tokens * width].cpu().flatten().long(), minlength=288)
                            assert layer["counts"] == expected.tolist()
            assert torch.cuda.memory_allocated() == allocated
        finally:
            executor.close()
        assert executor.drained
        state = owner.consumed.cpu().tolist()
        assert state == [value * width for value in (8, 8, 4, 4)]
        if enabled:
            ids[0, 0] = 999
            assert torch.equal(owner.route_ids, ids)
            assert samples[0]["submission"] == 1
            assert samples[0]["layers"][0]["counts"] == [1] * (16 * width) + [0] * (288 - 16 * width)
        return results, state

    plain = run(False)
    traced = run(True)
    assert traced == plain
