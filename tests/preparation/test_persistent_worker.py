# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ignition PersistentWorker declaration and PreparedExecution validation."""

import pytest
import torch

from ignition.components.completion import VECTOR_TOKENS_FIELD
from ignition.preparation.execution import GraphSignature, PersistentWorker, PreparedExecution


class _FakeGraph:
    def __init__(self, handle):
        self._handle = handle

    def raw_cuda_graph_exec(self):
        return self._handle


def _base_buffers(max_active=4, token_capacity=8):
    return {
        "frame_index": torch.zeros((1,), dtype=torch.int32),
        "submission_sequence": torch.zeros((1,), dtype=torch.int64),
        "completion_records": torch.zeros((2, max_active, VECTOR_TOKENS_FIELD + token_capacity), dtype=torch.int64),
        "worker_ctrl": torch.zeros((128,), dtype=torch.uint8),
    }


def _patch_bundle(monkeypatch, execution):
    def fake_validate(*, export_graphs=False):
        handles = [100 + i for i in range(len(execution.graphs))]
        spans = [(0x800_0000_0000 + i * 0x10000, t.numel() * t.element_size())
                 for i, t in enumerate(execution.buffers.values())]
        return (0x1234, handles, spans)

    monkeypatch.setattr(execution, "_validate_resources", fake_validate)


def test_persistent_worker_dataclass_validation():
    g = _FakeGraph(1)
    pw = PersistentWorker(
        name="mover",
        graph=g,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=0,
        metrics_buffer="worker_ctrl",
        metrics_offset=8,
        metrics_count=5,
    )
    assert pw.worker_sm_count == 8

    with pytest.raises(ValueError, match="nonempty graph name"):
        PersistentWorker(name=" ", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl")
    with pytest.raises(ValueError, match="positive uint32"):
        PersistentWorker(name="mover", graph=g, worker_sm_count=0, stop_flag_buffer="worker_ctrl")
    with pytest.raises(ValueError, match="4-byte-aligned"):
        PersistentWorker(name="mover", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl", stop_flag_offset=2)
    with pytest.raises(ValueError, match="8-byte aligned"):
        PersistentWorker(
            name="mover", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl",
            metrics_buffer="worker_ctrl", metrics_offset=4, metrics_count=2,
        )
    with pytest.raises(ValueError, match="empty/zero"):
        PersistentWorker(
            name="mover", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl",
            metrics_buffer="worker_ctrl", metrics_offset=0, metrics_count=0,
        )


def test_prepared_execution_persistent_worker_manifest_and_rejection(monkeypatch):
    decode_graph = _FakeGraph(10)
    worker_graph = _FakeGraph(20)
    sig = GraphSignature(kind="decode", batch_capacity=4, token_capacity=1)
    pw = PersistentWorker(
        name="mover",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=0,
        metrics_buffer="worker_ctrl",
        metrics_offset=8,
        metrics_count=5,
    )
    exe = PreparedExecution(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
        persistent_worker=pw,
    )
    _patch_bundle(monkeypatch, exe)
    assert "mover" in exe.graphs
    assert "mover" not in exe.serving_graphs

    live = exe.manifest()
    assert live["persistent_worker"]["graph"] == "mover"
    assert live["persistent_worker"]["worker_sm_count"] == 8
    assert live["persistent_worker"]["handle"] == 101

    exported = exe.export_manifest()
    assert "handle" not in exported["persistent_worker"]
    assert exported["persistent_worker"]["stop_flag_buffer"] == "worker_ctrl"
    assert exported["persistent_worker"]["metrics_offset"] == 8
    assert exported["persistent_worker"]["metrics_count"] == 5

    # Name collision with serving graph rejected
    with pytest.raises(ValueError, match="distinct from serving and tuning"):
        PreparedExecution(
            device=0,
            graphs={"mover": decode_graph},
            signatures={"mover": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_worker=pw,
        )

    # Overlapping stop_flag and metrics rejected
    bad_overlap = PersistentWorker(
        name="mover", graph=worker_graph, worker_sm_count=8,
        stop_flag_buffer="worker_ctrl", stop_flag_offset=8,
        metrics_buffer="worker_ctrl", metrics_offset=8, metrics_count=2,
    )
    with pytest.raises(ValueError, match="overlaps stop flag"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_worker=bad_overlap,
        )

    # Host-upload buffer rejected for stop flag
    bad_upload = PersistentWorker(
        name="mover", graph=worker_graph, worker_sm_count=8,
        stop_flag_buffer="frame_index", stop_flag_offset=0,
    )
    with pytest.raises(ValueError, match="cannot receive host uploads"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_worker=bad_upload,
        )
