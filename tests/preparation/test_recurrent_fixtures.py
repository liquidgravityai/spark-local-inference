# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recurrent tuning fixtures preserve live sequence lengths within padded verification metadata."""

import pytest
import torch

from ignition.preparation.fixtures.recurrent import BenchmarkStateArena, RecurrentBenchmarkInputs


@pytest.mark.parametrize("sequences,tokens,columns", [(32, 192, 8), (32, 32, 8), (4, 17, 8)])
@pytest.mark.parametrize("recording", [False, True])
def test_decode_fixture_allocates_only_live_token_checkpoints(sequences, tokens, columns, recording):
    state = torch.zeros(tokens + 14, 1, 2, 2)
    arguments = {
        "mixed_qkv": torch.zeros(tokens, 6, dtype=torch.bfloat16),
        "recurrent_state": state, "output": torch.zeros(tokens, 1, 2, dtype=torch.bfloat16),
        "query_start_loc": torch.zeros(sequences + 1, dtype=torch.int32),
        "num_tokens": torch.zeros(1, dtype=torch.int32), "num_seqs": torch.zeros(1, dtype=torch.int32),
        "num_accepted_tokens": torch.ones(sequences, dtype=torch.int32),
        "state_indices": torch.zeros(sequences, columns, dtype=torch.int64),
    }
    if recording:
        arguments["correction_cache"] = torch.zeros(tokens, 1, 2)
    arena = BenchmarkStateArena(state)
    fixture = RecurrentBenchmarkInputs(arguments, mode="decode", arena=arena)
    rows = sequences + 1 if recording else tokens + 1
    assert fixture.state_rows == rows
    assert fixture.args["num_seqs"].item() == sequences
    assert fixture.args["num_tokens"].item() == tokens
    boundaries = fixture.args["query_start_loc"]
    indices = fixture.args["state_indices"]
    live = []
    for row in range(sequences):
        length = int(boundaries[row + 1] - boundaries[row])
        if recording:
            length = 1
        assert not indices[row, length:].count_nonzero()
        live.extend(indices[row, :length].tolist())
    assert live == list(range(1, rows))
    arena.storage.fill_(127)
    fixture.reset()
    torch.testing.assert_close(fixture.state[:rows], fixture.initial_state)
    assert not fixture.state[0].count_nonzero()
    assert not arguments["state_indices"].count_nonzero()


def test_decode_fixture_preserves_transient_output_and_packed_beta_stride():
    sequences = 8
    state = torch.zeros(sequences + 2, 1, 2, 2)
    projection = torch.zeros(sequences, 12, dtype=torch.bfloat16)
    raw = torch.full((sequences, 1, 2), 37, dtype=torch.bfloat16)
    arguments = {
        "mixed_qkv": torch.zeros(sequences, 6, dtype=torch.bfloat16),
        "raw_beta": projection[:, 6:7], "raw_output": raw,
        "recurrent_state": state, "output": torch.zeros_like(raw),
        "query_start_loc": torch.zeros(sequences + 1, dtype=torch.int32),
        "num_tokens": torch.zeros(1, dtype=torch.int32), "num_seqs": torch.zeros(1, dtype=torch.int32),
        "num_accepted_tokens": torch.ones(sequences, dtype=torch.int32),
        "state_indices": torch.zeros(sequences, 1, dtype=torch.int64),
    }
    fixture = RecurrentBenchmarkInputs(arguments, mode="decode", arena=BenchmarkStateArena(state))
    assert fixture.args["raw_output"] is raw
    assert "raw_output" not in fixture.sources
    assert fixture.args["raw_beta"].stride() == arguments["raw_beta"].stride()
    fixture.reset()
    fixture.produce(False)
    torch.testing.assert_close(raw, torch.full_like(raw, 37), rtol=0, atol=0)
    torch.testing.assert_close(fixture.args["raw_beta"], fixture.sources["raw_beta"], rtol=0, atol=0)
