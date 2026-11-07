# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from ignition.components.native_operator import NativeConfig, native_contract


def test_native_contract_preserves_order_and_validates_each_query():
    @dataclass(frozen=True)
    class Query:
        extent: int

    calls = []

    def tactics(query):
        calls.append(query)
        return ([(query.extent, 2), [32, True]], [(query.extent, 1), [16, False]])

    contract = native_contract(component_id="flare.test_native", query_type=Query, tactics=tactics)
    first, second = Query(4), Query(8)
    config = contract.configure(first, device=None)
    assert config.default.tactic == ((4, 2), (32, True))
    assert contract.parameters(first, None)["tactic"] == (((4, 2), (32, True)), ((4, 1), (16, False)))
    contract.validate_config(first, NativeConfig(((4, 1), (16, False))), None)
    with pytest.raises(ValueError):
        contract.validate_config(first, NativeConfig(((8, 1), (16, False))), None)
    assert contract.configure(second, device=None).default.tactic == ((8, 2), (32, True))
    assert calls == [first, second]
