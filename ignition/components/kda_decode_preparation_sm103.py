# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Race complete ordinary KDA decode operations before graph capture."""

from dataclasses import dataclass, replace

import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x.sequence import gdn_decode

from .gated_norm import PreparedGatedRMSNorm
from .kda_decode_sm103 import OPERANDS, State as Recurrence, compile_decode, recurrence_plan
from .native_operator import native_contract, native_plan
from .recurrent import prepare_sequence_binding


@dataclass(frozen=True)
class Query:
    recurrence: object
    eps: float


CONTRACT = native_contract(component_id="flare.kda_decode", query_type=Query,
                           tactics=lambda query: ("b12x", 32, 16), semantic_version=2)
INPUTS = ("mixed_qkv", "raw_g", "raw_beta", "A_log", "dt_bias", "z", "norm_weight",
          "query_start_loc", "num_accepted_tokens", "state_indices", "num_seqs", "num_tokens")


@dataclass(frozen=True)
class Binding:
    inputs: tuple
    state: torch.Tensor
    raw: torch.Tensor
    output: torch.Tensor
    scratch: torch.Tensor
    native: object


@dataclass(frozen=True)
class State:
    baseline: object
    norm: object
    recurrence: object
    eps: float

    @property
    def layout(self):
        return self.baseline.layout

    def bind_kda(self, *, raw_output, scratch, **arguments):
        if isinstance(scratch, dict):
            scratch, = scratch.values()
        native = self.baseline.bind_kda(scratch=scratch, **arguments)
        return Binding(tuple(arguments[name] for name in INPUTS), arguments["recurrent_state"],
                       raw_output, arguments["output"], scratch, native)

    def run(self, binding, *, lower_bound=-5.0, eps):
        if lower_bound != -5.0 or eps != self.eps:
            raise ValueError("prepared ordinary KDA decode requires its declared gate and norm epsilon")
        if self.recurrence is None:
            self.baseline.run(binding.native, lower_bound=lower_bound, eps=eps)
        else:
            arguments = dict(zip(INPUTS, binding.inputs), recurrent_state=binding.state, raw_output=binding.raw)
            self.recurrence.run(tuple(arguments[name] for name in OPERANDS))
            self.norm.run((binding.raw, arguments["z"], arguments["norm_weight"],
                           binding.output, arguments["num_tokens"]))


@torch.library.custom_op("flare::kda_decode", mutates_args=("state", "raw", "output"))
def execute(inputs: list[torch.Tensor], state: torch.Tensor, raw: torch.Tensor,
            output: torch.Tensor, scratch: torch.Tensor, handle: int) -> None:
    prepared = require_prepared(plan_from_handle(handle), CONTRACT.component_id, state.device)
    arguments = dict(zip(INPUTS, inputs), recurrent_state=state, output=output)
    binding = prepared.bind_kda(raw_output=raw, scratch=scratch, **arguments)
    prepared.run(binding, eps=prepared.eps)


@execute.register_fake
def _fake(inputs, state, raw, output, scratch, handle):
    pass


class PreparedKDADecode:
    """Select b12x or packed recurrence plus gated norm using the complete operation.

    Persistent state and all activations remain caller-owned. Kernel compilation,
    scratch planning and tactic selection finish before capture. The b12x plan
    retains its own legal recurrent tile race; it also supplies binding validation.
    """

    def __init__(self, baseline, arguments, raw_output, *, session, name, workspace, eps):
        if arguments["state_indices"].shape[1] != 1:
            raise ValueError("ordinary KDA decode requires one state-index column")
        if raw_output.data_ptr() == arguments["output"].data_ptr():
            raise ValueError("ordinary KDA recurrence and normalized output require distinct storage")
        self.binding = None
        base_name, norm_name = f"{name}.b12x", f"{name}.norm"
        prepare_sequence_binding(baseline, arguments, session=session, name=base_name,
            public_bind=gdn_decode.bind_kda, bind_method="bind_kda", mode="decode", workspace=workspace,
            run_options={"lower_bound": -5.0, "eps": eps}, set_binding=lambda binding: None)
        norm = PreparedGatedRMSNorm(raw_output, arguments["z"], arguments["norm_weight"],
            arguments["output"], arguments["num_tokens"], eps=eps, session=session, name=norm_name)
        recurrence = recurrence_plan(dict(arguments, raw_output=raw_output))
        payload = recurrence.contract.encode_query(recurrence.query)
        query = Query(tuple(sorted(payload.items())), float(eps))

        def materialize(tile):
            program = None if tile == "b12x" else compile_decode(payload, tile, raw_output.device.index)
            if program is not None:
                load_programs(program)
            candidate = None if program is None else Recurrence(recurrence.query, tile, program)
            state = State(baseline.prepared.state, norm.plan.prepared.state, candidate, float(eps))
            return state if program is None else attach_programs(state, program)

        plan = native_plan(contract=CONTRACT, query=query, device=raw_output.device,
            scratch_bytes=lambda tile: 0, materialize=materialize,
            compile_jobs=lambda tile, device: () if tile == "b12x" else (CompileJob.create(
                "ignition.components.kda_decode_sm103:compile_decode", payload, tile, device.ordinal),))
        self.plan = replace(plan, dependencies=(base_name, norm_name))

        def bind(plan, **kwargs):
            return require_prepared(plan, CONTRACT.component_id).bind_kda(**kwargs)

        prepare_sequence_binding(self.plan, dict(arguments, raw_output=raw_output), session=session,
            name=name, public_bind=bind, bind_method="bind_kda", mode="decode", workspace=workspace,
            run_options={"lower_bound": -5.0, "eps": eps},
            set_binding=lambda binding: setattr(self, "binding", binding))

    def __call__(self):
        b = self.binding
        execute(list(b.inputs), b.state, b.raw, b.output, b.scratch, self.plan.handle)
