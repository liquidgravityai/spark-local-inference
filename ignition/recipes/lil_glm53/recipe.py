# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Local Inference Lab GLM checkpoint with resident expert-scale defaults."""

from dataclasses import replace
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, SamplingDefaults, SamplingRange
from ignition.recipes.glm53.recipe import RECIPE as GLM_RECIPE


RECIPE = replace(
    GLM_RECIPE,
    name="lil_glm53",
    served_model_name="LIL-GLM-5.3-Flash",
    # The LIL card has no sampling section. Upstream GLM evaluation presets:
    # https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
        ranges={"temperature": SamplingRange(.95, 1.), "top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(
        alias="lil-glm-5.3-flash",
        repository="local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
        revision="dec48abd33efa73c3bb7c95b74eee10cad34f9be",
        architecture="Glm5NextForConditionalGeneration",
        stored_bytes=177_680_678_236,
        # Native expansion replaces 9,825,076,532 stored CSF bytes with
        # 19,025,362,944 scale bytes; shared expansion scratch is in graph costs.
        device_bytes=186_880_964_648,
        scale_compressed_device_bytes={"csf": 177_680_678_236},
        draft_bytes=4_301_658_112),),
    expert_scale_compression=None,
    expert_scale_compression_by_rank=MappingProxyType({}),
)
