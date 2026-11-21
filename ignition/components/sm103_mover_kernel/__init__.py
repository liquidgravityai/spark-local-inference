# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation-only CUDA extension loader for SM103 GPU-side expert mover kernels."""

from functools import cache
import hashlib
from pathlib import Path


@cache
def prepare(kernel_cache: Path):
    import torch
    from torch.utils.cpp_extension import load

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM103 expert mover kernels must be compiled before capture")
    root = Path(__file__).parent
    source = root / "mover_kernels.cu"
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update((torch.__version__ + str(torch.version.cuda)).encode())
    key = digest.hexdigest()
    build = Path(kernel_cache) / "sm103-expert-mover" / key
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="flare_sm103_mover_" + key[:16],
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-gencode=arch=compute_103a,code=sm_103a",
            "--expt-relaxed-constexpr",
        ],
        build_directory=str(build),
        verbose=False,
    )
