// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
// Keep the preparation binding's runtime separate from the installed library.
#define deep_gemm flare_megamoe_host
#include "csrc/utils/exception.hpp"
#include <deep_gemm/scheduler/mega_moe.cuh>
#include "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#undef deep_gemm

void initialize(const std::string& root, const std::string& cache_dir) {
    flare_megamoe_host::init_jit(root, cache_dir);
}
void set_num_sms(int num_sms) {
    flare_megamoe_host::runtime->set_num_sms(num_sms);
}
int get_num_sms() {
    return flare_megamoe_host::runtime->get_num_sms();
}
void launch(torch::Tensor y, std::vector<std::optional<torch::Tensor>> acts,
            std::vector<torch::Tensor> weights, std::vector<torch::Tensor> shared,
            torch::Tensor control, torch::Tensor buffer, int capacity,
            int experts, int host_rows, int topk, int hidden, int intermediate, float clamp,
            bool shared_first, int hbm_slots) {
    TORCH_CHECK(acts.size() == 8 && weights.size() == 4);
    TORCH_CHECK(shared.size() == 0 || shared.size() == 4);
    TORCH_CHECK(control.scalar_type() == torch::kInt && control.is_contiguous());
    TORCH_CHECK(control.numel() == 16 + experts);
    const int expected_hbm_rows = hbm_slots > 0 ? hbm_slots : (experts - host_rows);
    if (hbm_slots > 0) {
        TORCH_CHECK(host_rows == experts && expected_hbm_rows > 0 && expected_hbm_rows <= experts);
    } else {
        TORCH_CHECK(host_rows > 0 && host_rows < experts);
    }
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kBFloat16 && y.is_contiguous());
    TORCH_CHECK(y.dim() == 2 && y.size(1) == hidden && y.size(0) > 0 && y.size(0) <= capacity);
    TORCH_CHECK(hidden > 0 && intermediate > 0 && hidden % 128 == 0 && intermediate % 128 == 0);
    TORCH_CHECK(topk > 0 && topk <= experts && std::isfinite(clamp) && clamp > 0);
    for (size_t i = 0; i < 4; ++i) {
        const int rows = i < 2 ? 2 * intermediate : hidden;
        const int columns = i < 2 ? hidden : intermediate;
        const bool scale = i % 2 != 0;
        const auto& t = weights[i];
        TORCH_CHECK(t.device() == y.device() && t.dim() == 3);
        TORCH_CHECK(t.size(0) == expected_hbm_rows && t.size(1) == rows);
        TORCH_CHECK(t.size(2) == columns / (scale ? 128 : 2));
        TORCH_CHECK(t.scalar_type() == (scale ? torch::kInt : torch::kInt8));
        TORCH_CHECK(scale ? (t.stride(0) == rows * (columns / 128) && t.stride(1) == 1 && t.stride(2) == rows)
                         : t.is_contiguous());
    }
    TORCH_CHECK(control.device() == y.device() && buffer.device() == y.device());
    for (const auto& t : acts) if (t.has_value()) TORCH_CHECK(t->device() == y.device());
    for (const auto& t : shared) TORCH_CHECK(t.device() == y.device());
    const c10::cuda::CUDAGuard device_guard(y.device());
    torch::Tensor absent;
    const auto act = [&](size_t i) { return acts[i].value_or(absent); };
    flare_megamoe_host::sm100_fp8_fp4_mega_moe(y,
        act(4), act(5), act(6), act(7), act(0), act(1), act(2), act(3),
        weights[0], weights[2], weights[1], weights[3],
        shared.empty() ? absent : shared[0], shared.empty() ? absent : shared[2],
        shared.empty() ? absent : shared[1], shared.empty() ? absent : shared[3],
        control, {reinterpret_cast<int64_t>(buffer.data_ptr())}, 0, capacity, experts,
        shared.empty() ? 0 : 1, y.size(0), topk, hidden, intermediate, clamp, 1.f, 0.f, true,
        shared_first);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialize", &initialize);
    m.def("set_num_sms", &set_num_sms);
    m.def("get_num_sms", &get_num_sms);
    m.def("launch", &launch,
          py::arg("y"), py::arg("acts"), py::arg("weights"), py::arg("shared"),
          py::arg("control"), py::arg("buffer"), py::arg("capacity"),
          py::arg("experts"), py::arg("host_rows"), py::arg("topk"),
          py::arg("hidden"), py::arg("intermediate"), py::arg("clamp"),
          py::arg("shared_first"), py::arg("hbm_slots") = -1);
}
