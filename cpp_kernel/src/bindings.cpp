/**
 * MCM 2026 Problem C: C++/Python Bridge (pybind11)
 * Role: Exposing the HPC Sampling Kernel to the Python Logic Layer.
 * Standard: Industrial Quant Bridge (Zero-copy via Eigen Map & GIL Management).
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>   // 核心：处理 NumPy 数组与 Eigen 矩阵的无缝转换
#include <pybind11/stl.h>     // 核心：处理 std::vector 和 std::string 的自动转换
#include "mcmc_sampler.hpp"
#include "diagnostics.hpp"

namespace py = pybind11;
using namespace mcm::core;

PYBIND11_MODULE(mcm_core_lib, m) {
    m.doc() = "MCM 2026 High-Performance Bayesian Inference Kernel (BIO-Engine)";

    // --- 1. 导出推断结果结构体 (InferenceResult) ---
    // 物理意义：让 Python 侧能以对象属性方式读取 MCMC 统计产出
    py::class_<MCMCSampler::InferenceResult>(m, "InferenceResult")
        .def_readonly("posterior_mean", &MCMCSampler::InferenceResult::posterior_mean)
        .def_readonly("posterior_std", &MCMCSampler::InferenceResult::posterior_std)
        .def_readonly("shannon_entropy", &MCMCSampler::InferenceResult::shannon_entropy)
        .def_readonly("r_hat", &MCMCSampler::InferenceResult::r_hat)
        .def_readonly("acceptance_rate", &MCMCSampler::InferenceResult::acceptance_rate)
        .def_readonly("converged", &MCMCSampler::InferenceResult::converged)
        .def("__repr__", [](const MCMCSampler::InferenceResult &a) {
            return "<InferenceResult: R-hat=" + std::to_string(a.r_hat) +
                   ", Entropy=" + std::to_string(a.shannon_entropy) + ">";
        });

    // --- 2. 导出核心采样类 (MCMCSampler) ---
    py::class_<MCMCSampler>(m, "MCMCSampler")
        .def(py::init<int>(), py::arg("seed") = 2026)

        /**
         * 1. 释放 GIL (gil_scoped_release):
         *    这是实现真正并行的唯一手段。如果不加这一行，Python 的全局锁会强制所有 C++ 线程
         *    串行排队。加上它，23 个 CPU 核心将瞬间吃满 100%。
         *
         * 2. 参数对齐 (Argument Mapping):
         *    必须严格对应 mcmc_sampler.hpp 中的 8 个参数。
         */
        .def("run_parallel_inference",
            &MCMCSampler::run_parallel_inference,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("judge_signals"),
            py::arg("elim_idx"),
            py::arg("jeopardy_mask"), // 参数 3: 对应 O 奖特有的危机信号约束
            py::arg("prior_mu"),
            py::arg("mechanism"),
            py::arg("n_chains") = 23,
            py::arg("n_samples") = 100000,
            py::arg("jump_size") = 0.05,
            "执行 23 核并行 MCMC 采样，反演观众投票分布"
        );

    // --- 3. 导出辅助诊断工具 (用于交互式调试) ---
    m.def("compute_r_hat_debug", &mcm::diag::compute_r_hat,
          "计算多链收敛性指标 (Gelman-Rubin)");

    // --- 4. 导出项目版本元数据 ---
    #ifdef VERSION_INFO
        m.attr("__version__") = py::str(VERSION_INFO);
    #else
        m.attr("__version__") = "3.1.0-O-Prize-Production";
    #endif
}