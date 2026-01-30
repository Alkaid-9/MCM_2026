/**
 * MCM 2026 Problem C: C++/Python Bridge (pybind11)
 * Role: Exposing the HPC Sampling Kernel to the Python Logic Layer.
 * Standard: Industrial Quant Bridge (Zero-copy via Eigen Map & GIL Management).
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>  // 极其重要：自动处理 NumPy 数组到 Eigen 矩阵的映射
#include <pybind11/stl.h>    // 处理 std::vector, std::string
#include "mcmc_sampler.hpp"
#include "diagnostics.hpp"

namespace py = pybind11;
using namespace mcm::core;

PYBIND11_MODULE(mcm_core_lib, m) {
    m.doc() = "MCM 2026 High-Performance Bayesian Inference Kernel (BIO-Engine)";

    // 1. 导出推断结果结构体 (InferenceResult)
    // 物理意义：让 Python 能够像访问对象属性一样读取 MCMC 的统计输出
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

    // 2. 导出核心采样类 (MCMCSampler)
    py::class_<MCMCSampler>(m, "MCMCSampler")
        .def(py::init<int>(), py::arg("seed") = 2026)

        /**
         * 【核心工程补丁】run_parallel_inference
         * 1. 使用 py::call_guard<py::gil_scoped_release>() 释放 GIL 锁
         *    这是让 23 核 CPU 并行的唯一方式。
         * 2. 利用 pybind11/eigen.h 实现 NumPy -> Eigen::VectorXd 的无缝转换。
         */
        .def("run_parallel_inference",
             &MCMCSampler::run_parallel_inference,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("judge_signals"),
             py::arg("elim_idx"),
             py::arg("prior_mu"),
             py::arg("mechanism"),
             py::arg("n_chains") = 23,
             py::arg("n_samples") = 100000,
             py::arg("jump_size") = 0.05,
             "执行 23 核并行 MCMC 采样，反演潜变量分布"
        );

    // 3. 辅助诊断工具 (用于 Jupyter Notebook 调试)
    m.def("compute_r_hat_debug", &mcm::diag::compute_r_hat, "计算 Gelman-Rubin 统计量");

    // 4. 版本元数据
    #ifdef VERSION_INFO
        m.attr("__version__") = py::str(VERSION_INFO);
    #else
        m.attr("__version__") = "3.0.0-dev";
    #endif
}