/**
 * @file bindings.cpp
 * @brief Python Bindings for the High-Performance C++ Kernel (BIO Engine v4.6)
 * @details 使用 pybind11 将支持全序一致性约束的采样引擎暴露给 Python。
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "mcmc_sampler.hpp"
#include "types.hpp"
#include "diagnostics.hpp"

namespace py = pybind11;
using namespace mcm::engine;
using namespace mcm::types;

// 确保版本号对齐重构版本
#ifndef VERSION_INFO
    #define VERSION_INFO "4.6.0-FullConsistency"
#endif

PYBIND11_MODULE(mcm_core_lib, m) {
    // 1. 模块元数据
    m.doc() = "MCM 2026 Problem C - High Performance Bayesian Inverse Optimization Engine (Full-Rank Consistent)";
    m.attr("__version__") = VERSION_INFO;

    // 2. 导出赛制枚举 (MechanismType)
    py::enum_<MechanismType>(m, "MechanismType", "DWTS 淘汰机制枚举")
        .value("RANK_BASED", MechanismType::RANK_BASED, "排名制: 使用序数信号聚合")
        .value("PERCENT_BASED", MechanismType::PERCENT_BASED, "百分比制: 使用基数信号叠加")
        .export_values();

    // 3. 导出采样器超参数配置 (SamplerConfig)
    // 物理意义：将 Python 侧的策略参数映射为 C++ 运行时的硬约束
    py::class_<SamplerConfig>(m, "SamplerConfig", "MCMC 采样器与似然函数配置")
        .def(py::init<>())
        // --- 采样链控制 ---
        .def_readwrite("n_chains", &SamplerConfig::n_chains)
        .def_readwrite("n_samples", &SamplerConfig::n_samples)
        .def_readwrite("burn_in_ratio", &SamplerConfig::burn_in_ratio)
        .def_readwrite("thinning", &SamplerConfig::thinning)
        .def_readwrite("init_step_size", &SamplerConfig::init_step_size)
        .def_readwrite("adaptive", &SamplerConfig::adaptive)
        .def_readwrite("seed", &SamplerConfig::seed)
        .def_readwrite("return_traces", &SamplerConfig::return_traces)
        // --- 似然函数刚度与贝叶斯先验控制 ---
        .def_readwrite("rank_tau", &SamplerConfig::rank_tau)
        .def_readwrite("elim_penalty", &SamplerConfig::elim_penalty)
        .def_readwrite("jeopardy_penalty", &SamplerConfig::jeopardy_penalty)
        .def_readwrite("prior_strength", &SamplerConfig::prior_strength)
        .def_readwrite("enable_judge_save", &SamplerConfig::enable_judge_save)
        .def("__repr__", [](const SamplerConfig &c) {
            return "<SamplerConfig: chains=" + std::to_string(c.n_chains) +
                   ", samples=" + std::to_string(c.n_samples) + ">";
        });

    // 4. 导出推断结果报告 (InferenceResult)
    py::class_<InferenceResult>(m, "InferenceResult", "贝叶斯推断后验统计报告")
        .def_readonly("posterior_mean", &InferenceResult::posterior_mean, "后验均值 (估计票数占比)")
        .def_readonly("posterior_std", &InferenceResult::posterior_std, "后验标准差 (估计不确定性)")
        .def_readonly("r_hat", &InferenceResult::r_hat, "Split-R-hat 收敛审计指标")
        .def_readonly("ess", &InferenceResult::ess, "有效样本当量 (ESS)")
        .def_readonly("acceptance_rate", &InferenceResult::acceptance_rate, "MCMC 链平均接受率")
        .def_readonly("fidelity_score", &InferenceResult::fidelity_score, "业务逻辑保真度 (淘汰一致性)")
        .def_readonly("converged", &InferenceResult::converged, "收敛审计状态")
        .def_readonly("traces", &InferenceResult::traces, "全量采样轨迹");

    // 5. 导出核心采样器 (MCMCSampler)
    py::class_<MCMCSampler>(m, "MCMCSampler", "高性能并行 MCMC 推理引擎")
        .def(py::init<const SamplerConfig&>(), py::arg("config"))
        // [核心导出]: run_parallel_inference
        // 注意：py::arg 的顺序和数量必须与 mcmc_sampler.hpp 严格一致
        .def("run_parallel_inference", &MCMCSampler::run_parallel_inference,
             py::call_guard<py::gil_scoped_release>(), // 释放 GIL，允许真正的多核并行
             py::arg("judge_scores"),
             py::arg("elim_idx"),
             py::arg("jeopardy_mask"),
             py::arg("prior_mean"),
             py::arg("mech_type"),
             py::arg("winner_idx") = -1, // <--- [New] 冠军索引，默认 -1 表示普通比赛周
             "执行基于 OpenMP 并行的贝叶斯隐变量反演"
        );
}