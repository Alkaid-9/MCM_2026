/**
 * @file bindings.cpp
 * @brief Python Bindings for the High-Performance C++ Kernel (BIO Engine v4.1)
 * @details 使用 pybind11 将 C++ 采样引擎暴露给 Python，实现零拷贝数据传输与多核并发。
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [核心职责]:
 * 1. Data Marshalling: 利用 Eigen::Ref 实现 Numpy 内存视图的零拷贝映射。
 * 2. Parameter Sync: 严格对齐 SamplerConfig 结构体，支持从 Python 注入物理约束。
 * 3. Concurrency: 在执行 run_parallel_inference 时释放 GIL，确保 23 核满载。
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

#ifndef VERSION_INFO
    #define VERSION_INFO "4.1.0-Platinum"
#endif

PYBIND11_MODULE(mcm_core_lib, m) {
    // 1. 模块元数据
    m.doc() = "MCM 2026 Problem C - High Performance Bayesian Inverse Optimization Engine";
    m.attr("__version__") = VERSION_INFO;

    // 2. 导出赛制枚举 (MechanismType)
    py::enum_<MechanismType>(m, "MechanismType", "DWTS 淘汰机制枚举")
        .value("RANK_BASED", MechanismType::RANK_BASED, "排名制: 使用 Soft-Rank 聚合信号")
        .value("PERCENT_BASED", MechanismType::PERCENT_BASED, "百分比制: 线性叠加信号")
        .export_values();

    // 3. 导出采样器超参数配置 (SamplerConfig)
    // [关键修正]: 补全了 elim_penalty, rank_tau 等缺失字段，修复 AttributeError
    py::class_<SamplerConfig>(m, "SamplerConfig", "MCMC 采样器与似然函数配置")
        .def(py::init<>())
        // --- 采样链控制 ---
        .def_readwrite("n_chains", &SamplerConfig::n_chains, "并行链数量")
        .def_readwrite("n_samples", &SamplerConfig::n_samples, "总迭代步数")
        .def_readwrite("burn_in", &SamplerConfig::burn_in, "预热期样本数")
        .def_readwrite("thinning", &SamplerConfig::thinning, "稀疏化步长")
        .def_readwrite("init_step_size", &SamplerConfig::init_step_size, "初始建议分布步长")
        .def_readwrite("adaptive", &SamplerConfig::adaptive, "是否开启 Dual-Averaging 自适应步长")
        .def_readwrite("seed", &SamplerConfig::seed, "随机数种子")
        .def_readwrite("return_traces", &SamplerConfig::return_traces, "是否回传全量采样轨迹")
        // --- 似然函数刚度 (Stiffness) 与物理约束 ---
        .def_readwrite("rank_tau", &SamplerConfig::rank_tau, "Soft-Rank 温度系数")
        .def_readwrite("elim_penalty", &SamplerConfig::elim_penalty, "淘汰事实违规惩罚强度 (Hard Constraint)")
        .def_readwrite("jeopardy_penalty", &SamplerConfig::jeopardy_penalty, "危险区违规惩罚强度 (Soft Constraint)")
        .def_readwrite("entropy_weight", &SamplerConfig::entropy_weight, "最大熵正则化权重")
        .def("__repr__", [](const SamplerConfig &c) {
            return "<SamplerConfig: chains=" + std::to_string(c.n_chains) +
                   ", samples=" + std::to_string(c.n_samples) + ">";
        });

    // 4. 导出推断结果报告 (InferenceResult)
    py::class_<InferenceResult>(m, "InferenceResult", "贝叶斯推断后验统计报告")
        .def_readonly("posterior_mean", &InferenceResult::posterior_mean, "后验均值 (估计票数占比)")
        .def_readonly("posterior_std", &InferenceResult::posterior_std, "后验标准差 (不确定性度量)")
        .def_readonly("r_hat", &InferenceResult::r_hat, "Gelman-Rubin 收敛指标")
        .def_readonly("ess", &InferenceResult::ess, "有效样本量估算")
        .def_readonly("acceptance_rate", &InferenceResult::acceptance_rate, "平均接受率")
        .def_readonly("fidelity_score", &InferenceResult::fidelity_score, "业务逻辑还原度")
        .def_readonly("converged", &InferenceResult::converged, "收敛审计状态")
        .def_readonly("traces", &InferenceResult::traces, "全量采样轨迹 (仅在 return_traces 为 true 时有效)");

    // 5. 导出核心采样器 (MCMCSampler)
    py::class_<MCMCSampler>(m, "MCMCSampler", "高性能并行 MCMC 推理引擎")
        .def(py::init<const SamplerConfig&>(), py::arg("config"))
        .def("run_parallel_inference", &MCMCSampler::run_parallel_inference,
             py::call_guard<py::gil_scoped_release>(), // [核心亮点] 释放 GIL，允许 OpenMP 23 核全速运行
             py::arg("judge_scores"),
             py::arg("elim_idx"),
             py::arg("jeopardy_mask"),
             py::arg("prior_mean"),
             py::arg("mech_type")
        );
}