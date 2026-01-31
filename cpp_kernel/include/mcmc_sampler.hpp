/**
 * @file mcmc_sampler.hpp
 * @brief High-Performance Parallel MCMC Engine (Advanced Architecture)
 * @details Implements Adaptive MH on the Simplex with Cache-Line Alignment.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 */

#ifndef MCMC_SAMPLER_HPP
#define MCMC_SAMPLER_HPP

#include <vector>
#include <random>
#include <Eigen/Dense>
#include "likelihood.hpp"
#include "diagnostics.hpp"
#include "types.hpp"

namespace mcm {
namespace engine {

    using namespace mcm::types;

    // =========================================================================
    // 1. 配置与结果结构体 (DTO - Data Transfer Objects)
    // =========================================================================

    /**
     * @struct SamplerConfig
     * @brief 采样器全量超参数配置
     * [重构点]: 整合了似然函数刚度参数，支持从 Python 侧进行实验调优。
     */
    struct SamplerConfig {
        // --- 采样控制 ---
        int n_chains = 23;            // 并行链数量 (建议对齐 CPU 核心)
        int n_samples = 100000;       // 每条链采样总深度
        int burn_in = 20000;          // 预热期 (不计入统计)
        int thinning = 10;            // 稀疏化步长 (降低自相关性)
        double init_step_size = 0.1;  // 初始扰动 Sigma
        bool adaptive = true;         // 是否开启自适应步长
        int seed = 2026;              // 全局随机种子
        bool return_traces = false;   // 是否回传全量轨迹 (用于画 Trace Plot)

        // --- 似然函数刚度 (Stiffness) ---
        // 物理意义：控制“历史事实”对模型推断的约束强度
        double rank_tau = 0.05;       // Soft-Rank 平滑温度
        double elim_penalty = 1000.0; // 淘汰逻辑惩罚刚度 (Hard Constraint)
        double jeopardy_penalty = 200.0; // 危险区惩罚刚度 (Soft Constraint)
        double entropy_weight = 0.05; // 最大熵正则化权重 (Occam's Razor)
    };

    /**
     * @struct InferenceResult
     * @brief 贝叶斯推断最终报告
     */
    struct InferenceResult {
        Eigen::VectorXd posterior_mean;   // 潜变量后验均值
        Eigen::VectorXd posterior_std;    // 估计不确定性 (StdDev)
        double r_hat;                     // 最大收敛因子 (Max PSRF)
        double ess;                       // 有效样本量估计
        double acceptance_rate;           // 链平均接受率
        double fidelity_score;            // 业务保真度 (淘汰匹配率)
        bool converged;                   // 是否通过收敛审计

        // 全量轨迹数据：[Chain][Sample][Dimension]
        // 仅在 return_traces = true 时填充
        std::vector<std::vector<Eigen::VectorXd>> traces;
    };

    // =========================================================================
    // 2. 核心采样器类
    // =========================================================================

    class MCMCSampler {
    public:
        explicit MCMCSampler(const SamplerConfig& config) : cfg_(config) {}

        /**
         * @brief 并行推断入口
         * 通过 OpenMP 调度 23 路独立采样流，并执行多维度 R-hat 审计。
         */
        InferenceResult run_parallel_inference(
            const Eigen::VectorXd& judge_scores,
            int elim_idx,
            const Eigen::VectorXi& jeopardy_mask,
            const Eigen::VectorXd& prior_mean,
            MechanismType mech_type
        );

    private:
        SamplerConfig cfg_;

        /**
         * @struct ChainState
         * @brief 线程局部状态 (硬件优化版)
         * alignas(64) 强制内存对齐到缓存行，消除 False Sharing，保障 23 核满载效率。
         */
        struct alignas(64) ChainState {
            Eigen::VectorXd current_position;
            double current_log_lik;
            double step_size;
            long long accepted_count = 0;
            std::vector<Eigen::VectorXd> samples;
        };

        /**
         * @brief 单链执行单元 (Metropolis-Hastings Kernel)
         */
        ChainState run_single_chain(
            int thread_id,
            const Eigen::VectorXd& start_pos,
            const LikelihoodEvaluator& evaluator,
            const Eigen::VectorXd& judge_scores,
            int elim_idx,
            const Eigen::VectorXi& jeopardy_mask,
            MechanismType mech_type
        ) const;

        /**
         * @brief 建议分布：单纯形上的对数空间映射游走
         * 物理意义：将有约束的单纯形问题转化为无约束的 R^N 空间高斯游走。
         */
        Eigen::VectorXd propose_log_space_move(
            const Eigen::VectorXd& current,
            double step_size,
            std::mt19937_64& rng
        ) const;

        /**
         * @brief 自适应律 (Target Acceptance = 0.234)
         */
        void adapt_step_size(double& step_size, double recent_acceptance_rate) const;
    };

} // namespace engine
} // namespace mcm

#endif // MCMC_SAMPLER_HPP