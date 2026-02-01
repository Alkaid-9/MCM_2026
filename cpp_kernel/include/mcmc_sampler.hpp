/**
 * @file mcmc_sampler.hpp
 * @brief High-Performance Parallel MCMC Engine Interface (v4.6 - Full-Rank Consistency)
 * @details Implements Adaptive Metropolis-Hastings on the Simplex with Full-Rank Anchor.
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

    /**
     * @struct SamplerConfig
     * @brief 采样器全量超参数配置 (对齐 rules.yaml)
     */
    struct SamplerConfig {
        // --- 采样控制 (Python 侧输入) ---
        int n_chains = 23;            // 并行链数量
        int n_samples = 100000;       // 每条链的总样本量
        double burn_in_ratio = 0.5;   // 预热期比例
        int thinning = 10;            // 稀疏化间隔
        double init_step_size = 0.1;  // 初始跳跃步长
        bool adaptive = true;         // 是否开启 Dual-Averaging
        int seed = 2026;              // 全局随机种子
        bool return_traces = false;   // 是否回传全量轨迹

        // --- 能量函数形状参数 (Stiffness) ---
        double rank_tau = 0.05;       // Soft-Rank 平滑温度
        double elim_penalty = 1200.0; // 淘汰违规惩罚强度
        double jeopardy_penalty = 150.0; // 危险区惩罚强度

        // --- 贝叶斯先验与逻辑开关 ---
        double prior_strength = 50.0; // 贝叶斯先验场引力强度
        bool enable_judge_save = false; // 是否启用第 28 季后的评委救济逻辑
    };

    /**
     * @struct InferenceResult
     * @brief 贝叶斯推断后验统计报告
     */
    struct InferenceResult {
        Eigen::VectorXd posterior_mean; // 潜变量后验均值 (估计票数占比)
        Eigen::VectorXd posterior_std;  // 估计不确定性 (标准差)
        double r_hat;                   // Split-R-hat 收敛指标
        double ess;                     // 有效样本量 (Effective Sample Size)
        double acceptance_rate;         // 链平均接受率 (Target ~0.234)
        double fidelity_score;          // 业务逻辑还原保真度
        bool converged;                 // 是否通过收敛性审计

        // 采样轨迹数据：[ChainID][SampleID][Dimension]
        std::vector<std::vector<Eigen::VectorXd>> traces;
    };

    class MCMCSampler {
    public:
        explicit MCMCSampler(const SamplerConfig& config) : cfg_(config) {}

        /**
         * @brief 并行推断总入口 (Python 调用点)
         *
         * @param winner_idx 冠军选手索引。若为 -1，则不执行冠军势能井约束。
         */
        InferenceResult run_parallel_inference(
            const Eigen::VectorXd& judge_scores,
            int elim_idx,
            const Eigen::VectorXi& jeopardy_mask,
            const Eigen::VectorXd& prior_mean,
            MechanismType mech_type,
            int winner_idx = -1 // <--- [New] 对齐 Python 接口
        );

        /**
         * @struct ChainState
         * @brief 单条马尔可夫链的局部状态
         * alignas(64) 确保每个线程的 State 独占缓存行，防止多核 False Sharing 性能损耗。
         */
        struct alignas(64) ChainState {
            Eigen::VectorXd current_position;
            double current_log_lik;
            double step_size;
            long long accepted_count = 0;
            std::vector<Eigen::VectorXd> samples;
        };

    private:
        SamplerConfig cfg_;

        /**
         * @brief 单链采样核心循环 (Metropolis-Hastings Kernel)
         */
        ChainState run_single_chain(
            int thread_id,
            const Eigen::VectorXd& start_pos,
            const LikelihoodEvaluator& evaluator,
            const Eigen::VectorXd& judge_scores,
            int elim_idx,
            const Eigen::VectorXi& jeopardy_mask,
            const Eigen::VectorXd& prior_mean,
            MechanismType mech_type,
            int winner_idx // <--- [New] 必须同步注入
        ) const;

        /**
         * @brief 建议分布：在对数空间执行游走并投影回单纯形
         */
        Eigen::VectorXd propose_log_space_move(
            const Eigen::VectorXd& current,
            double step_size,
            std::mt19937_64& rng
        ) const;

        /**
         * @brief Dual-Averaging 自适应律实现
         */
        void adapt_step_size(double& step_size, double recent_acceptance_rate) const;
    };

} // namespace engine
} // namespace mcm

#endif // MCMC_SAMPLER_HPP