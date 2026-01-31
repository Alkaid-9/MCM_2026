/**
 * @file mcmc_sampler.hpp
 * @brief High-Performance Parallel MCMC Engine Interface (Bayesian v4.5)
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
     * @brief 采样器全量超参数配置
     * [物理对齐]: 这里的字段必须与 bindings.cpp 和 rules.yaml 完全对应
     */
    struct SamplerConfig {
        // --- 采样控制 ---
        int n_chains = 23;
        int n_samples = 100000;
        double burn_in_ratio = 0.5; // [修正]: 使用比例而非绝对步数
        int thinning = 10;
        double init_step_size = 0.1;
        bool adaptive = true;
        int seed = 2026;
        bool return_traces = false;

        // --- 似然函数刚度 (Stiffness) ---
        double rank_tau = 0.05;
        double elim_penalty = 1200.0;
        double jeopardy_penalty = 150.0;

        // --- [核心补齐]: 贝叶斯先验与机制逻辑 ---
        double prior_strength = 50.0;  // <--- 解决编译错误：prior_strength
        bool enable_judge_save = false; // <--- 解决编译错误：enable_judge_save
    };

    /**
     * @struct InferenceResult
     * @brief 贝叶斯推断最终报告
     */
    struct InferenceResult {
        Eigen::VectorXd posterior_mean;
        Eigen::VectorXd posterior_std;
        double r_hat;
        double ess;
        double acceptance_rate;
        double fidelity_score;
        bool converged;
        std::vector<std::vector<Eigen::VectorXd>> traces;
    };

    class MCMCSampler {
    public:
        explicit MCMCSampler(const SamplerConfig& config) : cfg_(config) {}

        /**
         * @brief 并行推断入口
         */
        InferenceResult run_parallel_inference(
            const Eigen::VectorXd& judge_scores,
            int elim_idx,
            const Eigen::VectorXi& jeopardy_mask,
            const Eigen::VectorXd& prior_mean,
            MechanismType mech_type
        );

        // [修正]: 将 ChainState 移出 private 或设为 public，以便 run_parallel_inference 使用
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
         * @brief 单链执行单元 (Metropolis-Hastings Kernel)
         * [修正]: 这里的签名必须与 .cpp 完全对齐 (8个参数)
         */
        ChainState run_single_chain(
            int thread_id,
            const Eigen::VectorXd& start_pos,
            const LikelihoodEvaluator& evaluator,
            const Eigen::VectorXd& judge_scores,
            int elim_idx,
            const Eigen::VectorXi& jeopardy_mask,
            const Eigen::VectorXd& prior_mean, // <--- 参数 7
            MechanismType mech_type            // <--- 参数 8
        ) const;

        /**
         * @brief 建议分布：单纯形上的对数空间映射游走
         */
        Eigen::VectorXd propose_log_space_move(
            const Eigen::VectorXd& current,
            double step_size,
            std::mt19937_64& rng
        ) const;

        /**
         * @brief 自适应律
         */
        void adapt_step_size(double& step_size, double recent_acceptance_rate) const;
    };

} // namespace engine
} // namespace mcm

#endif // MCMC_SAMPLER_HPP