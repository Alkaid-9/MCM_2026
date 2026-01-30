/**
 * MCM 2026 Problem C: Bayesian MCMC Sampler - Header Definition
 * Role: Orchestrating Parallel Metropolis-Hastings Chains on the Probability Simplex.
 * Standard: High-Performance Computing (HPC) / Bayesian Rigor.
 */

#ifndef MCMC_SAMPLER_HPP
#define MCMC_SAMPLER_HPP

#include <vector>
#include <string>
#include <random>
#include <Eigen/Dense>
#include "math_utils.hpp"
#include "diagnostics.hpp"

namespace mcm {
namespace core {

/**
 * 【反演核心类】MCMCSampler
 * 职责：在 23 核并行环境下，通过马尔可夫链探索最符合历史淘汰结果的观众投票分布。
 * 物理约束：所有采样出的投票向量 v 必须满足 \sum v_i = 1 且 v_i > 0。
 */
class MCMCSampler {
public:
    // 构造函数：注入随机种子，初始化超参数
    explicit MCMCSampler(int seed = 2026);

    /**
     * 【主入口】并行推理引擎
     * @param judge_signals: 评委信号矩阵 (Episode-level normalized)
     * @param elim_idx: 当周真实淘汰选手的索引
     * @param prior_mu: 贝叶斯先验均值 (From Zipf's Law)
     * @param mechanism: "RANK" 或 "PERCENT"
     * @return: 后验分布统计结果 (Mean, Std, Entropy, R-hat)
     */
    struct InferenceResult {
        Eigen::VectorXd posterior_mean;      // 估计得票率均值
        Eigen::VectorXd posterior_std;       // 估计标准差 (不确定性指标1)
        double shannon_entropy;              // 香农熵 (不确定性指标2)
        double r_hat;                        // 收敛性指标 (Gelman-Rubin)
        double acceptance_rate;              // 采样接受率
        bool converged;                      // 是否满足收敛红线
    };

    InferenceResult run_parallel_inference(
        const Eigen::VectorXd& judge_signals,
        int elim_idx,
        const Eigen::VectorXd& prior_mu,
        const std::string& mechanism,
        int n_chains = 23,                   // 适配 23 核并行
        int n_samples = 100000,
        double jump_size = 0.05
    );

private:
    int seed_;

    /**
     * 【似然函数内核】Likelihood(Outcome | Votes)
     * 逻辑：如果当前投票组合 v 配合评委信号 j 能导致 elim_idx 排名垫底，则似然高；反之极低。
     */
    double compute_log_likelihood(
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& j,
        int elim_idx,
        const std::string& mech
    );

    /**
     * 【建议分布】Simplex Random Walk
     * 物理意义：在概率单纯形上生成一个新的、满足总和为 1 的投票向量。
     */
    Eigen::VectorXd propose_next_state(
        const Eigen::VectorXd& current_v,
        double jump_size,
        std::mt19937& gen
    );
};

} // namespace core
} // namespace mcm

#endif // MCMC_SAMPLER_HPP