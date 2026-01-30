/**
 * MCM 2026 Problem C: Bayesian MCMC Sampler - Header Definition
 * Role: Orchestrating Parallel Metropolis-Hastings Chains on the Probability Simplex.
 * Standard: High-Performance Computing (HPC) / Bayesian Rigor / 23-Core Ready.
 */

#ifndef MCMC_SAMPLER_HPP
#define MCMC_SAMPLER_HPP

#include <vector>
#include <string>
#include <random>
#include <Eigen/Dense>
#include "math_utils.hpp"

namespace mcm {
namespace core {

/**
 * @class MCMCSampler
 * @brief 隐变量反演核心引擎
 *
 * 职责：
 * 1. 在受约束的单纯形空间（Simplex）执行随机游走。
 * 2. 结合评委信号（Judge Signals）与危机信号（Jeopardy Mask）计算后验概率。
 * 3. 利用 OpenMP 实现多链并行，并计算收敛性指标。
 */
class MCMCSampler {
public:
    /**
     * @struct InferenceResult
     * @brief 存储贝叶斯推断的统计全家桶
     */
    struct InferenceResult {
        Eigen::VectorXd posterior_mean;  // 后验均值（估算的粉丝票数占比）
        Eigen::VectorXd posterior_std;   // 后验标准差（不确定性度量 1）
        double shannon_entropy;          // 香农熵（不确定性度量 2）
        double r_hat;                    // Gelman-Rubin 指标（收敛审计）
        double acceptance_rate;          // 采样接受率
        bool converged;                  // 是否通过收敛红线
    };

    // 构造函数：注入随机种子，初始化状态
    explicit MCMCSampler(int seed = 2026);

    /**
     * @brief 并行推理引擎入口 (8参数对齐版)
     *
     * @param judge_signals  评委打分的 Z-Score 向量
     * @param elim_idx       当周真实淘汰者的索引 (-1 代表无淘汰周)
     * @param jeopardy_mask  危险区/倒数两名标记向量 (1-在危险区, 0-安全)
     * @param prior_mu       贝叶斯先验均值向量 (来自 Zipf's Law)
     * @param mechanism      赛制类型 ("RANK" 或 "PERCENT")
     * @param n_chains       并行链数量 (默认适配 23 核)
     * @param n_samples      每条链的采样深度
     * @param jump_size      Metropolis 步长因子
     */
    InferenceResult run_parallel_inference(
        const Eigen::VectorXd& judge_signals,
        int elim_idx,
        const Eigen::VectorXi& jeopardy_mask,
        const Eigen::VectorXd& prior_mu,
        const std::string& mechanism,
        int n_chains = 23,
        int n_samples = 100000,
        double jump_size = 0.05
    );

private:
    int seed_; // 基础随机种子

    /**
     * @brief 核心似然函数内核
     * 逻辑：Outcome ~ Combined_Score(Judge, Fan_Vote)
     */
    double compute_log_likelihood(
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& j,
        int elim_idx,
        const Eigen::VectorXi& jeopardy_mask,
        const std::string& mech
    );

    /**
     * @brief 建议分布：单纯形上的随机游走
     * 物理意义：在 log-space 扰动后通过 Softmax 投影回单纯形
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