/**
 * MCM 2026 Problem C: High-Performance MCMC Sampler Implementation
 * Role: 23-Core Parallel Metropolis-Hastings on the Probability Simplex.
 * Standard: Industrial HPC / Bayesian Convergence Standards (Gelman-Rubin R-hat).
 */

#include "mcmc_sampler.hpp"
#include "math_utils.hpp"
#include "diagnostics.hpp" // 必须包含，用于收敛性审计
#include <omp.h>
#include <random>
#include <iostream>
#include <vector>
#include <numeric>

namespace mcm {
namespace core {

MCMCSampler::MCMCSampler(int seed) : seed_(seed) {}

/**
 * @brief 建议分布：单纯形上的对数空间随机游走
 * 物理意义：在无约束的 log-space 进行扰动，通过 Softmax 投影回 Sum=1 的单纯形。
 * 这种方法能自适应处理边界，避免产生负票数。
 */
Eigen::VectorXd MCMCSampler::propose_next_state(
    const Eigen::VectorXd& current_v,
    double jump_size,
    std::mt19937& gen)
{
    std::normal_distribution<double> dist(0.0, jump_size);
    int n = static_cast<int>(current_v.size());

    // 映射到对数空间 (加上极小值防止 log(0))
    Eigen::VectorXd log_v = (current_v.array() + 1e-9).log();

    // 注入各向同性高斯噪声
    for (int i = 0; i < n; ++i) {
        log_v[i] += dist(gen);
    }

    // 投影回单纯形
    return mcm::math::softmax(log_v);
}

MCMCSampler::InferenceResult MCMCSampler::run_parallel_inference(
    const Eigen::VectorXd& judge_signals,
    int elim_idx,
    const Eigen::VectorXi& jeopardy_mask,
    const Eigen::VectorXd& prior_mu,
    const std::string& mechanism,
    int n_chains,
    int n_samples,
    double jump_size)
{
    // --- 1. 初始化容器 ---
    const int n_contestants = static_cast<int>(judge_signals.size());
    std::vector<std::vector<Eigen::VectorXd>> all_chains(n_chains);
    std::vector<double> acceptance_rates(n_chains, 0.0);

    // --- 2. OpenMP 23 核并行采样点火 ---
    // 每个线程处理一条独立的马尔可夫链
    #pragma omp parallel for num_threads(n_chains) schedule(dynamic)
    for (int m = 0; m < n_chains; ++m) {
        // 使用独立随机数种子，确保链的独立性
        std::mt19937 gen(seed_ + m);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);

        Eigen::VectorXd current_state = prior_mu;
        double current_log_lik = compute_log_likelihood(current_state, judge_signals, elim_idx, jeopardy_mask, mechanism);

        int accepted = 0;
        std::vector<Eigen::VectorXd> chain_samples;
        chain_samples.reserve(n_samples / 5); // 预估 Thinning 后的容量

        for (int i = 0; i < n_samples; ++i) {
            // A. Propose: 生成新的候选状态
            Eigen::VectorXd proposal = propose_next_state(current_state, jump_size, gen);
            double proposal_log_lik = compute_log_likelihood(proposal, judge_signals, elim_idx, jeopardy_mask, mechanism);

            // B. Metropolis-Hastings 接受/拒绝判据
            double log_alpha = proposal_log_lik - current_log_lik;
            if (std::log(u_dist(gen)) < log_alpha) {
                current_state = proposal;
                current_log_lik = proposal_log_lik;
                accepted++;
            }

            // C. Burn-in 与 Thinning 处理
            // 丢弃前 20% 样本，每 5 步取 1 样，降低自相关性
            if (i >= (n_samples * 0.2) && i % 5 == 0) {
                chain_samples.push_back(current_state);
            }
        }
        all_chains[m] = chain_samples;
        acceptance_rates[m] = static_cast<double>(accepted) / n_samples;
    }

    // --- 3. 统计汇总与后验量化 ---
    InferenceResult res;
    res.posterior_mean = Eigen::VectorXd::Zero(n_contestants);
    res.posterior_std = Eigen::VectorXd::Zero(n_contestants);

    long long total_valid_samples = 0;
    // 准备 R-hat 计算用的多链容器
    std::vector<std::vector<double>> r_hat_chains(n_chains);

    // A. 遍历所有并行链进行汇总
    for (int m = 0; m < n_chains; ++m) {
        for (const auto& sample : all_chains[m]) {
            res.posterior_mean += sample;
            // 提取第一维度作为收敛性代表 (Gelman-Rubin 标准做法)
            r_hat_chains[m].push_back(sample[0]);
            total_valid_samples++;
        }
    }

    if (total_valid_samples > 0) {
        res.posterior_mean /= static_cast<double>(total_valid_samples);
    }

    // B. 计算贝叶斯核心指标
    // 此时 mcm::diag 已经通过 diagnostics.hpp 可见
    res.r_hat = mcm::diag::compute_r_hat(r_hat_chains);
    res.shannon_entropy = mcm::math::compute_entropy(res.posterior_mean);
    res.acceptance_rate = std::accumulate(acceptance_rates.begin(), acceptance_rates.end(), 0.0) / n_chains;
    res.converged = (res.r_hat < 1.1);

    // C. 计算后验标准差 (Posterior Std)
    // 用于量化 Task 1 中要求的“对估计结果的把握程度”
    Eigen::VectorXd var_sum = Eigen::VectorXd::Zero(n_contestants);
    for (int m = 0; m < n_chains; ++m) {
        for (const auto& sample : all_chains[m]) {
            Eigen::VectorXd diff = sample - res.posterior_mean;
            var_sum += diff.cwiseProduct(diff);
        }
    }

    if (total_valid_samples > 1) {
        res.posterior_std = (var_sum / (total_valid_samples - 1)).cwiseSqrt();
    }

    return res;
}

} // namespace core
} // namespace mcm