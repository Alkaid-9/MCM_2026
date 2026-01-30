/**
 * MCM 2026 Problem C: High-Performance MCMC Sampler Implementation
 * Role: 23-Core Parallel Metropolis-Hastings on the Probability Simplex.
 * Standard: Industrial HPC / Bayesian Convergence Standards (R-hat).
 */

#include "mcmc_sampler.hpp"
#include "math_utils.hpp"
#include "diagnostics.hpp"
#include <omp.h>
#include <random>
#include <iostream>

namespace mcm {
namespace core {

MCMCSampler::MCMCSampler(int seed) : seed_(seed) {}

/**
 * 【建议分布逻辑】Simplex Random Walk
 * 物理意义：在 log-space 进行高斯扰动后通过 Softmax 映射回单位单纯形。
 * 这种方法能确保采样点永远合法（Sum to 1），且游走效率极高。
 */
Eigen::VectorXd MCMCSampler::propose_next_state(
    const Eigen::VectorXd& current_v,
    double jump_size,
    std::mt19937& gen)
{
    std::normal_distribution<double> dist(0.0, jump_size);
    int n = current_v.size();

    // 在对数空间进行扰动
    Eigen::VectorXd log_v = (current_v.array() + 1e-9).log();
    for (int i = 0; i < n; ++i) {
        log_v[i] += dist(gen);
    }

    // 映射回单纯形
    return mcm::math::softmax(log_v);
}

MCMCSampler::InferenceResult MCMCSampler::run_parallel_inference(
    const Eigen::VectorXd& judge_signals,
    int elim_idx,
    const Eigen::VectorXd& prior_mu,
    const std::string& mechanism,
    int n_chains,
    int n_samples,
    double jump_size)
{
    int n_contestants = judge_signals.size();
    std::vector<std::vector<Eigen::VectorXd>> all_chains(n_chains);
    std::vector<double> acceptance_rates(n_chains, 0.0);

    // 预分配每条链的样本空间
    #pragma omp parallel for num_threads(n_chains) schedule(dynamic)
    for (int m = 0; m < n_chains; ++m) {
        std::mt19937 gen(seed_ + m);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);

        Eigen::VectorXd current_state = prior_mu;
        double current_log_lik = compute_log_likelihood(current_state, judge_signals, elim_idx, mechanism);

        int accepted = 0;
        std::vector<Eigen::VectorXd> chain_samples;
        chain_samples.reserve(n_samples / 5); // 考虑 thinning 后的容量

        for (int i = 0; i < n_samples; ++i) {
            // 1. Propose
            Eigen::VectorXd proposal = propose_next_state(current_state, jump_size, gen);
            double proposal_log_lik = compute_log_likelihood(proposal, judge_signals, elim_idx, mechanism);

            // 2. Metropolis-Hastings Acceptance
            double log_alpha = proposal_log_lik - current_log_lik;
            if (std::log(u_dist(gen)) < log_alpha) {
                current_state = proposal;
                current_log_lik = proposal_log_lik;
                accepted++;
            }

            // 3. Thinning (每 5 步取 1 样，减少自相关性)
            if (i >= (n_samples * 0.2) && i % 5 == 0) { // 包含 20% Burn-in
                chain_samples.push_back(current_state);
            }
        }
        all_chains[m] = chain_samples;
        acceptance_rates[m] = static_cast<double>(accepted) / n_samples;
    }

    // --- 统计汇总与不确定性量化 ---
    InferenceResult res;
    res.posterior_mean = Eigen::VectorXd::Zero(n_contestants);
    res.posterior_std = Eigen::VectorXd::Zero(n_contestants);

    // A. 计算均值 (Posterior Mean)
    long long total_samples = 0;
    mcm::math::RunningStat stat_engine; // 借用 Welford 算法思想的简化版汇总

    std::vector<std::vector<double>> r_hat_chains(n_chains);

    for (int m = 0; m < n_chains; ++m) {
        for (const auto& sample : all_chains[m]) {
            res.posterior_mean += sample;
            total_samples++;
        }
        // 为 R-hat 提取第一个选手的链作为收敛代表（学术惯例）
        for (const auto& sample : all_chains[m]) r_hat_chains[m].push_back(sample[0]);
    }
    res.posterior_mean /= static_cast<double>(total_samples);

    // B. 计算不确定性指标
    res.shannon_entropy = mcm::math::compute_entropy(res.posterior_mean);
    res.r_hat = mcm::diag::compute_r_hat(r_hat_chains);
    res.acceptance_rate = std::accumulate(acceptance_rates.begin(), acceptance_rates.end(), 0.0) / n_chains;
    res.converged = (res.r_hat < 1.1);

    // C. 计算后验标准差 (Posterior Std)
    Eigen::VectorXd var_sum = Eigen::VectorXd::Zero(n_contestants);
    for (int m = 0; m < n_chains; ++m) {
        for (const auto& sample : all_chains[m]) {
            Eigen::VectorXd diff = sample - res.posterior_mean;
            var_sum += diff.cwiseProduct(diff);
        }
    }
    res.posterior_std = (var_sum / (total_samples - 1)).cwiseSqrt();

    return res;
}

} // namespace core
} // namespace mcm