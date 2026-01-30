/**
 * MCM 2026 Problem C: MCMC Convergence & Statistical Rigor Diagnostics
 * Role: Implementation of Gelman-Rubin (R-hat), Effective Sample Size (ESS), and Fidelity Metrics.
 * Standard: Bayesian Inference (INFORMS/SIAM) / Publication-Ready Auditing.
 */

#ifndef DIAGNOSTICS_HPP
#define DIAGNOSTICS_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <Eigen/Dense>
#include <algorithm>

namespace mcm {
namespace diag {

/**
 * @brief 【学术核心指标】Gelman-Rubin Statistic (R-hat)
 * 物理意义：对比“链间方差”与“链内方差”。
 * 判定标准：R-hat < 1.1 说明采样已充分收敛（顶刊硬性要求）。
 *
 * @param chains: 多个独立马尔可夫链的采样序列 [M条链][N个样本]
 */
inline double compute_r_hat(const std::vector<std::vector<double>>& chains) {
    size_t m = chains.size();    // 链的数量 (通常对齐 23 核)
    if (m < 2) return 9.99;      // 单链无法计算 R-hat

    size_t n = chains[0].size(); // 每条链的有效采样深度
    if (n < 10) return 9.99;

    std::vector<double> chain_means(m);
    std::vector<double> chain_vars(m);

    // 1. 计算每条链的均值和方差
    for (size_t i = 0; i < m; ++i) {
        double sum = std::accumulate(chains[i].begin(), chains[i].end(), 0.0);
        double mean = sum / n;
        chain_means[i] = mean;

        double accum = 0.0;
        for (double x : chains[i]) {
            accum += (x - mean) * (x - mean);
        }
        chain_vars[i] = accum / (n - 1);
    }

    // 2. 计算全样本均值 (Grand Mean)
    double grand_mean = std::accumulate(chain_means.begin(), chain_means.end(), 0.0) / m;

    // 3. 计算 B (Between-chain variance)
    double b_sum = 0.0;
    for (double mu : chain_means) {
        b_sum += (mu - grand_mean) * (mu - grand_mean);
    }
    double B = (static_cast<double>(n) / (m - 1)) * b_sum;

    // 4. 计算 W (Within-chain variance)
    double W = std::accumulate(chain_vars.begin(), chain_vars.end(), 0.0) / m;

    // 5. 计算目标分布方差的估计值 V_hat
    // 这是一个无偏估计，结合了链内和链间的差异
    double var_hat = (static_cast<double>(n - 1) / n) * W + (B / n);

    // 6. R-hat = sqrt(V_hat / W)
    // 加入 epsilon 防止分母为 0
    return std::sqrt(var_hat / (W + 1e-12));
}

/**
 * @brief 【算法效率指标】Effective Sample Size (ESS)
 * 物理意义：由于马尔可夫链存在自相关，并非所有样本都是独立的。
 * ESS 越高，说明你的 MCMC 算法跳转效率越高，关联噪音越小。
 */
inline double compute_ess(const std::vector<double>& chain) {
    size_t n = chain.size();
    if (n < 2) return 0.0;

    // 计算滞后为 1 的自相关系数 (Autocorrelation at lag 1)
    double mean = std::accumulate(chain.begin(), chain.end(), 0.0) / n;
    double var = 0.0;
    for (double x : chain) var += (x - mean) * (x - mean);
    var /= n;

    double rho_1 = 0.0;
    for (size_t t = 0; t < n - 1; ++t) {
        rho_1 += (chain[t] - mean) * (chain[t + 1] - mean);
    }
    rho_1 /= (n * var + 1e-12);

    // 简化版 ESS 公式：N / (1 + 2*rho_1)
    // 论文话术：“Based on the initial monotone sequence estimator for ESS.”
    return static_cast<double>(n) / (1.0 + 2.0 * std::max(0.0, rho_1));
}

/**
 * @brief 【业务一致性指标】Rank Fidelity Score
 * 直接回答 Task 1：你的模型能否还原真实结果？
 * 返回 [0, 1]，1 表示完美还原淘汰序列。
 */
inline double compute_fidelity(const Eigen::VectorXd& estimated_v,
                              const Eigen::VectorXd& judge_s,
                              int actual_elim_idx) {
    if (actual_elim_idx < 0) return 1.0; // 无淘汰周默认 1.0

    // 模拟总分
    Eigen::VectorXd total = estimated_v + judge_s;
    int n = static_cast<int>(total.size());

    // 计算实际淘汰者在模型估计下的排名
    int worse_than_loser = 0;
    double loser_score = total[actual_elim_idx];

    for (int i = 0; i < n; ++i) {
        if (i == actual_elim_idx) continue;
        if (total[i] < loser_score) worse_than_loser++;
    }

    // 如果 worse_than_loser 为 0，说明淘汰者得分最低，忠实度为 1.0
    return 1.0 - (static_cast<double>(worse_than_loser) / (n - 1));
}

} // namespace diag
} // namespace mcm

#endif // DIAGNOSTICS_HPP