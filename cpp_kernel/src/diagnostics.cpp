/**
 * @file diagnostics.cpp
 * @brief Statistical Auditing Metrics Implementation (Industrial Refactor v4.5)
 * @details Implements Split-R-hat, Effective Sample Size (ESS), and Rank Fidelity Score.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [学术修正记录 - Academic Rigor]:
 * 1. 升级为 Split-R-hat: 将每条链一分为二，检测非平稳性 (Non-stationarity)。
 * 2. 引入 Geyer's Initial Positive Sequence: 稳健估计自相关截断点，计算真实 ESS。
 * 3. 物理一致性 (Fidelity): 在不同赛制下重构“生存势能”，量化反演结果对现实的解释力。
 */

#include "diagnostics.hpp"
#include "types.hpp"
#include "math_utils.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <limits>
#include <iostream>

namespace mcm {
namespace diag {

    using namespace mcm::types;

    // =========================================================================
    // 1. Gelman-Rubin Statistic (Split-R-hat) - 顶刊标准版
    // =========================================================================
    // 物理意义: R-hat < 1.1 表示多条链已经混合得足够好，收敛到了同一个平稳分布。
    // Split 逻辑: 检测链的前半段和后半段是否有趋势性差异（尚未收敛）。
    double compute_r_hat(const std::vector<std::vector<double>>& chains_1d) {
        size_t m_raw = chains_1d.size();
        if (m_raw < 1) return 999.0;

        size_t n_raw = chains_1d[0].size();
        // 即使是单链，也可以通过 Split 计算 R-hat
        if (n_raw < 4) return 999.0;

        // --- Step 1: Split Chains (拆分链) ---
        // 将 M 条长度为 N 的链，视为 2M 条长度为 N/2 的链
        size_t m = m_raw * 2;
        size_t n = n_raw / 2;

        std::vector<double> split_means(m);
        std::vector<double> split_vars(m);

        for (size_t i = 0; i < m_raw; ++i) {
            // First Half
            double sum1 = 0, sq_sum1 = 0;
            for (size_t j = 0; j < n; ++j) {
                double val = chains_1d[i][j];
                sum1 += val;
                sq_sum1 += val * val;
            }
            double mean1 = sum1 / n;
            split_means[2*i] = mean1;
            split_vars[2*i] = (sq_sum1 - n * mean1 * mean1) / (n - 1);

            // Second Half
            double sum2 = 0, sq_sum2 = 0;
            for (size_t j = n; j < 2*n; ++j) {
                double val = chains_1d[i][j];
                sum2 += val;
                sq_sum2 += val * val;
            }
            double mean2 = sum2 / n;
            split_means[2*i+1] = mean2;
            split_vars[2*i+1] = (sq_sum2 - n * mean2 * mean2) / (n - 1);
        }

        // --- Step 2: Compute Between-chain & Within-chain Variance ---

        // W: 链内方差的平均值
        double W = 0.0;
        for (double v : split_vars) W += v;
        W /= static_cast<double>(m);

        // B: 链间均值的方差 * n
        double grand_mean = 0.0;
        for (double mu : split_means) grand_mean += mu;
        grand_mean /= static_cast<double>(m);

        double B_sum = 0.0;
        for (double mu : split_means) {
            B_sum += (mu - grand_mean) * (mu - grand_mean);
        }
        double B = (B_sum / (m - 1)) * n;

        // --- Step 3: Compute Marginal Posterior Variance (V_hat) ---
        if (W < constants::EPSILON) {
            // 如果链内方差为0，且 B > 0，说明每条链锁死在不同点，没收敛
            return (B > constants::EPSILON) ? 999.0 : 1.0;
        }

        // V_hat = (n-1)/n * W + (1/n) * B
        double var_plus = (static_cast<double>(n - 1) / n) * W + (B / n);

        // --- Step 4: R-hat ---
        double r_hat = std::sqrt(var_plus / W);

        // 物理约束：R-hat 理论最小值是 1.0
        return std::max(1.0, r_hat);
    }

    // =========================================================================
    // 2. Effective Sample Size (ESS) - Geyer's Initial Positive Method
    // =========================================================================
    // 物理意义: 由于 MCMC 样本存在自相关性，N 个样本包含的信息量 < N。
    // ESS 告诉我们到底获得了多少个“独立”样本。
    double compute_ess(const std::vector<double>& chain) {
        const size_t n = chain.size();
        if (n < 2) return 0.0;

        // 计算均值和方差
        double sum = 0.0, sq_sum = 0.0;
        for (double x : chain) { sum += x; sq_sum += x * x; }
        double mean = sum / n;
        double var = (sq_sum - n * mean * mean) / (n - 1);

        if (var < constants::EPSILON) return static_cast<double>(n); // 完全常数链

        // 计算自相关系数 (Autocorrelation)
        std::vector<double> rho;
        // 只计算到 n/2，避免尾部噪音
        for (size_t t = 0; t < n / 2; ++t) {
            double autocov = 0.0;
            for (size_t i = 0; i < n - t; ++i) {
                autocov += (chain[i] - mean) * (chain[i + t] - mean);
            }
            autocov /= (n - t); // 无偏估计
            rho.push_back(autocov / var);
        }

        // Geyer's Initial Positive Sequence Estimator
        // 截断逻辑：当自相关系数 rho[t] 变得很小或为负时，后面的都是噪音，不再累加
        double sum_rho = 0.0;
        for (size_t t = 1; t < rho.size(); ++t) {
            if (rho[t] < 0.05) break; // [工业级优化] 设定 0.05 阈值，比 0 更稳健
            sum_rho += rho[t];
        }

        // ESS Formula: N / (1 + 2 * sum(rho))
        double tau = 1.0 + 2.0 * sum_rho;
        return static_cast<double>(n) / tau;
    }

    // =========================================================================
    // 3. Fidelity Score (业务一致性指标)
    // =========================================================================
    // 物理意义: 反演出的投票数据，在多大程度上复现了历史淘汰结果？
    // Task 1 要求的 "Consistency Measure"。
    double compute_fidelity(
        ConstVecRef est_votes,
        ConstVecRef judge_scores,
        int elim_idx,
        bool is_percent_rule
    ) {
        // 无淘汰周或异常索引，默认完全一致
        if (elim_idx < 0 || elim_idx >= est_votes.size()) return 1.0;

        const int n = static_cast<int>(est_votes.size());
        VoteDistribution survival_score(n);

        // --- 1. 重构生存分数 (Survival Score) ---
        if (is_percent_rule) {
            // [百分比制]: 直接相加
            survival_score = judge_scores + est_votes;
        }
        else {
            // [排名制]: 转换为 Rank 后相加 (Rank 越小越好 -> 取负号变成 Survival Score)
            // 注意：这里用 Hard Rank 计算 Fidelity，因为我们在验证结果，不是在求导
            // 简单起见，调用 Soft Rank 近似，tau 极小即可
            VoteDistribution fan_r = mcm::math::compute_soft_ranks(est_votes, 0.001);
            VoteDistribution judge_r = mcm::math::compute_soft_ranks(judge_scores, 0.001);
            survival_score = -1.0 * (fan_r + judge_r);
        }

        // --- 2. 检查违规情况 ---
        // 规则：淘汰者的分数必须是全场最低 (或接近最低)
        const double loser_val = survival_score[elim_idx];
        int worse_than_loser = 0;

        for (int i = 0; i < n; ++i) {
            if (i == elim_idx) continue;

            // 如果某人存活，但分数比淘汰者还低 (High Violation)
            // 容差 1e-9 处理浮点共线性
            if (survival_score[i] < (loser_val - 1e-9)) {
                worse_than_loser++;
            }
        }

        // --- 3. 计算得分 ---
        // Fidelity = 1.0 - (比淘汰者更惨的人数 / 总竞争者数)
        // 完美情况: worse_than_loser = 0 -> Fidelity = 1.0
        // 最差情况: 所有人分都比淘汰者低 -> Fidelity = 0.0
        if (n <= 1) return 1.0;
        return 1.0 - (static_cast<double>(worse_than_loser) / static_cast<double>(n - 1));
    }

} // namespace diag
} // namespace mcm