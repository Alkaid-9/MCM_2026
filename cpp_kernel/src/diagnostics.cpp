/**
 * @file diagnostics.cpp
 * @brief Implementation of Statistical Auditing Metrics (Industrial v4.0)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [物理意义 - Physical Intuition]:
 * 本模块充当“概率法官”：
 * 1. R-hat: 审计 23 个平行宇宙（采样链）是否最终坍缩成同一个真相。
 * 2. ESS: 审计样本中由于马尔可夫链“自相关性”导致的冗余，计算纯净信息量。
 * 3. Fidelity: 审计反演结果是否“亵渎”了历史淘汰事实（Task 1 核心指标）。
 */

#include "diagnostics.hpp"
#include "types.hpp"
#include "math_utils.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <limits>

namespace mcm {
namespace diag {

    using namespace mcm::types;

    // =========================================================================
    // 1. Gelman-Rubin Statistic (R-hat)
    // =========================================================================
    // 理论依据: Brooks & Gelman (1998)。比较链间方差 (B) 与链内方差 (W)。
    // =========================================================================
    double compute_r_hat(const std::vector<std::vector<double>>& chains_1d) {
        size_t m = chains_1d.size(); // 链的数量
        if (m < 2) return 0.0;       // 单链无法计算 R-hat

        size_t n = chains_1d[0].size(); // 每条链的样本数
        if (n < 10) return 999.0;      // 样本太少，返回极大值表示未收敛

        std::vector<double> chain_means(m);
        std::vector<double> chain_vars(m);

        // A. 计算每条链的均值和方差 (使用无偏估计)
        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            for (double x : chains_1d[i]) sum += x;
            double mean = sum / n;
            chain_means[i] = mean;

            double sq_sum = 0.0;
            for (double x : chains_1d[i]) {
                sq_sum += (x - mean) * (x - mean);
            }
            chain_vars[i] = sq_sum / (n - 1);
        }

        // B. 计算链间方差 B (Between-chain variance)
        double grand_mean = 0.0;
        for (double mu : chain_means) grand_mean += mu;
        grand_mean /= m;

        double B = 0.0;
        for (double mu : chain_means) {
            B += (mu - grand_mean) * (mu - grand_mean);
        }
        B *= static_cast<double>(n) / (m - 1);

        // C. 计算链内平均方差 W (Within-chain variance)
        double W = 0.0;
        for (double var : chain_vars) W += var;
        W /= m;

        // D. 估计边缘后验方差 V_hat
        // V_hat = (n-1)/n * W + 1/n * B
        if (W < constants::EPSILON) return 1.0; // 如果方差极小，视作完美收敛

        double var_plus = (static_cast<double>(n - 1) / n) * W + (B / n);
        return std::sqrt(var_plus / W);
    }

    // =========================================================================
    // 2. Effective Sample Size (ESS)
    // =========================================================================
    // 理论依据: 计算自相关函数 (ACF) 的积分。
    // 物理意义: MCMC 样本不是独立的，ESS 告诉我们这些样本等效于多少个独立样本。
    // =========================================================================
    double compute_ess(const std::vector<double>& chain) {
        size_t n = chain.size();
        if (n < 2) return 0.0;

        // 计算均值和方差
        double sum = 0.0;
        for (double x : chain) sum += x;
        double mean = sum / n;

        double var = 0.0;
        for (double x : chain) var += (x - mean) * (x - mean);
        var /= (n - 1);

        if (var < constants::EPSILON) return static_cast<double>(n);

        // 计算自相关系数 rho_t
        // 使用 Geyer 的初始单调序列估计器的简化版：只累加直到 rho 变为负数或极小
        double sum_rho = 0.0;
        for (size_t t = 1; t < n / 2; ++t) {
            double autocov = 0.0;
            for (size_t i = 0; i < n - t; ++i) {
                autocov += (chain[i] - mean) * (chain[i + t] - mean);
            }
            autocov /= (n - t);
            double rho = autocov / var;

            if (rho < 0.05) break; // 噪声主导时停止
            sum_rho += rho;
        }

        return static_cast<double>(n) / (1.0 + 2.0 * sum_rho);
    }

    // =========================================================================
    // 3. 业务一致性指标：Rank Fidelity Score (Task 1 核心答案)
    // =========================================================================
    // [关键修复]: 参数签名必须严格匹配 diagnostics.hpp 中的 ConstVecRef
    // =========================================================================
    double compute_fidelity(
        ConstVecRef est_votes,
        ConstVecRef judge_scores,
        int elim_idx,
        bool is_percent_rule
    ) {
        // 无淘汰周（如决赛周或开场周），默认完全一致
        if (elim_idx < 0 || elim_idx >= est_votes.size()) return 1.0;

        int n = static_cast<int>(est_votes.size());
        VoteDistribution survival_score(n);

        // A. 重建生存流形 (Survival Manifold)
        if (is_percent_rule) {
            // 百分比制：直接相加。数值越大，表现越好，越安全。
            survival_score = judge_scores + est_votes;
        }
        else {
            // 排名制：计算负的总排名。排名数字越小越好 -> 负排名越大越安全。
            // 使用 Soft-Rank 保证逻辑一致性
            VoteDistribution fan_r = mcm::math::compute_soft_ranks(est_votes, 0.01);
            VoteDistribution judge_r = mcm::math::compute_soft_ranks(judge_scores, 0.01);
            survival_score = -1.0 * (fan_r + judge_r);
        }

        // B. 违规审计 (Violation Audit)
        // 核心逻辑：如果有人生存分比被淘汰者 (elim_idx) 还要低，
        // 说明我们的估计结果无法解释“为什么那个人没走”，这就是一个 Fidelity 损失。
        double loser_val = survival_score[elim_idx];
        int worse_than_loser = 0;

        for (int i = 0; i < n; ++i) {
            if (i == elim_idx) continue;
            // 引入微小容差 (1e-7) 防止浮点数精度导致的误判
            if (survival_score[i] < (loser_val - 1e-7)) {
                worse_than_loser++;
            }
        }

        // Fidelity = 1 - (违背物理事实的人数比例)
        // 1.0 表示完美解释历史；< 0.5 表示模型与现实存在严重冲突。
        return 1.0 - (static_cast<double>(worse_than_loser) / (n - 1));
    }

} // namespace diag
} // namespace mcm