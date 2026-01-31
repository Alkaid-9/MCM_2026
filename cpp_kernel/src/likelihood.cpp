/**
 * @file likelihood.cpp
 * @brief Core Likelihood Engine for Bayesian Inverse Optimization (Industrial Refactor)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [物理意义 - Physics Interpretation]:
 * 本模块负责构建参数空间中的“能量景观” (Energy Landscape)。
 * P(v | Outcome) \propto exp(-E(v))
 *
 * 能量 E(v) 由三部分势能叠加而成：
 * 1. E_elim (淘汰势能): 历史事实形成的深井。被淘汰者的生存分必须处于低位。
 *    - 对于 Rank 制：使用 Soft-Rank 算子实现可导的排序约束。
 *    - 对于 Save 机制：放宽势能井，允许淘汰者处于倒数第二的位置。
 * 2. E_jeopardy (危机势能): "Bottom Two" 信号形成的软约束，将相关选手推向低分。
 * 3. E_entropy (负熵势能): 热力学约束，防止模型过拟合，倾向于“平坦”的选票分布。
 */

#include "likelihood.hpp"
#include "math_utils.hpp"
#include "types.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace mcm {
namespace engine {

    using namespace mcm::types;

    // =========================================================================
    // 核心接口: 计算总对数似然 ln P(Evidence | LatentVotes)
    // =========================================================================
    Real LikelihoodEvaluator::compute_log_likelihood(
        ConstVecRef fan_votes_share,
        ConstVecRef judge_scores,
        int elim_idx,
        ConstIntVecRef jeopardy_mask,
        MechanismType mech_type
    ) const {
        // --- 1. 物理合法性防御 (Stability Guard) ---
        // 任何 NaN 或 Inf 都会导致能量场崩塌，必须立即拦截
        if (!fan_votes_share.allFinite()) return constants::NEG_INF;

        Real log_lik = 0.0;
        long n = fan_votes_share.size();

        // --- 2. 构建统一生存分数 (Survival Score Construction) ---
        // 目标：将不同赛制下的表现统一为 "Higher is Better" 的连续标量
        VoteDistribution survival_score(n);

        if (mech_type == MechanismType::PERCENT_BASED) {
            // [百分比制 S3-S27]
            // 公式: Survival = Judge% + Fan%
            // 评委分 judge_scores 已经预处理为占比形式 (Sum=1) 或需要在此处归一化
            // 假设 ETL 层已传入归一化的 Z-Score 或占比，此处直接叠加
            survival_score = judge_scores + fan_votes_share;
        }
        else {
            // [排名制 S1-S2, S28+]
            // 公式: Total_Rank = Rank(Judge) + Rank(Fan)
            // 原始逻辑: Rank 1 (Min) 是最好的。
            // 转换逻辑: 取负号，转化为 Higher is Better。

            // A. 计算粉丝票的软排名 (Soft-Rank)
            // 使用 Sigmoid 平滑近似，使梯度可传导
            VoteDistribution fan_ranks = mcm::math::compute_soft_ranks(fan_votes_share, cfg_.rank_tau);

            // B. 计算评委分的软排名
            // 注意：judge_scores 通常是分数 (Higher is Better)，Soft-Rank 会自动处理为 "分数越高Rank越小"
            VoteDistribution judge_ranks = mcm::math::compute_soft_ranks(judge_scores, cfg_.rank_tau);

            // C. 合成生存分 (负的总排名)
            survival_score = -1.0 * (fan_ranks + judge_ranks);
        }

        // --- 3. 计算淘汰势能 (The "Hard" Constraint) ---
        // 解释：为什么这个人被淘汰了？因为他的生存分太低。
        if (elim_idx >= 0 && elim_idx < n) {
            log_lik += compute_elimination_penalty(survival_score, elim_idx);
        }

        // --- 4. 计算危险区势能 (The "Soft" Constraint) ---
        // 解释：为什么这些人进了 Bottom Two？
        log_lik += compute_jeopardy_penalty(survival_score, jeopardy_mask);

        // --- 5. 熵正则化 (Occam's Razor / Thermodynamic Prior) ---
        // 物理意义: 在满足上述约束的前提下，我们倾向于相信粉丝投票是尽可能“混乱/随机”的，
        // 而不是极端的 (比如一个人拿 99% 票)。这防止模型过拟合于某个特定的解。
        if (cfg_.entropy_regularization > 0.0) {
            Real entropy = mcm::math::compute_entropy(fan_votes_share);
            log_lik += cfg_.entropy_regularization * entropy;
        }

        return log_lik;
    }

    // =========================================================================
    // 辅助逻辑: 淘汰约束惩罚 (Quadratic Hinge Loss)
    // =========================================================================
    Real LikelihoodEvaluator::compute_elimination_penalty(
        const VoteDistribution& score,
        int elim_idx
    ) const {
        Real penalty = 0.0;
        Real loser_score = score[elim_idx];
        long n = score.size();

        // 设定安全边际，防止数值抖动导致约束失效
        Real margin = 1e-4;

        if (cfg_.enable_judge_save) {
            // [S28+ 新政: Judges' Save Mechanism]
            // 规则: 淘汰者只需处于 Bottom Two (倒数前两名)。
            // 物理意义: 全场允许有 1 个人的分数比淘汰者更低 (即 Saved 的那个倒霉蛋)。
            // 违规判定: 如果有 >= 2 个人比淘汰者还惨，说明淘汰者实际上排倒数第三或更好，这违背了物理事实。

            int worse_than_loser_count = 0;
            Real cumulative_violation = 0.0;

            for (int i = 0; i < n; ++i) {
                if (i == elim_idx) continue;

                // 如果 score[i] < loser_score，说明 i 比淘汰者更该死
                Real diff = loser_score - score[i]; // 正数代表违规程度
                if (diff > margin) {
                    worse_than_loser_count++;
                    // 累积违规幅度 (Hinge Loss)
                    cumulative_violation += diff * diff;
                }
            }

            // 只有当比淘汰者差的人数超过 1 人时，才施加惩罚
            if (worse_than_loser_count >= 2) {
                // 惩罚力度 = 基础刚度 * (违规人数超额部分) * 违规幅度
                Real count_factor = static_cast<Real>(worse_than_loser_count - 1);
                penalty -= cfg_.elim_penalty * count_factor * cumulative_violation;
            }

        } else {
            // [传统规则: Lowest Score Leaves]
            // 规则: 淘汰者必须是全场最低分 (Bottom One)。
            // 翻译: 任何存活者 (i != elim) 的分数都必须 > 淘汰者。

            for (int i = 0; i < n; ++i) {
                if (i == elim_idx) continue;

                // 违规情况: 存活者分数 (score[i]) <= 淘汰者分数 (loser_score)
                // 物理: 存活者本该死，但他活着 -> 模型推断错误 -> 罚！
                Real diff = loser_score - score[i];

                if (diff > -margin) { // 使用 -margin 允许极其微小的误差
                    // 使用平滑的二次惩罚函数 (L2 Loss)
                    // 这种平滑性对 MCMC 的 HMC/NUTS 变种极其重要
                    Real violation = diff + margin;
                    penalty -= cfg_.elim_penalty * (violation * violation);
                }
            }
        }

        return penalty;
    }

    // =========================================================================
    // 辅助逻辑: 危险区约束惩罚 (Jeopardy Potential)
    // =========================================================================
    Real LikelihoodEvaluator::compute_jeopardy_penalty(
        const VoteDistribution& score,
        ConstIntVecRef mask
    ) const {
        Real penalty = 0.0;

        // 1. 计算全场统计量作为“参考水位”
        Real mean_score = score.mean();

        for (int i = 0; i < score.size(); ++i) {
            // 如果 mask[i] == 1，说明此人在危险区 (Bottom Two/Three)
            if (mask[i] == 1) {
                // 理论上他的分应该很低，至少低于均值。
                // 违规情况: 他的分很高 (> mean)，却进了危险区。这不科学。
                Real gap = score[i] - mean_score;

                if (gap > 0) {
                    // 这是一个“软约束”，惩罚力度 (jeopardy_penalty) 通常小于淘汰惩罚
                    // 同样使用平方惩罚以保证导数连续
                    penalty -= cfg_.jeopardy_penalty * (gap * gap);
                }
            }
            // 可选扩展: 如果 mask[i] == 0 (Safe)，我们可以稍微惩罚他分数过低的情况
            // 但为了保持模型简洁性(Occam's Razor)，且避免与淘汰约束冲突，暂不加入。
        }

        return penalty;
    }

} // namespace engine
} // namespace mcm