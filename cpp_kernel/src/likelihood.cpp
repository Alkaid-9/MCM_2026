/**
 * @file likelihood.cpp
 * @brief Core Bayesian Energy Engine Implementation (Industrial Refactor v4.5)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [物理意义 - Physics Interpretation]:
 * 本模块负责构建高维参数空间中的“能量景观” (Energy Landscape)。
 * 公式: ln P(V | Data) = ln P(Data | V) + ln P(V | Prior) + C
 * 我们寻找满足淘汰硬约束 (Likelihood) 且最符合社会学先验 (Zipf Prior) 的后验分布。
 */

#include "likelihood.hpp"
#include "math_utils.hpp"
#include "types.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>

namespace mcm {
namespace engine {

    using namespace mcm::types;

    // =========================================================================
    // 核心接口: 计算总对数后验概率密度 (Target of MCMC)
    // =========================================================================
    Real LikelihoodEvaluator::compute_log_posterior(
        ConstVecRef fan_votes_share,
        ConstVecRef judge_scores,
        int elim_idx,
        ConstIntVecRef jeopardy_mask,
        ConstVecRef prior_mean,
        MechanismType mech_type
    ) const {
        // --- 1. 数值防御 (Stability Guard) ---
        // 贝叶斯反演对无效数值零容忍，任何状态坍缩立即返回物理禁区
        if (!fan_votes_share.allFinite()) return constants::NEG_INF;

        Real total_log_prob = 0.0;
        const long n = fan_votes_share.size();

        // --- 2. 计算贝叶斯先验势能 (The Zipfian Anchor) ---
        // 物理意义：将解拉向基于明星知名度预测的长尾分布，防止解向最大熵（均匀分布）退化。
        total_log_prob += compute_prior_log_density(fan_votes_share, prior_mean);

        // 如果当前点在先验场中概率极低，提前熔断以节省算力
        if (total_log_prob <= constants::NEG_INF) return constants::NEG_INF;

        // --- 3. 构建生存分流形 (Survival Score Manifold) ---
        // 目标：将不同赛制的规则统一映射为连续可导的标量场 "Higher is Better"
        VoteDistribution survival_score(n);

        if (mech_type == MechanismType::PERCENT_BASED) {
            // [百分比制 S3-S27]: 线性强度叠加
            // 评委信号与粉丝份额直接相加，此时 judge_scores 需预处理为同量纲
            survival_score = judge_scores + fan_votes_share;
        }
        else {
            // [排名制 S1-S2, S28+]: 序数信号叠加
            // 原始逻辑: 总排名 = Rank(Judge) + Rank(Fan)，1 为最强。
            // 物理映射: 取负号，将排名最小化转化为生存分最大化，并使用 Soft-Rank 保证可导性。

            // A. 计算粉丝票的平滑排名 (tau 控制规则硬度)
            VoteDistribution fan_ranks = mcm::math::compute_soft_ranks(fan_votes_share, cfg_.rank_tau);

            // B. 计算评委分的平滑排名
            VoteDistribution judge_ranks = mcm::math::compute_soft_ranks(judge_scores, cfg_.rank_tau);

            // C. 叠加并转化极性
            survival_score = -1.0 * (fan_ranks + judge_ranks);
        }

        // --- 4. 计算观测数据似然 (The "Hard" Constraint Likelihood) ---
        // 解释：如果推演出的票数违反了“淘汰者是表现最差”的事实，产生巨大的能量惩罚。
        if (elim_idx >= 0 && elim_idx < n) {
            total_log_prob += compute_constraint_penalty(survival_score, elim_idx);
        }

        // --- 5. 计算信号引导势能 (The "Soft" Jeopardy Likelihood) ---
        // 解释：处于 "Bottom Two" 信号中的选手，其生存分理论上应处于低位。
        total_log_prob += compute_jeopardy_penalty(survival_score, jeopardy_mask);

        return total_log_prob;
    }

    // =========================================================================
    // 组件 A: Dirichlet 先验密度评估 (The Prior)
    // =========================================================================
    Real LikelihoodEvaluator::compute_prior_log_density(
        ConstVecRef fan_votes,
        ConstVecRef prior_mean
    ) const {
        // [贝叶斯超参数映射]
        // 将 Zipf 预测值映射为 Dirichlet 分布的浓度参数 alpha。
        // 公式: alpha_i = 1.0 + prior_strength * mean_i
        // 物理直觉：
        // 1. strength 代表我们对明星固有流量的信念强度。
        // 2. "+1.0" 保证了分布的凸性 (alpha >= 1) 和计算稳定性。

        VoteDistribution alpha = (prior_mean * cfg_.prior_strength).array() + 1.0;

        // 调用 math_utils 中的向量化对数伽马算法
        return mcm::math::log_dirichlet_pdf(fan_votes, alpha);
    }

    // =========================================================================
    // 组件 B: 淘汰逻辑惩罚项 (The Likelihood)
    // =========================================================================
    Real LikelihoodEvaluator::compute_constraint_penalty(
        const VoteDistribution& score,
        int elim_idx
    ) const {
        Real penalty = 0.0;
        const Real loser_score = score[elim_idx];
        const long n = score.size();

        // 设置极小安全边际 (Safety Margin)，防止 MCMC 在边界处因导数消失而卡死
        const Real margin = 1e-4;

        if (cfg_.enable_judge_save) {
            // [S28+ 新政: Judges' Save 机制]
            // 规则：只要淘汰者处于 Bottom Two 即可。
            // 物理意义：全场只允许最多有一个选手的生存分比淘汰者更低（那个人被 Saved）。

            std::vector<Real> violations;
            for (int i = 0; i < n; ++i) {
                if (i == elim_idx) continue;

                // 计算差值：如果存活者分数 < 淘汰者分数，视为潜在违规点
                Real diff = loser_score - score[i];
                if (diff > margin) {
                    violations.push_back(diff);
                }
            }

            // [Top-K 惩罚逻辑]: 如果发现 >= 2 个人比淘汰者还差，说明 elim_idx 不是倒数前二。
            if (violations.size() >= 2) {
                // 对所有违规进行排序，选取最轻微的违规作为优化方向（Relaxed Barrier）
                std::sort(violations.begin(), violations.end());
                Real threshold_violation = violations[0];
                // 使用二次 Hinge Loss，保证势能景观平滑
                penalty -= cfg_.elim_penalty * (threshold_violation * threshold_violation);
            }
        }
        else {
            // [传统规则: Lowest Score Leaves]
            // 规则：淘汰者必须是全场表现绝对最差者 (Bottom One)。
            for (int i = 0; i < n; ++i) {
                if (i == elim_idx) continue;

                // 如果存活者表现差于淘汰者，直接施加惩罚
                Real diff = loser_score - score[i];
                if (diff > -margin) {
                    Real violation = diff + margin;
                    penalty -= cfg_.elim_penalty * (violation * violation);
                }
            }
        }
        return penalty;
    }

    // =========================================================================
    // 组件 C: 危机信号评估 (The Contextual Hint)
    // =========================================================================
    Real LikelihoodEvaluator::compute_jeopardy_penalty(
        const VoteDistribution& score,
        ConstIntVecRef mask
    ) const {
        Real penalty = 0.0;
        // 以周均分为“统计平衡点”
        const Real baseline = score.mean();

        for (int i = 0; i < score.size(); ++i) {
            // mask[i] == 1 表示当周宣布时，该选手身处危险区
            if (mask[i] == 1) {
                // 物理直觉：身处危险区的人，推演分应低于均值。
                // 违规：身处危险区却拿到了高分，说明投票估计偏离了当周气氛。
                Real violation = score[i] - baseline;
                if (violation > 0) {
                    // 施加较轻的软约束惩罚
                    penalty -= cfg_.jeopardy_penalty * (violation * violation);
                }
            }
        }
        return penalty;
    }

} // namespace engine
} // namespace mcm