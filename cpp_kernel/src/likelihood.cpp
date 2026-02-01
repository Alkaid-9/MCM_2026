/**
 * @file likelihood.cpp
 * @brief Core Bayesian Energy Engine Implementation (Industrial Refactor v4.6 - Full-Rank Consistency)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [物理意义 - Physics Interpretation]:
 * 本模块负责在高维单纯形空间中构建“能量景观” (Energy Landscape)。
 * 我们通过似然函数将“比赛规则”转化为势能井，将“名气先验”转化为引力锚点。
 *
 * 公式: ln P(V | Data) \propto ln P(Data | V) + ln P(V | Prior)
 * 最终解是规则约束、评委信号与粉丝偏好的三方博弈均衡点。
 */

#include "likelihood.hpp"
#include "math_utils.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace mcm {
namespace engine {

    using namespace mcm::types;

    // 定义静态常量以消除局部变量警告，并提升数值稳定性
    static constexpr Real MARGIN = 1e-4;

    // =========================================================================
    // 核心接口: 计算总对数后验概率密度 (Target of MCMC)
    // =========================================================================
    Real LikelihoodEvaluator::compute_log_posterior(
        ConstVecRef fan_votes_share,
        ConstVecRef judge_scores,
        int elim_idx,
        ConstIntVecRef jeopardy_mask,
        ConstVecRef prior_mean,
        MechanismType mech_type,
        int winner_idx // <--- 关键参数：冠军锚点
    ) const {
        // --- 1. 数值防御 (Stability Guard) ---
        if (!fan_votes_share.allFinite()) return constants::NEG_INF;

        Real total_log_prob = 0.0;
        const long n = fan_votes_share.size();

        // --- 2. 计算贝叶斯先验势能 (Zipfian Anchor) ---
        // 物理意义：将估计值限制在社会学合理的分布形态内
        total_log_prob += compute_prior_log_density(fan_votes_share, prior_mean);

        // 如果在先验场中已经属于“物理不可能”，立即熔断
        if (total_log_prob <= constants::NEG_INF) return constants::NEG_INF;

        // --- 3. 构建统一生存分流形 (Survival Score Construction) ---
        // 目标：将不同赛制的逻辑映射为连续可导的 "Higher is Better" 标量场
        VoteDistribution survival_score(n);

        if (mech_type == MechanismType::PERCENT_BASED) {
            // [百分比制 S3-S27]: 线性强度叠加
            // Survival = Judge_Signal + Fan_Vote_Share
            survival_score = judge_scores + fan_votes_share;
        }
        else {
            // [排名制 S1-S2, S28+]: 序数信号叠加
            // 物理映射: Rank 1 为最优 -> 取负号使生存分最大化。
            // 使用 Soft-Rank 算子将离散排名连续化，保证 MCMC 梯度流动。
            VoteDistribution fan_ranks = mcm::math::compute_soft_ranks(fan_votes_share, cfg_.rank_tau);
            VoteDistribution judge_ranks = mcm::math::compute_soft_ranks(judge_scores, cfg_.rank_tau);
            survival_score = -1.0 * (fan_ranks + judge_ranks);
        }

        // --- 4. 计算淘汰似然惩罚 (Censorship Likelihood) ---
        // 物理意义：规则要求被淘汰者的总生存分必须处于最低位
        if (elim_idx >= 0 && elim_idx < n) {
            total_log_prob += compute_elimination_penalty(survival_score, elim_idx);
        }

        // --- 5. 计算冠军一致性惩罚 (Full-Rank Consistency Anchor) ---
        // 物理意义：强制冠军总分全场最高。这是解决 Bobby Bones 等低分夺冠案例的关键！
        // 它会强迫 MCMC 分配足以“逆天改命”的粉丝票数。
        if (winner_idx >= 0 && winner_idx < n) {
            total_log_prob += compute_winner_penalty(survival_score, winner_idx);
        }

        // --- 6. 计算信号指导势能 (Jeopardy Soft Hint) ---
        total_log_prob += compute_jeopardy_penalty(survival_score, jeopardy_mask);

        return total_log_prob;
    }

    // =========================================================================
    // 组件 A: Dirichlet 先验场评估 (The Bayesian Prior)
    // =========================================================================
    Real LikelihoodEvaluator::compute_prior_log_density(
        ConstVecRef fan_votes,
        ConstVecRef prior_mean
    ) const {
        // alpha = 1.0 + strength * mean
        // 物理直觉：强度越高，模型越相信明星背景；强度越低，越尊重当周比赛证据。
        VoteDistribution alpha = (prior_mean * cfg_.prior_strength).array() + 1.0;
        return mcm::math::log_dirichlet_pdf(fan_votes, alpha);
    }

    // =========================================================================
    // 组件 B: 淘汰事实惩罚项 (Hard Constraint Penalty)
    // =========================================================================
    Real LikelihoodEvaluator::compute_elimination_penalty(
        const VoteDistribution& score,
        int elim_idx
    ) const {
        Real penalty = 0.0;
        const Real loser_score = score[elim_idx];
        const long n = score.size();

        if (cfg_.enable_judge_save) {
            // [S28+ 新政: Judges' Save 机制]
            // 规则：允许有 1 个人分比淘汰者低（即被评委 Save 的那个人）。
            int worse_than_loser_count = 0;
            std::vector<Real> violations;

            for (int i = 0; i < n; ++i) {
                if (i == elim_idx) continue;
                Real diff = loser_score - score[i]; // diff > 0 表示 i 更差
                if (diff > MARGIN) {
                    worse_than_loser_count++;
                    violations.push_back(diff);
                }
            }
            // 如果比淘汰者还差的人数 >= 2，说明 elim_idx 不在 Bottom Two。
            if (worse_than_loser_count >= 2) {
                std::sort(violations.begin(), violations.end());
                // 使用最轻微的违规项进行二次惩罚 (Quadratic Hinge Loss)
                penalty -= cfg_.elim_penalty * (violations[0] * violations[0]);
            }
        }
        else {
            // [传统规则: Lowest Score Leaves]
            // 规则：淘汰者必须是全场唯一最低分。
            for (int i = 0; i < n; ++i) {
                if (i == elim_idx) continue;
                Real diff = loser_score - score[i];
                if (diff > -MARGIN) {
                    Real violation = diff + MARGIN;
                    penalty -= cfg_.elim_penalty * (violation * violation);
                }
            }
        }
        return penalty;
    }

    // =========================================================================
    // 组件 C: 冠军优胜惩罚项 (Winner Consistency Anchor)
    // =========================================================================
    Real LikelihoodEvaluator::compute_winner_penalty(
        const VoteDistribution& score,
        int winner_idx
    ) const {
        Real penalty = 0.0;
        const Real winner_val = score[winner_idx];
        const long n = score.size();

        for (int i = 0; i < n; ++i) {
            if (i == winner_idx) continue;
            // 违规判定：如果存活选手的总分高于冠军
            Real diff = score[i] - winner_val;
            if (diff > -MARGIN) {
                Real violation = diff + MARGIN;
                // 使用高刚度惩罚，强制锁定冠军的上位者身份
                penalty -= cfg_.elim_penalty * (violation * violation);
            }
        }
        return penalty;
    }

    // =========================================================================
    // 组件 D: 危险区背景势能 (Jeopardy Soft Constraint)
    // =========================================================================
    Real LikelihoodEvaluator::compute_jeopardy_penalty(
        const VoteDistribution& score,
        ConstIntVecRef mask
    ) const {
        Real penalty = 0.0;
        const Real mean_score = score.mean();

        for (int i = 0; i < score.size(); ++i) {
            // mask[i] == 1 表示当周身处 "In Jeopardy" 区域
            if (mask[i] == 1) {
                // 物理直觉：处于危险区的人其分数应低于平均。
                Real violation = score[i] - mean_score;
                if (violation > 0) {
                    penalty -= cfg_.jeopardy_penalty * (violation * violation);
                }
            }
        }
        return penalty;
    }

} // namespace engine
} // namespace mcm