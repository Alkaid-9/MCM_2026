/**
 * @file likelihood.hpp
 * @brief Core Bayesian Energy Engine Interface (BIO Engine v4.6 - Full-Rank Consistency)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [物理架构说明]:
 * 本模块定义了单纯形流形上的势能函数评估接口。
 * 采样器通过调用 compute_log_posterior 获取当前投票分布的负能量。
 *
 * 核心势能组件：
 * 1. Prior Potential: 齐夫定律引力场。
 * 2. Elimination Potential: 淘汰硬约束形成的深井。
 * 3. Winner Potential: 冠军全序一致性约束 (解决 Bobby Bones 悖论的关键)。
 * 4. Jeopardy Potential: 危险区信号的软约束。
 */

#ifndef LIKELIHOOD_HPP
#define LIKELIHOOD_HPP

#include "types.hpp"
#include <Eigen/Dense>

namespace mcm {
namespace engine {

    using namespace mcm::types;

    /**
     * @struct LikelihoodConfig
     * @brief 似然函数刚度与贝叶斯先验参数配置
     */
    struct LikelihoodConfig {
        // --- 几何形状参数 ---
        Real rank_tau = constants::RANK_TAU_DEFAULT; // Soft-Rank 温度

        // --- 似然约束刚度 (Likelihood Stiffness) ---
        Real elim_penalty = 1200.0;      // 淘汰/冠军违规惩罚 (Hard Penalty)
        Real jeopardy_penalty = 150.0;   // 危险区违规惩罚 (Soft Penalty)

        // --- 贝叶斯正则化强度 ---
        Real prior_strength = 50.0;      // 先验集中度 (Dirichlet Strength)
                                         // 物理意义：值越大，明星的“出厂名气”对结果影响力越大

        // --- 机制逻辑开关 ---
        bool enable_judge_save = false;  // S28+ 的评委救济机制逻辑开关
    };

    /**
     * @class LikelihoodEvaluator
     * @brief 势能评估算子 (Energy Operator)
     */
    class LikelihoodEvaluator {
    public:
        /**
         * @brief 显式构造函数
         */
        explicit LikelihoodEvaluator(const LikelihoodConfig& config) : cfg_(config) {}

        /**
         * @brief 计算非归一化对数后验概率密度 (Target Density)
         *
         * @param fan_votes_share 候选粉丝投票向量 (Sum=1)
         * @param judge_scores 评委评分信号
         * @param elim_idx 真实淘汰选手索引 (0-based, -1 表示无淘汰)
         * @param jeopardy_mask 危险区标记向量
         * @param prior_mean Zipf 先验均值向量 (由 Python 侧注入)
         * @param mech_type 机制类型 (RANK/PERCENT)
         * @param winner_idx 冠军选手索引 (-1 表示非决赛周或无确定冠军)
         * @return Real 对数概率密度值
         */
        Real compute_log_posterior(
            ConstVecRef fan_votes_share,
            ConstVecRef judge_scores,
            int elim_idx,
            ConstIntVecRef jeopardy_mask,
            ConstVecRef prior_mean,
            MechanismType mech_type,
            int winner_idx = -1  // <--- 核心新增参数
        ) const;

    private:
        const LikelihoodConfig cfg_;

        // =====================================================================
        // 内部势能算子 (Internal Potential Operators)
        // 这些声明必须与 src/likelihood.cpp 严格一致，否则会引发链接错误
        // =====================================================================

        /**
         * @brief 计算贝叶斯先验场对数密度 (The Zipf-Dirichlet Prior)
         */
        Real compute_prior_log_density(
            ConstVecRef fan_votes,
            ConstVecRef prior_mean
        ) const;

        /**
         * @brief 计算淘汰约束惩罚项 (Censorship Likelihood)
         */
        Real compute_elimination_penalty(
            const VoteDistribution& score,
            int elim_idx
        ) const;

        /**
         * @brief 计算冠军全序一致性惩罚项 (Full-Rank Consistency Anchor)
         * 物理意义：强制冠军总分全场最高，用于还原高人气明星的真实票数。
         */
        Real compute_winner_penalty(
            const VoteDistribution& score,
            int winner_idx
        ) const;

        /**
         * @brief 计算危险区信号惩罚项 (Jeopardy Soft Hint)
         */
        Real compute_jeopardy_penalty(
            const VoteDistribution& score,
            ConstIntVecRef mask
        ) const;
    };

} // namespace engine
} // namespace mcm

#endif // LIKELIHOOD_HPP