/**
 * @file likelihood.hpp
 * @brief Core Likelihood Engine Interface (BIO Engine v3.1)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [物理意义]:
 * 定义概率能量景观的几何特征。Evaluator 作为一个算子，输入候选投票分布，
 * 输出该分布在给定淘汰事实下的负能量（对数似然）。
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
     * @brief 似然函数刚度/超参数配置
     * [重构点]: 字段名已对齐 SamplerConfig，支持最大熵正则化。
     */
    struct LikelihoodConfig {
        Real rank_tau = constants::RANK_TAU_DEFAULT; // Soft-Rank 温度 (默认 0.05)
        Real elim_penalty = 1000.0;     // 强约束: 违反淘汰事实的代价
        Real jeopardy_penalty = 200.0;  // 中约束: 违反危险区(Bottom 2/3)事实的代价
        Real entropy_regularization = 0.05; // 熵正则权重: 倾向于平滑分布 (奥卡姆剃刀)
        bool enable_judge_save = false; // 是否启用 S28+ 的“评委救人”宽容逻辑
    };

    /**
     * @class LikelihoodEvaluator
     * @brief 似然函数评估器
     * 负责将业务逻辑（DWTS规则）映射到数学上的连续流形。
     */
    class LikelihoodEvaluator {
    public:
        /**
         * @brief 显式构造函数
         */
        explicit LikelihoodEvaluator(const LikelihoodConfig& config) : cfg_(config) {}

        /**
         * @brief 计算总对数似然 (ln P)
         * [实现于 likelihood.cpp]
         *
         * @param fan_votes_share 建议的粉丝投票占比向量 (Sum=1)
         * @param judge_scores 评委打分信号 (Robust Z-Scores)
         * @param elim_idx 真实淘汰者索引
         * @param jeopardy_mask 危险区掩码 (1=Bottom Two, 0=Safe)
         * @param mech_type 机制枚举 (RANK_BASED / PERCENT_BASED)
         */
        Real compute_log_likelihood(
            ConstVecRef fan_votes_share,
            ConstVecRef judge_scores,
            int elim_idx,
            ConstIntVecRef jeopardy_mask,
            MechanismType mech_type
        ) const;

    private:
        const LikelihoodConfig cfg_;

        /**
         * @brief 内部辅助：计算淘汰约束势能
         */
        Real compute_elimination_penalty(
            const VoteDistribution& survival_scores,
            int elim_idx
        ) const;

        /**
         * @brief 内部辅助：计算危险区(Jeopardy)势能
         */
        Real compute_jeopardy_penalty(
            const VoteDistribution& survival_scores,
            ConstIntVecRef mask
        ) const;
    };

} // namespace engine
} // namespace mcm

#endif // LIKELIHOOD_HPP