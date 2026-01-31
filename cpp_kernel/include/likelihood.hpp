/**
 * @file likelihood.hpp
 * @brief Core Bayesian Energy Engine Interface (BIO Engine v4.2 - Bayesian Fixed)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [物理架构说明 - Physics Architecture]:
 * 本模块定义了参数空间中的“能量景观” (Energy Landscape)。
 * 根据玻尔兹曼分布原理: P(v | Outcome) \propto exp(-E(v))
 *
 * 总能量 E(v) 由三部分势能叠加而成：
 * 1. E_prior (先验势能): 由 Zipf's Law (幂律分布) 形成的引力场，将解拉向长尾分布。
 * 2. E_constraint (约束势能): 由淘汰事实形成的“无限深势能井”，通过惩罚项实现。
 * 3. E_jeopardy (危机势能): 由 "Bottom Two" 信号形成的软约束浅井。
 *
 * 这种设计将离散的组合优化问题转化为了连续流形上的哈密顿动力学问题 (Hamiltonian Dynamics)。
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
     * @brief 能量函数超参数配置 (Energy Function Hyperparameters)
     * 控制先验信念与观测数据之间的权重平衡 (Bias-Variance Tradeoff)。
     */
    struct LikelihoodConfig {
        // --- 势能形状控制 ---
        Real rank_tau = constants::RANK_TAU_DEFAULT; // Soft-Rank 温度: 越小越接近阶跃函数，梯度越陡峭

        // --- 约束强度 (Constraint Stiffness) ---
        Real elim_penalty = 1000.0;      // 强约束: 违反淘汰事实的能量惩罚 (Hard Barrier)
        Real jeopardy_penalty = 200.0;   // 中约束: 违反危险区信号的能量惩罚 (Soft Barrier)

        // --- 贝叶斯先验控制 (Bayesian Priors) ---
        // [关键重构] 替代原有的 Entropy Regularization
        Real prior_strength = 50.0;      // 先验置信度 (Dirichlet Concentration Parameter)
                                         // 物理含义: 相当于我们拥有多少个“虚拟样本”来支撑 Zipf 先验。
                                         // strength -> 0: 数据驱动 (MLE); strength -> inf: 锁定先验。

        // --- 赛制特例逻辑 ---
        bool enable_judge_save = false;  // S28+ 评委救济机制: 放宽淘汰势能井的宽度
    };

    /**
     * @class LikelihoodEvaluator
     * @brief 似然评估器 (The Energy Function)
     * 负责计算给定投票分布 v 在当前观测证据下的非归一化对数后验概率。
     */
    class LikelihoodEvaluator {
    public:
        /**
         * @brief 显式构造函数
         */
        explicit LikelihoodEvaluator(const LikelihoodConfig& config) : cfg_(config) {}

        /**
         * @brief 计算非归一化对数后验概率 (Log-Posterior Density)
         *
         * Formula:
         * log P(v | D) = log P(D | v) + log P(v | alpha) + Const
         *              = (Constraint_Penalty) + (Dirichlet_Log_PDF)
         *
         * @param fan_votes_share 提议的粉丝投票占比向量 (Sum=1)
         * @param judge_scores 评委评分信号 (已归一化或 Z-Score)
         * @param elim_idx 真实被淘汰选手的索引 (0-based)
         * @param jeopardy_mask 危险区掩码 (1=Bottom Two, 0=Safe)
         * @param prior_mean [新增] 基于 Zipf's Law 预测的先验均值向量 (Python 端生成)
         * @param mech_type 赛制机制 (RANK_BASED / PERCENT_BASED)
         * @return Real 对数概率值 (负能量)
         */
        Real compute_log_posterior(
            ConstVecRef fan_votes_share,
            ConstVecRef judge_scores,
            int elim_idx,
            ConstIntVecRef jeopardy_mask,
            ConstVecRef prior_mean,  // <--- Bayesian Anchor Point
            MechanismType mech_type
        ) const;

    private:
        const LikelihoodConfig cfg_;

        // =====================================================================
        // 内部势能组件 (Internal Potential Components)
        // =====================================================================

        /**
         * @brief 计算淘汰约束势能 (The "Hard" Likelihood)
         * 物理意义: 如果模型推断出存活者分数低于淘汰者，产生巨大的能量惩罚。
         */
        Real compute_constraint_penalty(
            const VoteDistribution& survival_score,
            int elim_idx
        ) const;

        /**
         * @brief 计算危险区势能 (The "Soft" Likelihood)
         * 物理意义: 使得处于 Bottom Two 的选手倾向于获得较低的总分。
         */
        Real compute_jeopardy_penalty(
            const VoteDistribution& survival_score,
            ConstIntVecRef mask
        ) const;

        /**
         * @brief 计算先验场势能 (The Bayesian Prior)
         * 物理意义: 计算当前分布 v 属于以 prior_mean 为中心的 Dirichlet 分布的概率密度。
         * 这充当了正则化项，防止模型在缺乏信息时产生最大熵(均匀)分布，而是保持幂律特征。
         */
        Real compute_prior_log_density(
            ConstVecRef fan_votes,
            ConstVecRef prior_mean
        ) const;
    };

} // namespace engine
} // namespace mcm

#endif // LIKELIHOOD_HPP