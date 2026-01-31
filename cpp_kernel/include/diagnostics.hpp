/**
* @file diagnostics.hpp
 * @brief MCMC Convergence & Statistical Rigor Auditing (Interface)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [学术地位]:
 * 本模块提供的指标直接回答了 Task 1 中的“一致性指标”与“不确定性度量”。
 * 1. R-hat (Gelman-Rubin): 量化采样链的全局收敛质量。
 * 2. ESS (Effective Sample Size): 量化样本的独立性与统计功效。
 * 3. Fidelity (保真度): 核心业务指标，衡量反演结果对淘汰事实的解释力。
 */

#ifndef DIAGNOSTICS_HPP
#define DIAGNOSTICS_HPP

#include <vector>
#include <Eigen/Dense>
#include "types.hpp"

namespace mcm {
    namespace diag {

        // 引入全项目统一的物理量别名
        using namespace mcm::types;

        /**
         * @brief 计算 Gelman-Rubin Statistic (R-hat)
         * @details 比较多条链的链间方差与链内方差。R-hat < 1.1 是进入顶级期刊发表的门槛。
         * @param chains_1d 数据布局: [M条独立链][N个时间步样本]
         */
        double compute_r_hat(const std::vector<std::vector<double>>& chains_1d);

        /**
         * @brief 计算有效样本量 (Effective Sample Size)
         * @details 通过对数自相关函数积分，消除 MCMC 序列的相关性，计算“真实有效”的样本当量。
         */
        double compute_ess(const std::vector<double>& chain);

        /**
         * @brief 业务一致性指标：Rank Fidelity Score
         * @details
         * 物理意义：
         * 对应题目 Task 1: "Does your model correctly estimate fan votes that lead to
         * results consistent with who was eliminated each week?"
         *
         * 算法逻辑：
         * 在模型估计的粉丝票下，如果被淘汰者的“总生存分”确实是全场最低，得分 1.0。
         * 若存在“秩逆转”（即有人分比他低却没走），根据违规程度扣分。
         *
         * @param est_votes 估计出的后验均值向量 (Numpy 内存对齐视图)
         * @param judge_scores 评委评分信号 (Robust Z-Scores)
         * @param elim_idx 实际被淘汰选手的 0-based 索引
         * @param is_percent_rule 赛制切换标志 (True=S3-S27, False=S1-S2/S28+)
         *
         * [关键修复]: 使用 ConstVecRef 确保签名与 bindings.cpp 及 diagnostics.cpp 绝对一致
         */
        double compute_fidelity(
            ConstVecRef est_votes,
            ConstVecRef judge_scores,
            int elim_idx,
            bool is_percent_rule
        );

    } // namespace diag
} // namespace mcm

#endif // DIAGNOSTICS_HPP