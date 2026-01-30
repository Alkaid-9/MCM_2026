/**
 * MCM 2026 Problem C: Heterogeneous Likelihood Engine
 * Role: Defining Log-Probability for RANK and PERCENT mechanisms.
 * Standard: Bayesian Inverse Optimization / Penalty-based Manifold.
 */

#include "mcmc_sampler.hpp"
#include "math_utils.hpp"
#include <Eigen/Dense>
#include <string>
#include <cmath>

namespace mcm {
namespace core {

/**
 * 【核心数学逻辑】计算对数似然
 * 物理意义：量化“假设的投票 v”与“真实的淘汰 elim_idx”之间的匹配度。
 *
 * 公式：ln L = - \lambda * \sum ReLU(Margin_violation)
 */
double MCMCSampler::compute_log_likelihood(
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& j,
    int elim_idx,
    const std::string& mech)
{
    double log_lik = 0.0;
    int n = v.size();

    // 强制正则项：确保 v 始终在概率单纯形内 (由于 proposal 已经处理，此处做 double check)
    if (std::abs(v.sum() - 1.0) > 1e-6) return -1e18;

    if (mech == "PERCENT") {
        /**
         * PERCENT 机制似然：
         * 逻辑：Total_Score = Judge_Pct + Fan_Pct
         * 约束：Score[elim_idx] 必须是全场最小。
         */
        Eigen::VectorXd total_scores = j + v;
        double loser_score = total_scores[elim_idx];

        for (int i = 0; i < n; ++i) {
            if (i == elim_idx) continue;
            // 物理直觉：如果存活者 i 的分数低于淘汰者 (violation)
            double margin = loser_score - total_scores[i];
            if (margin > 0) {
                // 施加平方惩罚，使得远离可行域的采样点被快速剔除
                log_lik -= 500.0 * std::pow(margin, 2);
            }
        }
    }
    else if (mech == "RANK") {
        /**
         * RANK 机制似然（高阶处理）：
         * 逻辑：Total_Rank = Rank(Judge) + Rank(Fan)
         * 约束：Total_Rank[elim_idx] 必须是全场最大 (排名越靠后，数字越大)。
         * 技术点：使用 math_utils 里的 soft_rank_operator 使似然函数平滑。
         */
        double tau = 0.02; // 极锐利的温度参数
        Eigen::VectorXd fan_ranks = mcm::math::compute_soft_ranks(v, tau);
        Eigen::VectorXd total_ranks = j + fan_ranks;

        double loser_rank_sum = total_ranks[elim_idx];

        for (int i = 0; i < n; ++i) {
            if (i == elim_idx) continue;
            // 物理直觉：如果存活者 i 的排名和大于淘汰者 (排名更靠后)
            double margin = total_ranks[i] - loser_rank_sum;
            if (margin > 0) {
                log_lik -= 100.0 * std::pow(margin, 2);
            }
        }
    }

    /**
     * 【不确定性补丁】最大熵正则化
     * 对应论文 Task 1：在满足约束的前提下，倾向于选择更“平滑”的分布。
     */
    log_lik += 0.05 * mcm::math::compute_entropy(v);

    return log_lik;
}

} // namespace core
} // namespace mcm