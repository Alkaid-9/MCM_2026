/**
 * MCM 2026 Problem C: High-Performance Numerical Utilities
 * Role: Simplex Projection, Log-Space Arithmetic, and Soft-Rank Kernels.
 * Standard: Academic Rigor & Industrial Scalability.
 */

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions> // 用于 lgamma (对数伽马函数)
#include <cmath>
#include <algorithm>
#include <vector>

namespace mcm {
namespace math {

/**
 * @brief 数值稳定的 Sigmoid 函数
 * 物理意义：将分数差异映射为胜率概率，避免 e^x 溢出。
 * 分段处理逻辑：当 x 极小时，使用 exp(x)/(1+exp(x)) 防止分母过小。
 */
inline double stable_sigmoid(double x, double tau) {
    const double x_scaled = x / tau;
    if (x_scaled >= 0) {
        return 1.0 / (1.0 + std::exp(-x_scaled));
    } else {
        const double exp_x = std::exp(x_scaled);
        return exp_x / (1.0 + exp_x);
    }
}

/**
 * @brief Soft-Rank 算子 (核心杀手锏)
 * 物理意义：将离散的阶梯排名函数光滑化。
 * 公式: Rank_i = 1 + \sum_{j!=i} Sigmoid((Score_j - Score_i) / tau)
 * 学术价值：解决了排名函数在贝叶斯推断中梯度为 0 的“断裂”问题。
 */
inline Eigen::VectorXd compute_soft_ranks(const Eigen::VectorXd& scores, double tau = 0.02) {
    const int n = static_cast<int>(scores.size());
    Eigen::VectorXd soft_ranks = Eigen::VectorXd::Ones(n);

    // O(N^2) 计算，但由于 N (选手数量) 通常小于 15，计算量完全可控
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            // 如果选手 j 分数高于 i，则 i 的排名增加 (排名数字变大，代表表现变差)
            soft_ranks[i] += stable_sigmoid(scores[j] - scores[i], tau);
        }
    }
    return soft_ranks;
}

/**
 * @brief Log-Sum-Exp (LSE) 算子
 * 物理意义：在对数空间进行安全求和。
 * 解决痛点：直接计算 exp(p1) + exp(p2) 极易导致浮点数溢出。
 */
inline double log_sum_exp(const Eigen::VectorXd& v) {
    const double max_val = v.maxCoeff();
    if (std::isinf(max_val)) return max_val;

    double sum = 0.0;
    for (int i = 0; i < v.size(); ++i) {
        sum += std::exp(v[i] - max_val);
    }
    return max_val + std::log(sum);
}

/**
 * @brief 对数狄利克雷概率密度 (Log-Dirichlet PDF)
 * 物理意义：量化“粉丝投票分布”偏离 Zipf's Law 先验的程度。
 * 公式: ln P(v|alpha) = ln Gamma(sum alpha) - sum ln Gamma(alpha) + sum (alpha-1) ln v
 */
inline double log_dirichlet_pdf(const Eigen::VectorXd& v, const Eigen::VectorXd& alpha) {
    // 强制物理边界检查：得票率必须为正
    if ((v.array() <= 0).any()) return -1e18;

    // 利用 Eigen 提供的对数伽马函数实现高性能计算
    double term1 = std::lgamma(alpha.sum());
    double term2 = alpha.array().lgamma().sum();
    double term3 = ((alpha.array() - 1.0) * (v.array() + 1e-15).log()).sum();

    return term1 - term2 + term3;
}

/**
 * @brief 信息论算子：香农熵 (Shannon Entropy)
 * 物理意义：量化反演结果的不确定性。
 * 应用：直接回答 Task 1 中关于“估计结果有多大把握”的度量。
 */
inline double compute_entropy(const Eigen::VectorXd& probs) {
    double entropy = 0.0;
    for (int i = 0; i < probs.size(); ++i) {
        if (probs[i] > 1e-12) {
            entropy -= probs[i] * (std::log(probs[i]) / std::log(2.0));
        }
    }
    return entropy;
}

/**
 * @brief Softmax 投影算子
 * 物理意义：将无约束的采样空间映射回概率单纯形 (Sum to 1)。
 */
inline Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x / exp_x.sum();
}

} // namespace math
} // namespace mcm

#endif // MATH_UTILS_HPP