/**
 * MCM 2026 Problem C: High-Performance Numerical Utilities
 * Role: Simplex Projection, Log-Space Arithmetic, and Soft-Rank Kernels.
 * Standard: Quant Finance Numerical Stability / IEEE 754 Robustness.
 */

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

namespace mcm {
namespace math {

/**
 * 【数值稳定算子】Log-Sum-Exp (LSE)
 * 物理意义：在对数空间安全地计算 exp(x) + exp(y)，防止浮点数溢出。
 * 贝叶斯归一化常数计算的标准算法。
 */
inline double log_sum_exp(const Eigen::VectorXd& v) {
    double max_val = v.maxCoeff();
    double sum = 0.0;
    for (int i = 0; i < v.size(); ++i) {
        sum += std::exp(v[i] - max_val);
    }
    return max_val + std::log(sum);
}

/**
 * 【物理约束算子】Softmax Projection
 * 物理意义：将 R^n 空间的无约束向量映射到概率单纯形 (Sum to 1)。
 * 用于将 MCMC 的随机扰动转化为合法的投票权重分布。
 */
inline Eigen::VectorXd softmax(const Eigen::VectorXd& v) {
    Eigen::VectorXd res = (v.array() - v.maxCoeff()).exp();
    return res / res.sum();
}

/**
 * 【核心数学算子】Stable Sigmoid
 * 用于 Soft-Rank 近似。针对正负极大值进行了分段处理，彻底杜绝 NaN。
 */
inline double stable_sigmoid(double x, double tau) {
    double x_scaled = x / tau;
    if (x_scaled >= 0) {
        double z = std::exp(-x_scaled);
        return 1.0 / (1.0 + z);
    } else {
        double z = std::exp(x_scaled);
        return z / (1.0 + z);
    }
}

/**
 * 【信息论算子】Shannon Entropy
 * 直接回答 Task 1：量化估计的“不确定性”。
 */
inline double compute_entropy(const Eigen::VectorXd& probs) {
    double entropy = 0.0;
    for (int i = 0; i < probs.size(); ++i) {
        if (probs[i] > 1e-12) {
            entropy -= probs[i] * std::log2(probs[i]);
        }
    }
    return entropy;
}

/**
 * 【收敛审计算子】Online Variance
 * Welford 算法：单次遍历计算方差，用于 C++ 侧实时监控 MCMC 链的稳定性。
 */
struct RunningStat {
    long long n = 0;
    double old_m = 0, new_m = 0, old_s = 0, new_s = 0;

    void push(double x) {
        n++;
        if (n == 1) {
            old_m = new_m = x;
            old_s = 0.0;
        } else {
            new_m = old_m + (x - old_m) / n;
            new_s = old_s + (x - old_m) * (x - new_m);
            old_m = new_m;
            old_s = new_s;
        }
    }

    double mean() const { return (n > 0) ? new_m : 0.0; }
    double variance() const { return (n > 1) ? new_s / (n - 1) : 0.0; }
};

} // namespace math
} // namespace mcm

#endif // MATH_UTILS_HPP