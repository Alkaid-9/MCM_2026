/**
 * @file math_utils.cpp
 * @brief High-Performance Numerical Kernels Implementation (Industrial Refactor v5.2)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [学术架构说明]:
 * 本模块作为 BIO (Bayesian Inverse Optimization) 引擎的底层算子库，
 * 严格执行“数值稳定性第一”原则。通过移位不变性 (Shift-Invariance)
 * 和对数空间映射，消除了 MCMC 在探索极端概率分布时的浮点数溢出风险。
 *
 * [优化特性]:
 * 1. 矩阵化 Soft-Rank: 采用广播机制 (Broadcasting) 替代嵌套循环。
 * 2. 向量化 Lgamma: 深度集成 Eigen Unsupported 模块。
 * 3. 内存友好性: 计算过程尽量保持在 L1/L2 Cache 命中的局部空间内。
 */

#include "math_utils.hpp"
#include <unsupported/Eigen/SpecialFunctions> // [关键依赖] 提供向量化 lgamma
#include <iostream>
#include <algorithm>

namespace mcm {
namespace math {

    using namespace mcm::types;

    // =========================================================================
    // 1. Soft-Rank 核心实现 (Differentiable Ranking Engine)
    // =========================================================================
    /**
     * @brief 矩阵广播加速的平滑排名算子
     * 物理意义：将离散的排位博弈映射为连续可导的势能场。
     * 公式：Rank_i = 1 + \sum_{j!=i} Sigmoid((Score_j - Score_i) / tau)
     *
     * [论文 3.1 节关键点]: 通过 tau 参数控制排名的“硬度”，
     * 解决了离散 Rank 函数在贝叶斯推断中梯度为 0 的“收敛断头台”问题。
     */
    VoteDistribution compute_soft_ranks(ConstVecRef scores, Real tau) {
        const long n = scores.size();
        if (n == 0) return VoteDistribution(0);

        // [HPC 优化]: 避免 O(N^2) 标量循环。
        // diff_matrix(j, i) = Score_j - Score_i
        // 使用 replicate 进行广播，产生一个全对差分矩阵。
        // 这将触发编译器的循环展开和 SIMD 优化。
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> diff_matrix =
            scores.replicate(1, n) - scores.transpose().replicate(n, 1);

        // 原子级向量化应用 Sigmoid。
        // 这里的 StableSigmoidOp 定义在头文件中以支持内联。
        auto prob_matrix = diff_matrix.unaryExpr(StableSigmoidOp(tau));

        // 对列求和 (ColWise Sum) 计算每个选手的“劣后计数”。
        // 修正逻辑：
        // 1. 矩阵包含对角线 (i vs i)，Sigmoid(0)=0.5。需扣除此 0.5。
        // 2. 序数从 1.0 开始累加。
        // Correction = -0.5 (self-diag) + 1.0 (base) = +0.5
        return prob_matrix.colwise().sum().array() + 0.5;
    }

    // =========================================================================
    // 2. 贝叶斯正则化算子 (Bayesian Regularization)
    // =========================================================================
    /**
     * @brief 对数狄利克雷概率密度 (Log-Dirichlet PDF)
     * 物理意义：量化当前采样点对 Zipf's Law 粉丝先验的偏离程度。
     * [重构点]: 必须在对数域完成所有运算，否则高维连乘会导致精度瞬间下溢。
     */
    Real log_dirichlet_pdf(ConstVecRef v, ConstVecRef alpha) {
        // [防御性熔断]: 概率单纯形边界检查
        if ((v.array() <= constants::EPSILON).any()) {
            return constants::NEG_INF;
        }

        // 公式: ln P = ln Gamma(sum alpha) - sum ln Gamma(alpha) + sum (alpha - 1)ln(v)
        const Real sum_alpha = alpha.sum();

        // 1. 归一化项 (Log-Beta Function)
        Real term1 = std::lgamma(sum_alpha);

        // 2. 向量化计算每个维度的 Gamma 贡献
        // 使用 Eigen::unaryExpr 配合 lambda 以触发寄存器级并行
        Real term2 = alpha.unaryExpr([](Real x) { return std::lgamma(x); }).sum();

        // 3. 核心核函数 (Kernel Function)
        // 向量化点乘: (alpha - 1.0) * log(v)
        Real term3 = ((alpha.array() - 1.0) * v.array().log()).sum();

        return term1 - term2 + term3;
    }

    // =========================================================================
    // 3. 信息论算子 (Information Theoretic Metrics)
    // =========================================================================
    /**
     * @brief 香农熵 (Shannon Entropy) - 工业级向量化版
     * 物理意义：量化反演结果的混乱度 (Task 1 核心度量指标)。
     * [学术严谨性]: 引入 constants::EPSILON 以防止边界处 log(0) 崩溃。
     */
    Real compute_entropy(ConstVecRef probs) {
        // 1 / ln(2) 用于将自然对数 (nats) 转换为比特 (bits)
        constexpr Real LOG2_INV = 1.4426950408889634;

        // 向量化计算: - \sum p * ln(p + eps)
        // 利用表达式模板 (Expression Templates) 减少临时内存分配
        Real raw_entropy = (probs.array() * (probs.array() + constants::EPSILON).log()).sum();

        return -1.0 * raw_entropy * LOG2_INV;
    }

    // =========================================================================
    // 4. 数值稳定工具 (Numerical Stability Guards)
    // =========================================================================
    /**
     * @brief Log-Sum-Exp (LSE) 算法
     * 场景：用于 MCMC 接受率计算 P(Accept) = min(1, exp(log_lik_new - log_lik_old))。
     * 物理意义：在保持指数精度的前提下，防止浮点数溢出。
     */
    Real log_sum_exp(ConstVecRef v) {
        if (v.size() == 0) return constants::NEG_INF;

        Real max_val = v.maxCoeff();
        if (std::isinf(max_val)) return constants::NEG_INF;

        // [核心技巧]: 提公因式 exp(max_val)
        // log(sum(exp(v))) = max_val + log(sum(exp(v - max_val)))
        Real sum_exp = (v.array() - max_val).exp().sum();
        return max_val + std::log(sum_exp);
    }

    /**
     * @brief Softmax 投影算子 (Manifold Projection)
     * 物理意义：将 R^N 空间中的随机游走扰动重新投影至概率单纯形 Delta^{n-1}。
     * [重构点]: 必须执行 Shift-Invariant 变换以抵御大数值波动。
     */
    VoteDistribution softmax(ConstVecRef x) {
        // 减去最大值以确保 exp() 结果处于 (0, 1] 区间
        Real max_val = x.maxCoeff();
        VoteDistribution exp_x = (x.array() - max_val).exp();

        Real sum_val = exp_x.sum();

        // 极端崩溃防御：如果所有输入均为负无穷
        if (sum_val < constants::EPSILON) {
            return VoteDistribution::Constant(x.size(), 1.0 / static_cast<Real>(x.size()));
        }

        return exp_x / sum_val;
    }

} // namespace math
} // namespace mcm