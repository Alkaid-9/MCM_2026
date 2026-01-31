/**
 * @file math_utils.cpp
 * @brief High-Performance Numerical Kernels Implementation (Industrial Refactor v4.5)
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [架构说明 - Architecture]:
 * 本模块是整个 BIO (Bayesian Inverse Optimization) 引擎的数学底座。
 * 它负责将抽象的业务逻辑（如排名、不确定性）转化为计算机能高效处理的浮点运算。
 *
 * [核心优化 - Optimization]:
 * 1. Soft-Rank: 基于矩阵广播 (Broadcasting) 实现全向量化的平滑排名算子。
 * 2. Log-Space Arithmetic: 所有概率运算均在对数域完成，防止下溢。
 * 3. SIMD: 配合 Eigen 库，确保核心循环被编译为 AVX/AVX2 指令集。
 */

#include "math_utils.hpp"
#include "types.hpp"
#include <unsupported/Eigen/SpecialFunctions> // [核心依赖] 用于 std::lgamma 的向量化版本
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>

namespace mcm {
namespace math {

    using namespace mcm::types;

    // =========================================================================
    // 1. Soft-Rank 核心实现 (Differentiable Ranking)
    // =========================================================================
    // 物理意义:
    // 将离散的 Rank (1, 2, 3...) 映射为连续可导的势能函数。
    // 公式: Rank_i = 1 + Sum_{j!=i} Sigmoid( (Score_j - Score_i) / tau )
    //
    // 逻辑流:
    // - 输入 Score 越高，Rank 数值应该越小 (1.0 = 第一名)。
    // - Score_j > Score_i 时，Diff > 0，Sigmoid -> 1，Rank_i 增加 (变差)。
    // - tau (温度系数) 决定了排名的"硬度"。tau->0 逼近真实排名；tau->inf 趋向均匀。
    // =========================================================================
    VoteDistribution compute_soft_ranks(ConstVecRef scores, Real tau) {
        const long n = scores.size();
        if (n == 0) return VoteDistribution(0);

        // [核心优化] 利用 Eigen 广播机制构建差分矩阵
        // diff_matrix[j, i] = Score_j - Score_i
        // 物理含义: 衡量 j 比 i 强多少
        // replicate(r, c) 在内存中是虚拟的表达式，不会立即产生深拷贝，直到 eval
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> diff_matrix =
            scores.replicate(1, n) - scores.transpose().replicate(n, 1);

        // 应用数值稳定的 Sigmoid 算子 (Functor 定义在 math_utils.hpp)
        // Prob_ji = P(Score_j > Score_i)
        // 使用 unaryExpr 进行元素级变换，这是 SIMD 友好的
        auto prob_matrix = diff_matrix.unaryExpr(StableSigmoidOp(tau));

        // 列求和: 算出"有多少人比 i 强"
        // Rank_i = 1 + Sum_{j} Prob_ji
        VoteDistribution rank_sums = prob_matrix.colwise().sum();

        // 修正:
        // 1. 矩阵包含对角线 (i vs i)，此时 Diff=0, Sigmoid(0)=0.5。我们需要扣除这 0.5。
        // 2. 排名从 1 开始，所以基准是 +1.0。
        // Total Correction = -0.5 + 1.0 = +0.5
        return rank_sums.array() + 0.5;
    }

    // =========================================================================
    // 2. 狄利克雷分布对数概率密度 (Log-Dirichlet PDF)
    // =========================================================================
    // 物理意义:
    // 计算当前投票分布 v 是否符合先验 alpha (Zipf's Law)。
    // 公式: ln P(v|alpha) = ln B(alpha) + sum ( (alpha_i - 1) * ln v_i )
    // 其中 ln B(alpha) = ln Gamma(sum(alpha)) - sum(ln Gamma(alpha_i))
    // =========================================================================
    Real log_dirichlet_pdf(ConstVecRef v, ConstVecRef alpha) {
        // [防御] 单纯形边界检查: 票数不能 <= 0，否则 log 无定义
        // 使用 EPSILON 容差，允许极其微小的正数
        if ((v.array() <= constants::EPSILON).any()) {
            return constants::NEG_INF;
        }

        // 1. 计算归一化常数 (Log Beta Function)
        Real sum_alpha = alpha.sum();

        // term1 = ln Gamma(sum(alpha))
        Real term1 = std::lgamma(sum_alpha);

        // term2 = sum(ln Gamma(alpha_i))
        // 使用 Eigen 的 digest/lgamma 实现向量化计算
        // 注意：unsupported 模块中的 lgamma 可能需要 array() 调用
        Real term2 = alpha.array().lgamma().sum();

        // 2. 计算核心项 (Kernel)
        // term3 = sum( (alpha_i - 1) * ln(v_i) )
        Real term3 = ((alpha.array() - 1.0) * v.array().log()).sum();

        return term1 - term2 + term3;
    }

    // =========================================================================
    // 3. 香农熵 (Shannon Entropy)
    // =========================================================================
    // 物理意义:
    // 衡量系统的混乱度/不确定性。Task 1 中用于量化置信度。
    // H(p) = - sum p * log2(p)
    // =========================================================================
    Real compute_entropy(ConstVecRef probs) {
        // 预计算 1/ln(2) 用于底数转换 (ln -> log2)
        constexpr Real LOG2_INV = 1.4426950408889634;

        // [防御]
        // x * log(x) 在 x->0 时极限为 0，但计算机直接算 log(0) 会爆炸。
        // 加上 EPSILON 是为了数值稳定性。
        // 由于 probs 是归一化的，加 eps 造成的误差极小可忽略。
        Real sum_p_ln_p = (probs.array() * (probs.array() + constants::EPSILON).log()).sum();

        return -1.0 * sum_p_ln_p * LOG2_INV;
    }

    // =========================================================================
    // 4. Log-Sum-Exp (LSE) - 数值稳定版
    // =========================================================================
    // 物理意义:
    // 在对数域进行加法运算: log(sum(exp(v)))
    // 场景: 用于 MCMC 中的接受率计算，防止概率连乘导致下溢。
    // =========================================================================
    Real log_sum_exp(ConstVecRef v) {
        if (v.size() == 0) return constants::NEG_INF;

        Real max_val = v.maxCoeff();

        // 如果最大值已经是负无穷，说明全是 0 概率
        if (max_val <= constants::NEG_INF) return constants::NEG_INF;

        // 核心技巧: 提公因式 exp(max)，保证指数部分 <= 1 (即 e^0)
        // log( sum exp(vi) ) = log( exp(max) * sum exp(vi - max) )
        //                    = max + log( sum exp(vi - max) )
        Real sum_exp = (v.array() - max_val).exp().sum();
        return max_val + std::log(sum_exp);
    }

    // =========================================================================
    // 5. Softmax 投影算子
    // =========================================================================
    // 物理意义:
    // 将 R^N 空间的无约束向量映射回单纯形 (Sum=1)。
    // 场景: MCMC 建议分布 (Proposal Distribution) 中从 Log-Normal 投影回 Simplex。
    // =========================================================================
    VoteDistribution softmax(ConstVecRef x) {
        // 同样使用 shift-invariant trick 防止 exp 溢出
        Real max_val = x.maxCoeff();

        // 分子: exp(x_i - max)
        VoteDistribution exp_x = (x.array() - max_val).exp();
        Real sum = exp_x.sum();

        // [防御] 极罕见情况防御 (如所有 x 都是 -inf)
        if (sum < constants::EPSILON) {
            // 退化为均匀分布
            long n = x.size();
            return VoteDistribution::Constant(n, 1.0 / static_cast<Real>(n));
        }

        return exp_x / sum;
    }

} // namespace math
} // namespace mcm