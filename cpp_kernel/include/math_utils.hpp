/**
 * @file math_utils.hpp
 * @brief High-Performance Numerical Kernels (Interface Definition)
 * @details Declares SIMD-optimized math routines. Implementations are in math_utils.cpp.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [架构规范]:
 * 1. 只有 Functor (仿函数) 和 Template 保留在头文件中，以支持内联和泛型。
 * 2. 核心数学函数的实现移至 .cpp，减少编译依赖。
 * 3. 默认参数 (Default Arguments) 必须且只能在这里定义。
 */

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include "types.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace mcm {
namespace math {

    using namespace mcm::types;

    // =========================================================================
    // 1. 向量化仿函数 (Functors for Eigen::unaryExpr)
    // =========================================================================
    // 注意：仿函数必须保留在头文件中，因为 Eigen 的 unaryExpr 是模板方法，
    // 编译器需要在实例化时看到完整的 operator() 定义。
    // =========================================================================

    /**
     * @struct StableSigmoidOp
     * @brief 数值稳定的 Sigmoid 算子
     * 物理意义: 将分数差转化为胜率概率 P(Win) = 1 / (1 + exp(-x/tau))
     */
    struct StableSigmoidOp {
        Real tau;
        explicit StableSigmoidOp(Real t) : tau(t) {}

        // EIGEN_STRONG_INLINE 提示编译器强制内联，这在紧凑循环中至关重要
        EIGEN_STRONG_INLINE Real operator()(Real x) const {
            Real x_scaled = x / tau;
            // 分段函数防止 exp 溢出 (Numerical Stability Guard)
            if (x_scaled >= 0) {
                return 1.0 / (1.0 + std::exp(-x_scaled));
            } else {
                Real exp_x = std::exp(x_scaled);
                return exp_x / (1.0 + exp_x);
            }
        }
    };

    // =========================================================================
    // 2. 核心数学内核声明 (Function Declarations)
    // =========================================================================
    // 注意：不要在这里写函数体！只写分号。
    // 注意：默认参数 (Default Arguments) 只能写在这里，不能写在 .cpp 里。
    // =========================================================================

    /**
     * @brief Soft-Rank 算子 (矩阵广播加速版)
     * 将离散排名平滑化。分数越高，Rank 数值越小 (1.0 = Best)。
     * @param scores 评分向量
     * @param tau 温度系数 (默认值来自 types.hpp)
     */
    VoteDistribution compute_soft_ranks(ConstVecRef scores, Real tau = constants::RANK_TAU_DEFAULT);

    /**
     * @brief 降序软排名别名 (Wrapper)
     * 方便语义化调用，实际直接调用 compute_soft_ranks
     */
    inline VoteDistribution soft_rank_descending(ConstVecRef scores, Real tau = constants::RANK_TAU_DEFAULT) {
        return compute_soft_ranks(scores, tau);
    }

    /**
     * @brief Log-Sum-Exp (LSE) 数值稳定版
     * 在对数空间进行加法: log(sum(exp(v)))
     */
    Real log_sum_exp(ConstVecRef v);

    /**
     * @brief 对数狄利克雷分布 PDF (Log-Dirichlet)
     * 计算 P(v | alpha) 的对数概率密度
     */
    Real log_dirichlet_pdf(ConstVecRef v, ConstVecRef alpha);

    /**
     * @brief 香农熵 (Shannon Entropy)
     * H(p) = -sum(p * log2(p))
     */
    Real compute_entropy(ConstVecRef probs);

    /**
     * @brief Softmax 投影算子
     * 将 R^N 空间映射回单纯形 (Sum=1)
     */
    VoteDistribution softmax(ConstVecRef x);

} // namespace math
} // namespace mcm

#endif // MATH_UTILS_HPP