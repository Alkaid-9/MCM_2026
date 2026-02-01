/**
 * @file math_utils.hpp
 * @brief High-Performance Numerical Kernels (Interface Definition)
 * @details Declares SIMD-optimized math routines. Implementations are in math_utils.cpp.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 * @version 5.2.0-O-Prize-Edition
 *
 * [架构规范]:
 * 1. 只有 Functor (仿函数) 和 Template 保留在头文件中，以支持 Eigen 指令级内联。
 * 2. 核心数学函数的实现移至 .cpp，减少编译依赖，提升大规模并行时的链接速度。
 * 3. 严格遵循语义化类型系统 (mcm::types::Real)。
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
     * [论文 3.1 节]: 用于平滑离散排名函数。
     */
    struct StableSigmoidOp {
        Real tau;
        explicit StableSigmoidOp(Real t) : tau(t) {}

        // EIGEN_STRONG_INLINE 提示编译器强制内联，这在 23 核满载循环中至关重要
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

    /**
     * @brief Soft-Rank 算子 (矩阵广播加速版)
     * 物理意义: 将离散排名平滑化。分数越高，Rank 数值越小 (1.0 = Best)。
     * [重构清单]: 移除 O(N^2) 标量循环，采用矩阵外积向量化实现。
     * @param scores 评分向量
     * @param tau 温度系数 (默认值来自 types.hpp)
     */
    VoteDistribution compute_soft_ranks(ConstVecRef scores, Real tau = constants::RANK_TAU_DEFAULT);

    /**
     * @brief 降序软排名别名 (Wrapper)
     */
    EIGEN_STRONG_INLINE VoteDistribution soft_rank_descending(ConstVecRef scores, Real tau = constants::RANK_TAU_DEFAULT) {
        return compute_soft_ranks(scores, tau);
    }

    /**
     * @brief Log-Sum-Exp (LSE) 数值稳定版
     * 物理意义: 在对数空间进行安全加法。log(sum(exp(v)))
     * 用于计算 MCMC 接受率，防止概率连乘导致的浮点下溢。
     */
    Real log_sum_exp(ConstVecRef v);

    /**
     * @brief 对数狄利克雷分布 PDF (Log-Dirichlet)
     * 物理意义: 计算当前采样点 V 匹配先验 Alpha 的概率密度。
     * [论文 2.2 节]: 用于锚定 Zipf's Law 先验场。
     */
    Real log_dirichlet_pdf(ConstVecRef v, ConstVecRef alpha);

    /**
     * @brief 香农熵 (Shannon Entropy)
     * 物理意义: 量化推断结果的不确定性。H(p) = -sum(p * log2(p))
     * [Task 1]: 直接用于回答“你对估计结果有多大把握”。
     */
    Real compute_entropy(ConstVecRef probs);

    /**
     * @brief Softmax 投影算子
     * 物理意义: 将无约束的 R^N 空间映射回概率单纯形 (Sum=1)。
     * [重构点]: 必须包含 shift-invariant 技巧以防止数值爆炸。
     */
    VoteDistribution softmax(ConstVecRef x);

    /**
     * @brief Manifold Mapping: Logit to Simplex
     * [论文 2.3 节]: 将无约束的参数空间映射到投票单纯形。
     * 物理意义: 确保 MCMC 游走永远不会跑出“概率必须为正”的物理边界。
     */
    inline VoteDistribution logit_to_simplex(ConstVecRef x) {
        return softmax(x); // 在对数空间游走即等价于 Logit-Transform
    }

} // namespace math
} // namespace mcm

#endif // MATH_UTILS_HPP