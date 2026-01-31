/**
 * @file types.hpp
 * @brief Global Type Definitions & Physical Constants (The Bedrock)
 * @details Defines high-dimensional tensor aliases, memory view interfaces, and logic enums.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 */

#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/Dense>
#include <limits>
#include <vector>

namespace mcm {
namespace types {

    // =========================================================================
    // 1. 基础精度控制 (Precision Control)
    // =========================================================================
    // 在贝叶斯反演中，数值稳定性是生命线，默认使用双精度浮点
    using Real = double;
    using Int = int;

    // =========================================================================
    // 2. 线性代数对象 (Linear Algebra Objects)
    // =========================================================================
    // 使用 Eigen 的动态列向量 (Column Vectors)
    // VoteDistribution: 单纯形上的点 (Sum = 1.0)
    using VoteDistribution = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    // JudgeSignal: 评委打分向量 (Normalized or Z-Score)
    using JudgeSignal = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    // JeopardyMask: 危险区标记 (0/1 Vector)
    using JeopardyMask = Eigen::Matrix<Int, Eigen::Dynamic, 1>;

    // =========================================================================
    // 3. 内存视图接口 (Memory View Interfaces - 核心：零拷贝)
    // =========================================================================
    // 物理意义：
    // ConstVecRef 是 Python Numpy 数组到 C++ Eigen 对象的“透明窗口”。
    // 必须定义为 const Eigen::Ref<const Vector>&，这是 pybind11 实现零拷贝的标准签名。
    // 警告：不要移除 const引用，否则会触发深拷贝 (Deep Copy)，导致性能崩塌。
    // =========================================================================
    using ConstVecRef = const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>>&;
    using ConstIntVecRef = const Eigen::Ref<const Eigen::Matrix<Int, Eigen::Dynamic, 1>>&;

    // =========================================================================
    // 4. 全局物理常量 (Global Constants)
    // =========================================================================
    namespace constants {
        // 防止 log(0) 爆炸的极小值 (Numerical Stability Floor)
        constexpr Real EPSILON = 1e-12;

        // 逻辑不可能事件的 Log-Likelihood 惩罚 (Proxy for -Inf)
        // 物理意义：能量势井的“无限深”底部
        constexpr Real NEG_INF = -1e18;

        // Soft-Rank 温度系数默认值
        // 物理意义：决定了 Sigmoid 函数的陡峭程度。
        // tau -> 0: Heaviside Step Function (Hard Rank)
        // tau -> inf: Uniform Distribution
        constexpr Real RANK_TAU_DEFAULT = 0.05;

        // Dual-Averaging 目标接受率
        // 理论来源: Roberts et al. (1997) 证明了 0.234 是高维 MCMC 的最优接受率
        constexpr Real ADAPTIVE_TARGET_ACCEPT = 0.234;
    }

    // =========================================================================
    // 5. 业务逻辑枚举 (Business Logic Enums)
    // =========================================================================
    /**
     * @enum MechanismType
     * @brief 赛制逻辑开关
     * 决定了似然函数中 E_constraint 的计算公式
     */
    enum class MechanismType {
        RANK_BASED,     // 排名法 (S1-S2, S28+): Survival = Rank(J) + Rank(F)
        PERCENT_BASED,  // 百分比法 (S3-S27): Survival = Score(J) + Score(F)
        UNKNOWN
    };

} // namespace types
} // namespace mcm

#endif // TYPES_HPP