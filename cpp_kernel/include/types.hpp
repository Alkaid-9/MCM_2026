/**
 * @file types.hpp
 * @brief 全局类型定义与物理常量 (全项目核心基石)
 * @details 定义了高维度 MCMC 采样所需的张量别名、内存视图接口及赛制枚举。
 */

#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/Dense>
#include <limits>

namespace mcm {
namespace types {

    // =========================================================================
    // 1. 基础精度控制 (Precision Control)
    // =========================================================================
    // 在贝叶斯反演中，数值稳定性是生命线，默认使用双精度
    using Real = double;
    using Int  = int;

    // =========================================================================
    // 2. 线性代数对象 (Linear Algebra Objects)
    // =========================================================================
    // 使用 Eigen 的动态列向量
    using VoteDistribution = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using JudgeSignal      = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using JeopardyMask     = Eigen::Matrix<int,  Eigen::Dynamic, 1>;

    // =========================================================================
    // 3. 内存视图接口 (Memory View Interfaces - 核心：零拷贝)
    // =========================================================================
    // 物理意义：
    // ConstVecRef 是 Python Numpy 数组到 C++ Eigen 对象的“透明窗口”。
    // 必须定义为 const Eigen::Ref<const Vector>&，这是 pybind11 实现零拷贝的官方标准签名。
    // [注意]：这里的签名改变会导致 Mangled Name 变化，从而修复 undefined symbol 错误。
    // =========================================================================
    using ConstVecRef    = const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>>&;
    using ConstIntVecRef = const Eigen::Ref<const Eigen::Matrix<int,  Eigen::Dynamic, 1>>&;

    // =========================================================================
    // 4. 全局物理常量 (Global Constants)
    // =========================================================================
    namespace constants {
        // 防止 log(0) 爆炸的极小值
        constexpr Real EPSILON = 1e-12;

        // 逻辑不可能事件的 Log-Likelihood 惩罚 (Proxy for -Inf)
        constexpr Real NEG_INF = -1e18;

        // Soft-Rank 温度系数默认值：越小越接近真实排名
        constexpr Real RANK_TAU_DEFAULT = 0.05;

        // 目标接受率：高维采样理论最优值 (Roberts et al. 1997)
        constexpr Real ADAPTIVE_TARGET_ACCEPT = 0.234;
    }

    // =========================================================================
    // 5. 业务逻辑枚举 (Business Logic Enums)
    // =========================================================================
    /**
     * @enum MechanismType
     * @brief 赛制逻辑开关
     */
    enum class MechanismType {
        RANK_BASED,    // 排名法 (S1-S2, S28+)
        PERCENT_BASED, // 百分比法 (S3-S27)
        UNKNOWN
    };

} // namespace types
} // namespace mcm

#endif // TYPES_HPP