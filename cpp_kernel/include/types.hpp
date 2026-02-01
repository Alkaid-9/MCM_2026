/**
 * @file types.hpp
 * @brief Global Semantic Type Definitions & Physical Constants (The Bedrock)
 * @details Defines high-dimensional tensor aliases, memory view interfaces, and logic enums.
 *          Serves as the "Constitution" for the hybrid Python/C++ architecture.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 * @version 5.1.0-Edition
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

    /**
     * @brief 浮点精度选择器
     * [学术考量]: 贝叶斯反演涉及 log-sum-exp 等高动态范围运算，默认使用 double。
     * [工程考量]: 若需极致利用 AVX-512 指令集吞吐量，可切换为 float。
     */
    using Real = double;

    /**
     * @brief 整数索引类型
     * 用于处理 Rank, ID, Mask 等离散状态。
     */
    using Int = int;

    // =========================================================================
    // 2. 物理量语义别名 (Semantic Algebra Objects)
    // =========================================================================

    // 动态大小列向量 (Column Vectors)

    /**
     * @typedef VoteDistribution
     * @brief 观众投票分布向量
     * [数学约束]: 必须位于单纯形 (Simplex) 上，即 Elements > 0 且 Sum = 1.0。
     * [物理意义]: 潜变量 (Latent Variable)，代表不可观测的真实民意。
     */
    using VoteDistribution = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    /**
     * @typedef JudgeSignal
     * @brief 评委打分信号
     * [物理意义]: 可观测变量 (Observed Variable)，通常经过 Robust Z-Score 标准化。
     */
    using JudgeSignal = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    /**
     * @typedef JeopardyMask
     * @brief 危险区标记向量
     * [约束]: 0 (Safe) 或 1 (Bottom Two/Three)。用于似然函数中的软约束惩罚。
     */
    using JeopardyMask = Eigen::Matrix<Int, Eigen::Dynamic, 1>;

    // =========================================================================
    // 3. 内存视图接口 (Memory View Interfaces - The Bridge)
    // =========================================================================

    /**
     * @typedef ConstVecRef
     * @brief 只读浮点向量引用 (Zero-Copy View)
     * [核心技术]: 直接映射 Python Numpy 数组的内存地址，避免深拷贝。
     * [要求]: Python 侧必须保证内存布局为 C-Contiguous (由 src/bridge 模块保障)。
     */
    using ConstVecRef = const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>>&;

    /**
     * @typedef ConstIntVecRef
     * @brief 只读整数向量引用
     */
    using ConstIntVecRef = const Eigen::Ref<const Eigen::Matrix<Int, Eigen::Dynamic, 1>>&;

    // =========================================================================
    // 4. 全局物理常量 (Global Constants)
    // =========================================================================

    namespace constants {
        // --- 数值稳定性阈值 ---

        /**
         * @brief 极小值 epsilon
         * 防止 log(0) 导致的 NaN 扩散。
         */
        constexpr Real EPSILON = 1e-12;

        /**
         * @brief 物理不可能事件的能量势井
         * 当 MCMC 游走到单纯形边界外时，返回此负对数似然值 (Proxy for -Inf)。
         */
        constexpr Real NEG_INF = -1e18;

        // --- 算法超参数 (Hyperparameters) ---

        /**
         * @brief Soft-Rank 温度系数 (Temperature)
         * [论文 3.1 节]: 决定了 Sigmoid 函数的陡峭程度。
         * tau -> 0: 逼近 Heaviside Step Function (Hard Rank, 不可导)。
         * tau -> inf: 逼近均匀分布 (梯度消失)。
         * 0.05 是我们在 Task 2 灵敏度分析中确定的最佳平衡点。
         */
        constexpr Real RANK_TAU_DEFAULT = 0.05;

        /**
         * @brief MCMC 最优接受率 (Optimal Acceptance Rate)
         * [理论来源]: Roberts, G. O., et al. (1997). "Weak convergence...".
         * 证明了当维度 N -> Infinity 时，RWM 算法的最优接受率为 0.234。
         * 我们的自适应步长 (Dual-Averaging) 将自动收敛至此值。
         */
        constexpr Real ADAPTIVE_TARGET_ACCEPT = 0.234;
    }

    // =========================================================================
    // 5. 业务逻辑枚举 (Business Logic Enums)
    // =========================================================================

    /**
     * @enum MechanismType
     * @brief 赛制逻辑开关 (Mechanism Regime)
     * 对应论文中的 "Comparative Statics" 分析。
     */
    enum class MechanismType {
        /**
         * @brief 排名制 (S1-S2, S28+)
         * 逻辑: Survival ~ Rank(Judge) + Rank(Fan)
         * [特性]: 序数聚合 (Ordinal Aggregation)，充当低通滤波器。
         */
        RANK_BASED,

        /**
         * @brief 百分比制 (S3-S27)
         * 逻辑: Survival ~ Score(Judge) + Share(Fan)
         * [特性]: 基数聚合 (Cardinal Aggregation)，放大长尾效应。
         */
        PERCENT_BASED,

        /**
         * @brief 动态自适应权重制 (Task 4 Proposed)
         * 逻辑: Survival ~ w(t)*Rank(J) + (1-w(t))*Rank(F)
         * [特性]: 帕累托最优机制，具备激励相容性 (Incentive Compatibility)。
         */
        DAW_DYNAMIC,

        UNKNOWN
    };

} // namespace types
} // namespace mcm

#endif // TYPES_HPP