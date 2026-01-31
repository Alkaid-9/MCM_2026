/**
 * @file rng.hpp
 * @brief Thread-Safe, Reproducible Random Number Generation Infrastructure
 * @details Implements seeded RNG factories for OpenMP parallel regions using SplitMix64.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [核心技术点]:
 * 1. 独立性 (Independence): 每个 Chain 持有独立的 mt19937_64 实例，绝不共享。
 * 2. 种子分裂 (Seed Splitting): 使用 SplitMix64 算法将 (MasterSeed + ChainID)
 *    映射为高熵的初始状态，防止相邻种子带来的序列相关性 (Seed Correlation)。
 * 3. 零锁设计 (Lock-Free): 完全消除并发下的锁竞争。
 */

#ifndef RNG_HPP
#define RNG_HPP

#include <random>
#include <vector>
#include <cstdint>
#include <cmath>

namespace mcm {
namespace rng {

    // 使用 64 位梅森旋转算法，周期长达 2^19937-1，适合大规模蒙特卡洛
    using EngineType = std::mt19937_64;

    // =========================================================================
    // 1. 种子混淆算法 (Seed Splitting)
    // =========================================================================
    /**
     * @brief SplitMix64 伪随机数生成器 (Stateful)
     * 物理意义: 标准的 mt19937 初始化非常慢且对种子敏感。
     * SplitMix64 是一个极快且“雪崩效应”极好的哈希函数，用于初始化。
     */
    inline uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // =========================================================================
    // 2. 随机数引擎工厂 (Factory)
    // =========================================================================
    class RngFactory {
    public:
        /**
         * @brief 为特定链构建一个确定性的 RNG 引擎
         *
         * @param master_seed 全局主种子 (来自 rules.yaml)
         * @param chain_id 当前处理的链 ID (或线程 ID)
         * @param stream_offset (可选) 如果同一链需要多个独立流，可增加偏移
         */
        static EngineType create_engine(int master_seed, int chain_id, int stream_offset = 0) {
            // 1. 混合输入熵：将种子、链ID、偏移量打包进 64位 状态
            // 移位操作保证了高低位的充分利用
            uint64_t seed_state = static_cast<uint64_t>(master_seed)
                                ^ (static_cast<uint64_t>(chain_id) << 32)
                                ^ (static_cast<uint64_t>(stream_offset) << 48);

            // 2. 运行几次 SplitMix64 消除输入模式的规律性
            // 相当于对种子进行"预热" (Warm-up)
            uint64_t robust_seed = splitmix64(seed_state);
            robust_seed = splitmix64(robust_seed);

            // 3. 初始化梅森旋转引擎
            return EngineType(robust_seed);
        }
    };

    // =========================================================================
    // 3. 常用分布的辅助函数 (Distribution Helpers)
    // =========================================================================
    // 注意：现代编译器优化下，在栈上创建 distribution 对象开销极小。
    // 不要使用 static thread_local，这在 OpenMP 动态调度中是危险的。

    /**
     * @brief 生成标准正态分布 N(0, 1)
     */
    inline double randn(EngineType& engine) {
        std::normal_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    }

    /**
     * @brief 生成 [0, 1) 均匀分布
     */
    inline double randu(EngineType& engine) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    }

    /**
     * @brief 生成 Gamma 分布 (用于 Dirichlet 采样等)
     * Gamma(alpha, 1.0)
     */
    inline double rand_gamma(EngineType& engine, double alpha) {
        // alpha 必须 > 0，否则 Gamma 分布无定义
        if (alpha <= 0.0) alpha = 1e-6;
        std::gamma_distribution<double> dist(alpha, 1.0);
        return dist(engine);
    }

} // namespace rng
} // namespace mcm

#endif // RNG_HPP