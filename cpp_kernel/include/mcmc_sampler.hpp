/**
 * @file mcmc_sampler.hpp
 * @brief High-Performance Parallel MCMC Engine Interface (v4.6 - Full-Rank Consistency)
 * @details Implements Adaptive Metropolis-Hastings on the Simplex with Full-Rank Anchor.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 * @version 5.2.0-O-Prize-Edition
 */

#ifndef MCMC_SAMPLER_HPP
#define MCMC_SAMPLER_HPP

#include <vector>
#include <random>
#include <Eigen/Dense>
#include "likelihood.hpp"
#include "diagnostics.hpp"
#include "types.hpp"

namespace mcm {
namespace engine {

    using namespace mcm::types;

    /**
     * @struct SamplerConfig
     * @brief 采样器全量超参数配置 (对齐 rules.yaml)
     * [物理意义]: 定义贝叶斯推断的“时空刻度”。
     */
    struct SamplerConfig {
        // --- 采样控制 (Python 侧调度) ---
        int n_chains = 23;              // 并行链数量 (严格匹配 23 核物理核心)
        int n_samples = 100000;         // 每条链的总样本量 (生产环境建议 10w+)
        double burn_in_ratio = 0.5;     // 预热期比例 (弃置前 50% 以消除初始偏置)
        int thinning = 10;              // 稀疏化间隔 (降低自相关，节省内存)
        double init_step_size = 0.1;    // 初始跳跃步长 (对数空间波动率)
        bool adaptive = true;           // 是否开启 Nesterov-style Dual-Averaging
        int seed = 2026;                // 全局随机种子基准
        bool return_traces = false;     // 是否向 Python 回传全量采样轨迹

        // --- 能量函数刚度参数 (Stiffness) ---
        Real rank_tau = constants::RANK_TAU_DEFAULT; // Soft-Rank 平滑温度
        Real elim_penalty = 1200.0;     // 淘汰事实硬约束权重
        Real jeopardy_penalty = 150.0;  // 危险区软约束权重
        Real prior_strength = 50.0;     // 贝叶斯先验集中度 (Dirichlet Strength)
        bool enable_judge_save = false; // 是否启用 S28+ 的评委救济逻辑开关
    };

    /**
     * @struct InferenceResult
     * @brief 贝叶斯推断后验统计报告
     * [学术地位]: 这是模型对 Task 1 "Consistency & Uncertainty" 的直接输出。
     */
    struct InferenceResult {
        VoteDistribution posterior_mean; // 潜变量后验均值 (估计选票占比)
        VoteDistribution posterior_std;  // 估计不确定性 (后验标准差)
        double r_hat;                    // Split-R-hat 收敛审计指标 (<1.1 则可信)
        double ess;                      // 有效样本当量 (真实信息量统计)
        double acceptance_rate;          // MCMC 链平均接受率 (Target: 0.234)
        double fidelity_score;           // 业务逻辑还原保真度 (淘汰一致性)
        bool converged;                  // 是否通过统计审计红线

        // 采样轨迹数据：[ChainID][SampleID][Dimension]
        // 仅在 return_traces=true 时填充，用于绘制“毛毛虫图”证明遍历性
        std::vector<std::vector<Eigen::VectorXd>> traces;
    };

    /**
     * @class MCMCSampler
     * @brief 高性能并行 MCMC 推理引擎
     */
    class MCMCSampler {
    public:
        /**
         * @brief 显式构造函数
         */
        explicit MCMCSampler(const SamplerConfig& config) : cfg_(config) {}

        /**
         * @brief 并行推断总入口 (Python 调用点)
         * [并行拓扑]: 将 23 条链分发至独立 CPU 核心，最后进行全量规约。
         *
         * @param judge_scores 评委评分信号 (Normalized)
         * @param elim_idx 真实淘汰选手索引 (-1 表示无淘汰)
         * @param jeopardy_mask 危险区标记向量 (Bottom Two)
         * @param prior_mean 齐夫定律先验均值向量 (Zipf Anchor)
         * @param mech_type 赛制类型 (RANK/PERCENT)
         * @param winner_idx 冠军选手索引。若为 -1，则不执行冠军全序一致性约束。
         */
        InferenceResult run_parallel_inference(
            const Eigen::VectorXd& judge_scores,
            int elim_idx,
            const Eigen::VectorXi& jeopardy_mask,
            const Eigen::VectorXd& prior_mean,
            MechanismType mech_type,
            int winner_idx = -1
        );

        /**
         * @struct ChainState
         * @brief 单条马尔可夫链的局部运行时状态
         * [极致性能]: alignas(64) 确保每个线程的 State 独占高速缓存行，防止多核竞争性能损耗。
         */
        struct alignas(64) ChainState {
            Eigen::VectorXd current_position;
            Real current_log_lik;
            double step_size;
            long long accepted_count = 0;
            std::vector<Eigen::VectorXd> samples;
        };

    private:
        SamplerConfig cfg_;

        /**
         * @brief 单链采样核心循环 (Worker Function)
         * [算法核心]: 执行基于单纯形约束的对数空间随机游走。
         */
        ChainState run_single_chain(
            int thread_id,
            const Eigen::VectorXd& start_pos,
            const LikelihoodEvaluator& evaluator,
            const Eigen::VectorXd& judge_scores,
            int elim_idx,
            const Eigen::VectorXi& jeopardy_mask,
            const Eigen::VectorXd& prior_mean,
            MechanismType mech_type,
            int winner_idx
        ) const;

        /**
         * @brief 建议分布：在对数空间执行游走并投影回单纯形
         * 物理意义: x -> Logit(x) -> Walk -> Softmax(x)
         */
        Eigen::VectorXd propose_log_space_move(
            const Eigen::VectorXd& current,
            double step_size,
            std::mt19937_64& rng
        ) const;

        /**
         * @brief Nesterov-style 对偶平均步长自适应
         * [学术价值]: 确保 MCMC 永远在后验概率密集的区域（Typical Set）游走。
         */
        void adapt_step_size(double& step_size, double recent_acceptance_rate) const;
    };

} // namespace engine
} // namespace mcm

#endif // MCMC_SAMPLER_HPP