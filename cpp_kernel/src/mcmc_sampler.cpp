/**
 * @file mcmc_sampler.cpp
 * @brief Ultra-High Performance MCMC Engine (Industrial Refactor v4.2 - Bayesian Enabled)
 * @details Implements Dual-Averaging Adaptation with Synchronized Sample Accounting.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
 *
 * [核心修正记录 - Final Sync]:
 * 1. 贝叶斯闭环: 将 prior_mean 注入采样核心，确保采样过程受到 Zipf 先验场的牵引。
 * 2. 能量函数替换: 切换至 compute_log_posterior，实现 P(v|D) 的正确采样。
 * 3. 并行安全: 严格执行 OpenMP 线程局部变量保护，确保 23 核在高负载下不发生 False Sharing。
 */

#include "mcmc_sampler.hpp"
#include "math_utils.hpp"
#include "rng.hpp"
#include "diagnostics.hpp"
#include <omp.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <vector>

namespace mcm {
namespace engine {

    using namespace mcm::types;

    // =========================================================================
    // 1. 建议分布：对数空间映射游走 (Log-Normal Proposal)
    // =========================================================================
    // 物理意义: 单纯形(Simplex)是有界的，直接游走容易越界。
    // 我们将状态映射到无界的对数空间进行高斯游走，再投影回来。
    Eigen::VectorXd MCMCSampler::propose_log_space_move(
        const Eigen::VectorXd& current,
        double step_size,
        std::mt19937_64& rng
    ) const {
        // Simplex -> Log-space
        Eigen::VectorXd log_proposal = (current.array() + constants::EPSILON).log();

        std::normal_distribution<double> dist(0.0, step_size);
        for (int i = 0; i < current.size(); ++i) {
            log_proposal[i] += dist(rng);
        }

        // Log-space -> Simplex (via Shift-Invariant Softmax)
        return mcm::math::softmax(log_proposal);
    }

    // =========================================================================
    // 2. 单链采样核心：Dual-Averaging 动力学
    // =========================================================================
    MCMCSampler::ChainState MCMCSampler::run_single_chain(
        int thread_id,
        const Eigen::VectorXd& start_pos,
        const LikelihoodEvaluator& evaluator,
        const Eigen::VectorXd& judge_scores,
        int elim_idx,
        const Eigen::VectorXi& jeopardy_mask,
        const Eigen::VectorXd& prior_mean, // <--- [New] 贝叶斯锚点
        MechanismType mech_type
    ) const {
        ChainState state;
        state.current_position = start_pos;
        state.accepted_count = 0;

        // 为该线程构建独立的 RNG 引擎，确保并行结果的可复现性
        auto rng = mcm::rng::RngFactory::create_engine(cfg_.seed, thread_id);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);

        // [初始化能量] 计算初始位置的后验概率 (Log Posterior)
        // 注意：这里使用的是 compute_log_posterior 而非 likelihood
        state.current_log_lik = evaluator.compute_log_posterior(
            state.current_position, judge_scores, elim_idx, jeopardy_mask, prior_mean, mech_type
        );

        // 预计算 Thinning 后的样本容量
        // 逻辑：只有过了 Burn-in 且满足 Thinning 间隔的样本才会被记录
        int burn_in_steps = static_cast<int>(cfg_.n_samples * cfg_.burn_in_ratio);

        // --- Dual-Averaging 自适应步长初始化 (NUTS 风格) ---
        double step_size = cfg_.init_step_size;
        double log_step_size = std::log(step_size);
        double log_step_size_bar = log_step_size;
        double H_bar = 0.0;
        const double gamma = 0.05, t0 = 10.0, kappa = 0.75;
        const double mu = std::log(10.0 * step_size);

        // --- MCMC 主循环 ---
        for (int iter = 1; iter <= cfg_.n_samples; ++iter) {

            // A. Propose
            Eigen::VectorXd proposal = propose_log_space_move(state.current_position, step_size, rng);

            // B. Evaluate Energy (Log Posterior)
            double proposal_log_lik = evaluator.compute_log_posterior(
                proposal, judge_scores, elim_idx, jeopardy_mask, prior_mean, mech_type
            );

            // C. Metropolis-Hastings Acceptance
            // Log(alpha) = Log(P_new) - Log(P_old) + Log(Correction)
            // 注：由于 Proposal 是对称的对数正态游走，校正项相互抵消 (近似)
            double log_alpha = proposal_log_lik - state.current_log_lik;
            double accept_prob = std::min(1.0, std::exp(log_alpha));

            bool accepted = false;
            if (u_dist(rng) < accept_prob) {
                state.current_position = proposal;
                state.current_log_lik = proposal_log_lik;
                state.accepted_count++;
                accepted = true;
            }

            // D. Dual-Averaging Adaptation (仅在预热期调整步长)
            // 目标：将接受率收敛到理论最优值 0.234
            if (cfg_.adaptive && iter <= burn_in_steps) {
                double eta = 1.0 / (iter + t0);
                H_bar = (1.0 - eta) * H_bar + eta * (constants::ADAPTIVE_TARGET_ACCEPT - accept_prob);
                log_step_size = mu - (std::sqrt(iter) / gamma) * H_bar;
                double iter_pow = std::pow(iter, -kappa);
                log_step_size_bar = iter_pow * log_step_size + (1.0 - iter_pow) * log_step_size_bar;
                step_size = std::exp(log_step_size);
            } else if (iter == burn_in_steps + 1) {
                // 预热结束，锁定步长
                step_size = std::exp(log_step_size_bar);
            }

            // E. Synchronized Recording (Thinning)
            // 只有正式采样期才记录数据
            if (iter > burn_in_steps && (iter - burn_in_steps) % cfg_.thinning == 0) {
                state.samples.push_back(state.current_position);
            }
        }

        state.step_size = step_size; // 记录最终步长供调试
        return state;
    }

    // =========================================================================
    // 3. 并行推断调度与 Map-Reduce 聚合
    // =========================================================================
    InferenceResult MCMCSampler::run_parallel_inference(
        const Eigen::VectorXd& judge_scores,
        int elim_idx,
        const Eigen::VectorXi& jeopardy_mask,
        const Eigen::VectorXd& prior_mean,
        MechanismType mech_type
    ) {
        long n_vars = judge_scores.size();

        // 构建 Evaluator 配置
        LikelihoodConfig l_config;
        l_config.rank_tau = cfg_.rank_tau;
        l_config.elim_penalty = cfg_.elim_penalty;
        l_config.jeopardy_penalty = cfg_.jeopardy_penalty;
        l_config.prior_strength = cfg_.prior_strength; // [New] 传递先验强度

        // --- PHASE 1: Parallel Map (并行采样) ---
        std::vector<ChainState> results(cfg_.n_chains);

        // 动态调度 OpenMP，根据 CPU 负载自动分配链
        #pragma omp parallel for num_threads(cfg_.n_chains) schedule(dynamic)
        for (int i = 0; i < cfg_.n_chains; ++i) {
            // 每个线程拥有独立的 Evaluator 副本 (只读配置，无锁)
            LikelihoodEvaluator local_evaluator(l_config);

            // 初始点生成：在先验均值附近施加扰动
            auto rng = mcm::rng::RngFactory::create_engine(cfg_.seed, i, 777);
            Eigen::VectorXd start = propose_log_space_move(prior_mean, 0.2, rng);

            // 执行单链
            results[i] = run_single_chain(
                i, start, local_evaluator,
                judge_scores, elim_idx, jeopardy_mask, prior_mean, mech_type
            );
        }

        // --- PHASE 2: Synchronized Reduce (Welford's Algorithm) ---
        // 目的：在不存储海量原始数据的情况下，计算精确的均值和方差
        InferenceResult final_res;
        final_res.posterior_mean = Eigen::VectorXd::Zero(n_vars);
        Eigen::VectorXd M2 = Eigen::VectorXd::Zero(n_vars); // 二阶矩累加器
        long long total_count = 0;

        // 关键数据结构：用于 R-hat 审计的 [变量][链][样本] 立方体
        // 即使内存开销大，为了计算 R-hat 也是必须的
        std::vector<std::vector<std::vector<double>>> r_hat_cube(n_vars,
            std::vector<std::vector<double>>(cfg_.n_chains));

        double sum_acc = 0.0;

        for (int c = 0; c < cfg_.n_chains; ++c) {
            sum_acc += static_cast<double>(results[c].accepted_count) / cfg_.n_samples;

            // 遍历该链产生的所有有效样本
            for (const auto& sample : results[c].samples) {
                total_count++;

                // Welford 在线更新均值与 M2
                Eigen::VectorXd delta = sample - final_res.posterior_mean;
                final_res.posterior_mean += delta / static_cast<double>(total_count);
                Eigen::VectorXd delta2 = sample - final_res.posterior_mean;
                M2 += delta.cwiseProduct(delta2);

                // 填充 R-hat 立方体 (转置视角：按变量聚合)
                for (int d = 0; d < n_vars; ++d) {
                    r_hat_cube[d][c].push_back(sample[d]);
                }
            }

            // 如果需要回传轨迹 (Trace Plot)
            if (cfg_.return_traces) {
                final_res.traces.push_back(results[c].samples);
            }
        }

        // --- PHASE 3: Statistical Auditing (统计审计) ---

        // 计算标准差 (无偏估计)
        if (total_count > 1) {
            final_res.posterior_std = (M2 / static_cast<double>(total_count - 1)).cwiseSqrt();
        } else {
            final_res.posterior_std = Eigen::VectorXd::Zero(n_vars);
        }

        final_res.acceptance_rate = sum_acc / cfg_.n_chains;

        // 计算 R-hat (Gelman-Rubin Statistic)
        // 取所有变量中 R-hat 最大的那个作为该周的收敛指标
        double max_r = 0.0;
        for (int d = 0; d < n_vars; ++d) {
            double r = mcm::diag::compute_r_hat(r_hat_cube[d]);
            if (r > max_r) max_r = r;
        }
        final_res.r_hat = max_r;
        final_res.converged = (max_r < 1.1 && max_r > 0.0); // 顶刊标准

        // 计算业务保真度 (Fidelity)
        final_res.fidelity_score = mcm::diag::compute_fidelity(
            final_res.posterior_mean,
            judge_scores,
            elim_idx,
            (mech_type == MechanismType::PERCENT_BASED)
        );

        // 计算 ESS (Effective Sample Size) - 取第一个变量作为代表
        if (!r_hat_cube.empty() && !r_hat_cube[0].empty()) {
             final_res.ess = mcm::diag::compute_ess(r_hat_cube[0][0]) * cfg_.n_chains;
        } else {
            final_res.ess = 0.0;
        }

        return final_res;
    }

} // namespace engine
} // namespace mcm