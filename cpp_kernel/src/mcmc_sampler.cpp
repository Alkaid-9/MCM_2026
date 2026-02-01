/**
 * @file mcmc_sampler.cpp
 * @brief Ultra-High Performance MCMC Engine Implementation (v4.6 - Full Consistency)
 * @details Implements Dual-Averaging Adaptation with Bayesian Full-Rank Constraints.
 * @author MCM 2026 Problem C - "The Invisible Hand" Team
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
    // 1. 建议分布：单纯形上的对数空间映射游走
    // =========================================================================
    Eigen::VectorXd MCMCSampler::propose_log_space_move(
        const Eigen::VectorXd& current,
        double step_size,
        std::mt19937_64& rng
    ) const {
        // 物理映射: Simplex (有界) -> Log-space (无界)
        Eigen::VectorXd log_proposal = (current.array() + constants::EPSILON).log();

        // 施加各向同性的高斯扰动
        std::normal_distribution<double> dist(0.0, step_size);
        for (int i = 0; i < current.size(); ++i) {
            log_proposal[i] += dist(rng);
        }

        // 投影回单纯形: Log-space -> Simplex (确保 Sum=1)
        return mcm::math::softmax(log_proposal);
    }

    // =========================================================================
    // 2. 单链采样核心：Metropolis-Hastings 动力学
    // =========================================================================
    MCMCSampler::ChainState MCMCSampler::run_single_chain(
        int thread_id,
        const Eigen::VectorXd& start_pos,
        const LikelihoodEvaluator& evaluator,
        const Eigen::VectorXd& judge_scores,
        int elim_idx,
        const Eigen::VectorXi& jeopardy_mask,
        const Eigen::VectorXd& prior_mean,
        MechanismType mech_type,
        int winner_idx // <--- [New] 冠军索引注入点
    ) const {
        ChainState state;
        state.current_position = start_pos;
        state.accepted_count = 0;

        auto rng = mcm::rng::RngFactory::create_engine(cfg_.seed, thread_id);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);

        // 计算初始位置的对数后验 (Log-Posterior)
        state.current_log_lik = evaluator.compute_log_posterior(
            state.current_position, judge_scores, elim_idx, jeopardy_mask, prior_mean, mech_type, winner_idx
        );

        // 预热期步数计算
        int burn_in_steps = static_cast<int>(cfg_.n_samples * cfg_.burn_in_ratio);

        // --- Dual-Averaging 自适应步长控制 ---
        double step_size = cfg_.init_step_size;
        double log_step_size = std::log(step_size);
        double log_step_size_bar = log_step_size;
        double H_bar = 0.0;
        const double gamma = 0.05, t0 = 10.0, kappa = 0.75;
        const double mu = std::log(10.0 * step_size);

        // --- MCMC 主循环 ---
        for (int iter = 1; iter <= cfg_.n_samples; ++iter) {

            // A. Propose & Evaluate
            Eigen::VectorXd proposal = propose_log_space_move(state.current_position, step_size, rng);

            double proposal_log_lik = evaluator.compute_log_posterior(
                proposal, judge_scores, elim_idx, jeopardy_mask, prior_mean, mech_type, winner_idx
            );

            // B. Metropolis Acceptance Criterion
            double log_alpha = proposal_log_lik - state.current_log_lik;
            double accept_prob = std::min(1.0, std::exp(log_alpha));

            if (u_dist(rng) < accept_prob) {
                state.current_position = proposal;
                state.current_log_lik = proposal_log_lik;
                state.accepted_count++;
            }

            // C. 步长自适应 (仅在预热期运行)
            if (cfg_.adaptive && iter <= burn_in_steps) {
                double eta = 1.0 / (iter + t0);
                H_bar = (1.0 - eta) * H_bar + eta * (constants::ADAPTIVE_TARGET_ACCEPT - accept_prob);
                log_step_size = mu - (std::sqrt(iter) / gamma) * H_bar;
                double iter_pow = std::pow(iter, -kappa);
                log_step_size_bar = iter_pow * log_step_size + (1.0 - iter_pow) * log_step_size_bar;
                step_size = std::exp(log_step_size);
            } else if (iter == burn_in_steps + 1) {
                step_size = std::exp(log_step_size_bar);
            }

            // D. 记录样本 (Thinning 逻辑)
            if (iter > burn_in_steps && (iter - burn_in_steps) % cfg_.thinning == 0) {
                state.samples.push_back(state.current_position);
            }
        }

        state.step_size = step_size;
        return state;
    }

    // =========================================================================
    // 3. 并行调度与 Map-Reduce 聚合
    // =========================================================================
    InferenceResult MCMCSampler::run_parallel_inference(
        const Eigen::VectorXd& judge_scores,
        int elim_idx,
        const Eigen::VectorXi& jeopardy_mask,
        const Eigen::VectorXd& prior_mean,
        MechanismType mech_type,
        int winner_idx
    ) {
        long n_vars = judge_scores.size();

        // 封装似然配置
        LikelihoodConfig l_config;
        l_config.rank_tau = cfg_.rank_tau;
        l_config.elim_penalty = cfg_.elim_penalty;
        l_config.jeopardy_penalty = cfg_.jeopardy_penalty;
        l_config.prior_strength = cfg_.prior_strength;
        l_config.enable_judge_save = cfg_.enable_judge_save;

        // --- PHASE 1: Parallel Map (并行链采样) ---
        std::vector<ChainState> results(cfg_.n_chains);

        #pragma omp parallel for num_threads(cfg_.n_chains) schedule(dynamic)
        for (int i = 0; i < cfg_.n_chains; ++i) {
            LikelihoodEvaluator local_evaluator(l_config);

            // 初始点预热：在先验均值附近开始探测
            auto rng = mcm::rng::RngFactory::create_engine(cfg_.seed, i, 999);
            Eigen::VectorXd start = propose_log_space_move(prior_mean, 0.1, rng);

            results[i] = run_single_chain(
                i, start, local_evaluator,
                judge_scores, elim_idx, jeopardy_mask, prior_mean, mech_type, winner_idx
            );
        }

        // --- PHASE 2: Synchronized Reduce (统计规约) ---
        InferenceResult final_res;
        final_res.posterior_mean = Eigen::VectorXd::Zero(n_vars);
        Eigen::VectorXd M2 = Eigen::VectorXd::Zero(n_vars);
        long long total_count = 0;

        // 构建审计立方体 [维度][链][样本]
        std::vector<std::vector<std::vector<double>>> r_hat_cube(n_vars,
            std::vector<std::vector<double>>(cfg_.n_chains));

        double sum_acc = 0.0;

        for (int c = 0; c < cfg_.n_chains; ++c) {
            sum_acc += static_cast<double>(results[c].accepted_count) / cfg_.n_samples;

            for (const auto& sample : results[c].samples) {
                total_count++;

                // Welford Online Variance Algorithm (数值稳定性之王)
                Eigen::VectorXd delta = sample - final_res.posterior_mean;
                final_res.posterior_mean += delta / static_cast<double>(total_count);
                Eigen::VectorXd delta2 = sample - final_res.posterior_mean;
                M2 += delta.cwiseProduct(delta2);

                for (int d = 0; d < n_vars; ++d) {
                    r_hat_cube[d][c].push_back(sample[d]);
                }
            }

            if (cfg_.return_traces) {
                final_res.traces.push_back(results[c].samples);
            }
        }

        // --- PHASE 3: Diagnostics & Auditing (学术审计) ---
        if (total_count > 1) {
            final_res.posterior_std = (M2 / static_cast<double>(total_count - 1)).cwiseSqrt();
        } else {
            final_res.posterior_std = Eigen::VectorXd::Zero(n_vars);
        }

        final_res.acceptance_rate = sum_acc / cfg_.n_chains;

        // 计算 Split-R-hat 收敛因子
        double max_r = 0.0;
        for (int d = 0; d < n_vars; ++d) {
            double r = mcm::diag::compute_r_hat(r_hat_cube[d]);
            if (r > max_r) max_r = r;
        }
        final_res.r_hat = max_r;
        final_res.converged = (max_r < 1.1 && max_r > 0.0);

        // 计算 Fidelity (业务保真度)
        final_res.fidelity_score = mcm::diag::compute_fidelity(
            final_res.posterior_mean, judge_scores, elim_idx, (mech_type == MechanismType::PERCENT_BASED)
        );

        // 计算 ESS (保守估计)
        if (!r_hat_cube.empty() && !r_hat_cube[0].empty()) {
            final_res.ess = mcm::diag::compute_ess(r_hat_cube[0][0]) * cfg_.n_chains;
        } else {
            final_res.ess = 0.0;
        }

        return final_res;
    }

} // namespace engine
} // namespace mcm