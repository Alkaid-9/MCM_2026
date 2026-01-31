# ==============================================================================
# src/bridge/mcmc_wrapper.py
# Role: Industrial High-Performance MCMC Python Bridge (v4.1)
# Function: Orchestrating zero-copy data marshalling from Pandas to C++ Kernels.
# Standard: Parallel Efficiency & Binary Compatibility.
# ==============================================================================

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# --- 1. 动态加载 C++ 内核 (ABI Safe Loading) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
sys.path.append(str(BIN_DIR))

try:
    import mcm_core_lib
except ImportError as e:
    logging.critical(f"🛑 核心加载失败：未在 {BIN_DIR} 找到编译好的 mcm_core_lib。")
    logging.critical("请确保已执行：cd cpp_kernel && mkdir -p build && cd build && cmake .. && make")
    raise e

from src.etl.config_loader import ConfigLoader
from src.core.priors import VotePriors

class MCMCInferenceWrapper:
    """
    贝叶斯推理指挥官：
    负责将数据科学界的 DataFrames 转化为工业级 C++ 计算核心可识别的 Eigen 张量。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("MCMC_BRIDGE")

        # 从 rules.yaml 提取全局随机种子
        self.seed = self.cfg._config.get('project', {}).get('random_seed', 2026)

        # 初始化先验概率生成器
        self.prior_generator = VotePriors()

    def _prepare_sampler_config(self) -> Any:
        """
        [协议对齐]: 将 YAML 的业务语义参数映射到 C++ 的物理计算参数。
        物理意义：决定了似然函数能量景观的“陡峭程度”与“约束硬度”。
        """
        cpp_cfg = mcm_core_lib.SamplerConfig()

        # 获取推理配置块
        inf_params = self.cfg.get_inference_params()
        m_params = inf_params.get('mcmc_strategy', {})
        l_params = inf_params.get('constraints', {})

        # 1. 采样链控制 (Sampling Control)
        cpp_cfg.n_chains = int(m_params.get('n_chains', 23))
        cpp_cfg.n_samples = int(m_params.get('samples_per_chain', 100000))
        cpp_cfg.burn_in = int(cpp_cfg.n_samples * m_params.get('burn_in_ratio', 0.2))
        cpp_cfg.thinning = int(m_params.get('thinning', 10))
        cpp_cfg.init_step_size = float(m_params.get('jump_size', 0.05))
        cpp_cfg.adaptive = True
        cpp_cfg.seed = self.seed
        cpp_cfg.return_traces = m_params.get('return_traces', False)

        # 2. 似然函数刚度 (Likelihood Stiffness - 核心修复点)
        # 物理映射关系：
        # rank_tau -> Soft-Rank 平滑度
        # rank_violation_penalty -> 淘汰事实的硬约束力度
        # bottom_two_penalty_weight -> 危险区信号的软约束力度
        cpp_cfg.rank_tau = float(l_params.get('rank_tau', 0.05))
        cpp_cfg.elim_penalty = float(l_params.get('rank_violation_penalty', 1000.0))
        cpp_cfg.jeopardy_penalty = float(l_params.get('bottom_two_penalty_weight', 200.0))
        cpp_cfg.entropy_weight = float(l_params.get('entropy_regularization', 0.05))

        return cpp_cfg

    def run_week_inference(self, week_df: pd.DataFrame) -> Optional[Any]:
        """
        执行单周贝叶斯反演。
        """
        if week_df.empty: return None

        season = int(week_df['season'].iloc[0])
        mechanism_str = self.cfg.get_mechanism(season)

        # 映射机制类型到 C++ 枚举
        mech_type = (mcm_core_lib.MechanismType.RANK_BASED
                     if mechanism_str == "RANK"
                     else mcm_core_lib.MechanismType.PERCENT_BASED)

        # 1. [零拷贝契约]：强制 Numpy 数组在内存中连续且对齐
        # 理由：C++ Eigen::Ref 需要 C-style 连续内存块
        judge_signals = np.ascontiguousarray(
            week_df['week_avg_score'].values, dtype=np.float64
        )

        # 2. [Censorship 定位]：识别被淘汰选手的索引
        # 逻辑：在当前周 (week_num) 状态为 Eliminated 且淘汰周等于当前周
        elim_mask = (week_df['final_status'] == 'Eliminated') & \
                    (week_df['eliminated_week'] == week_df['week_num'])
        elim_indices = np.where(elim_mask)[0]
        elim_idx = int(elim_indices[0]) if len(elim_indices) > 0 else -1

        # 3. [危险区信号提取]
        if 'had_bottom_two_record' in week_df.columns:
            jeopardy_mask = np.ascontiguousarray(
                week_df['had_bottom_two_record'].values, dtype=np.int32
            )
        else:
            jeopardy_mask = np.zeros(len(week_df), dtype=np.int32)

        # 4. [贝叶斯先验注入]
        # 基于行业、舞伴等因子生成的 Informative Prior
        prior_mu = np.ascontiguousarray(
            self.prior_generator.informative_prior(week_df), dtype=np.float64
        )

        # 5. [跨语言执行]
        try:
            cpp_cfg = self._prepare_sampler_config()
            sampler = mcm_core_lib.MCMCSampler(cpp_cfg)

            # 这里由于 pybind11 的 call_guard，GIL 会被释放，23 核 CPU 将瞬间满载
            result = sampler.run_parallel_inference(
                judge_signals,
                elim_idx,
                jeopardy_mask,
                prior_mu,
                mech_type
            )
            return result

        except Exception as e:
            self.logger.error(f"❌ C++ 推理核心在 Season {season} 发生崩溃: {str(e)}")
            return None

    def run_batch_inference(self, df_gold: pd.DataFrame) -> pd.DataFrame:
        """
        全量批处理流水线：将 Gold 层因子熔炼为 Platinum 后验层。
        """
        # 预加载配置打印日志
        sample_config = self._prepare_sampler_config()
        self.logger.info(f"🚀 启动分布式推理机 [Chains: {sample_config.n_chains} | Depth: {sample_config.n_samples}]")

        platinum_records = []
        # 确保按时间线顺序处理，防止因果回溯
        df_sorted = df_gold.sort_values(['season', 'week_num'])
        groups = df_sorted.groupby(['season', 'week_num'])

        for (s, w), week_data in groups:
            # 必须 reset_index，确保 DataFrame 内部索引与传给 C++ 的数组 Offset 严格一致 (0-based)
            week_data = week_data.reset_index(drop=True)

            # 跳过异常样本（如只有 1 人的周）
            if len(week_data) < 2: continue

            self.logger.debug(f"正在反演 S{s} Week {w} (N={len(week_data)})...")
            res = self.run_week_inference(week_data)

            if res:
                contestants = week_data['celebrity_name'].values
                for i in range(len(contestants)):
                    platinum_records.append({
                        'season': s,
                        'week_num': w,
                        'celebrity_name': contestants[i],
                        # 物理产物：后验分布的一阶/二阶矩
                        'est_fan_vote_mu': res.posterior_mean[i],
                        'est_fan_vote_sigma': res.posterior_std[i],
                        # 统计诊断：R-hat < 1.1 代表学术级收敛
                        'r_hat_max': res.r_hat,
                        'fidelity_score': res.fidelity_score,
                        'mcmc_converged': res.converged,
                        'ess_estimate': res.ess,
                        'acceptance_rate': res.acceptance_rate
                    })

        return pd.DataFrame(platinum_records)

if __name__ == "__main__":
    # 物理直觉压力测试
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    # 模拟一个 Bobby Bones 案例 (Season 27 Finals)
    mock_gold = pd.DataFrame({
        'season': [27, 27, 27, 27],
        'week_num': [10, 10, 10, 10],
        'celebrity_name': ['A', 'B', 'C', 'Bobby_Bones'],
        'week_avg_score': [9.5, 9.2, 9.0, 7.5], # 明星跳得烂
        'final_status': ['Safe', 'Safe', 'Safe', 'Winner'], # 但他赢了
        'eliminated_week': [10, 10, 10, 10],
        'had_bottom_two_record': [0, 0, 0, 0],
        'ballroom_partner': ['P1', 'P2', 'P3', 'P4']
    })

    wrapper = MCMCInferenceWrapper()
    print(">>> 启动混合架构集成测试...")
    results = wrapper.run_batch_inference(mock_gold)

    if not results.empty:
        print("\n✅ 测试通过：后验数据成功回传。")
        print(results[['celebrity_name', 'est_fan_vote_mu', 'r_hat_max']])