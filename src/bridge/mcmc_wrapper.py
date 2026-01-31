"""
MCMC Python Bridge (Industrial Refactor v4.5)
Role: High-Performance Data Marshalling & Orchestrator.
Function: Bridging Pandas DataFrames to C++ Eigen Tensors with Bayesian Prior Injection.
Standard: O-Prize Hybrid Architecture / C++ Parallelism Support.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# --- 1. 动态加载 C++ 编译内核 (ABI Safe Loading) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
sys.path.append(str(BIN_DIR))

try:
    import mcm_core_lib  # 编译产生的 .so 或 .pyd 文件
except ImportError as e:
    logging.critical("❌ [CRITICAL] 核心 C++ 动态库加载失败。")
    logging.critical(f"建议操作：检查 {BIN_DIR} 是否存在 mcm_core_lib，或重新执行编译命令。")
    raise e

from src.etl.config_loader import ConfigLoader
from src.core.priors import VotePriors


class MCMCInferenceWrapper:
    """
    贝叶斯推理指挥官：
    负责将 Python 侧的统计特征转化为 C++ 采样引擎可识别的内存视图。
    """

    def __init__(self):
        self.cfg_loader = ConfigLoader()
        self.logger = logging.getLogger("MCMC_BRIDGE")
        # 初始化先验生成器
        self.prior_engine = VotePriors()

    def _prepare_sampler_config(self) -> mcm_core_lib.SamplerConfig:
        """
        [协议转换]：将 YAML 业务语义映射为 C++ 物理计算参数。
        """
        inf_cfg = self.cfg_loader.get_inference_params()
        strategy = inf_cfg.get('mcmc_strategy', {})
        constraints = inf_cfg.get('constraints', {})

        cpp_cfg = mcm_core_lib.SamplerConfig()

        # --- 采样链控制 ---
        cpp_cfg.n_chains = int(strategy.get('n_chains', 23))
        cpp_cfg.n_samples = int(strategy.get('n_samples', 100000))
        cpp_cfg.burn_in_ratio = float(strategy.get('burn_in_ratio', 0.5))
        cpp_cfg.thinning = int(strategy.get('thinning', 20))
        cpp_cfg.init_step_size = float(strategy.get('jump_size', 0.05))
        cpp_cfg.adaptive = True
        cpp_cfg.seed = self.cfg_loader._config.get('project', {}).get('random_seed', 2026)
        cpp_cfg.return_traces = bool(strategy.get('return_traces', True))

        # --- 能量函数刚度 (The Likelihood Manifold) ---
        cpp_cfg.rank_tau = float(constraints.get('rank_tau', 0.05))
        cpp_cfg.elim_penalty = float(constraints.get('rank_violation_penalty', 1200.0))
        cpp_cfg.jeopardy_penalty = float(constraints.get('bottom_two_penalty_weight', 150.0))

        # --- [关键注入] 贝叶斯先验强度 ---
        cpp_cfg.prior_strength = float(constraints.get('prior_strength', 50.0))

        # --- 机制特殊逻辑 ---
        # 是否启用第 28 季后的“评委保人”宽容约束
        cpp_cfg.enable_judge_save = False  # 初始为 False，将在单周推断中动态判定

        return cpp_cfg

    def run_week_inference(self, week_df: pd.DataFrame) -> Optional[mcm_core_lib.InferenceResult]:
        """
        执行单周的贝叶斯隐变量反演。
        """
        if week_df.empty: return None

        season = int(week_df['season'].iloc[0])
        week_num = int(week_df['week_num'].iloc[0])

        # 1. 确定当前周的赛制机制
        mechanism_str = self.cfg_loader.get_mechanism(season)
        mech_type = (mcm_core_lib.MechanismType.RANK_BASED
                     if mechanism_str == "RANK"
                     else mcm_core_lib.MechanismType.PERCENT_BASED)

        # 2. [内存对齐] 准备评委分信号 (Eigen::Ref 需要连续内存)
        judge_signals = np.ascontiguousarray(
            week_df['week_avg_score'].values, dtype=np.float64
        )

        # 3. [Censorship 定位] 识别本周被淘汰选手的索引
        # 逻辑：找出本周状态为 'Eliminated' 且淘汰周次等于当前周的记录
        elim_mask = (week_df['final_status'] == 'Eliminated') & \
                    (week_df['eliminated_week'] == week_num)
        elim_indices = np.where(elim_mask)[0]
        elim_idx = int(elim_indices[0]) if len(elim_indices) > 0 else -1

        # 4. [信号提取] 危险区标记 (Jeopardy Mask)
        if 'had_bottom_two_record' in week_df.columns:
            jeopardy_mask = np.ascontiguousarray(
                week_df['had_bottom_two_record'].values, dtype=np.int32
            )
        else:
            jeopardy_mask = np.zeros(len(week_df), dtype=np.int32)

        # 5. [贝叶斯注入] 构建 Zipf 先验均值向量
        prior_mu = np.ascontiguousarray(
            self.prior_engine.informative_prior(week_df), dtype=np.float64
        )

        # 6. [跨语言执行]
        try:
            cpp_cfg = self._prepare_sampler_config()
            # 动态调整 S28+ 的救济机制逻辑开关
            if season >= self.cfg_loader._config['mechanisms']['transition_season']:
                cpp_cfg.enable_judge_save = True

            sampler = mcm_core_lib.MCMCSampler(cpp_cfg)

            # 这里将释放 GIL，23 核 CPU 将瞬间满载计算
            result = sampler.run_parallel_inference(
                judge_signals,
                elim_idx,
                jeopardy_mask,
                prior_mu,
                mech_type
            )
            return result
        except Exception as e:
            self.logger.error(f"❌ C++ 推理核心在 Season {season} Week {week_num} 崩溃: {str(e)}")
            return None

    def run_batch_inference(self, df_gold: pd.DataFrame) -> pd.DataFrame:
        """
        全量批处理：将 Gold 层特征库熔炼为 Platinum 后验统计层。
        """
        self.logger.info("🚀 [STAGE 2/3] 启动大规模贝叶斯反演流水线...")

        platinum_records = []
        # 确保按时间线处理
        df_sorted = df_gold.sort_values(['season', 'week_num'])
        groups = df_sorted.groupby(['season', 'week_num'])

        for (s, w), week_data in groups:
            # 必须重置索引以保证 (0-indexed) 索引与 C++ 数组偏移一致
            week_data = week_data.reset_index(drop=True)

            if len(week_data) < 2: continue  # 过滤数据不足的异常周

            self.logger.info(f"正在反演 S{s:02d} Week {w:02d} (N={len(week_data)}) ...")

            res = self.run_week_inference(week_data)

            if res:
                # 提取后验数据并与选手信息对齐
                contestants = week_data['celebrity_name'].values
                for i in range(len(contestants)):
                    platinum_records.append({
                        'season': s,
                        'week_num': w,
                        'celebrity_name': contestants[i],
                        # 物理产物：估计得票率的后验分布特征
                        'est_fan_vote_mu': res.posterior_mean[i],
                        'est_fan_vote_sigma': res.posterior_std[i],
                        # 统计审计指标
                        'r_hat': res.r_hat,
                        'ess': res.ess,
                        'fidelity': res.fidelity_score,
                        'is_converged': res.converged,
                        'acceptance_rate': res.acceptance_rate
                    })

        return pd.DataFrame(platinum_records)