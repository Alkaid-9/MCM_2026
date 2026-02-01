# ==============================================================================
# src/bridge/mcmc_wrapper.py
# Role: Industrial High-Performance MCMC Python Bridge (v5.7 - Final Sync)
# Fix: Aligned with ConfigLoader v2.2 APIs (Removed deprecated getters).
# ==============================================================================

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any

# --- 动态加载 C++ 内核 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
if str(BIN_DIR) not in sys.path:
    sys.path.append(str(BIN_DIR))

try:
    import mcm_core_lib
except ImportError as e:
    # 允许在无编译环境下仅做代码检查，但在运行时熔断
    mcm_core_lib = None

from src.etl.config_loader import ConfigLoader
from src.core.priors import VotePriors


class MCMCInferenceWrapper:
    """
    贝叶斯推理指挥官：
    负责将 Pandas 数据流封装为 C++ 强类型张量。
    """

    def __init__(self):
        if mcm_core_lib is None:
            raise ImportError("C++ 内核未加载。请先编译: cd cpp_kernel/build && cmake .. && make")

        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("MCMC_BRIDGE")

        # [关键修复] 使用通用的 get_inference_params 获取所有 MCMC 配置
        inf_params = self.cfg.get_inference_params()

        self.strategy = inf_params.get('mcmc_strategy', {})
        self.constraints = inf_params.get('constraints', {})

        # 随机种子
        self.seed = self.cfg.load_config().get('project', {}).get('random_seed', 2026)

        # 初始化先验生成器 (它现在也使用修复后的 ConfigLoader)
        self.prior_generator = VotePriors()

    def _build_cpp_config(self, season: int) -> Any:
        """构建 C++ SamplerConfig 结构体"""
        cpp_cfg = mcm_core_lib.SamplerConfig()

        # 1. 采样控制
        cpp_cfg.n_chains = int(self.strategy.get('n_chains', 23))
        cpp_cfg.n_samples = int(self.strategy.get('n_samples', 100000))
        cpp_cfg.burn_in_ratio = float(self.strategy.get('burn_in_ratio', 0.5))
        cpp_cfg.thinning = int(self.strategy.get('thinning', 20))
        cpp_cfg.init_step_size = float(self.strategy.get('jump_size', 0.05))
        cpp_cfg.adaptive = True
        cpp_cfg.seed = self.seed
        cpp_cfg.return_traces = bool(self.strategy.get('return_traces', False))

        # 2. 能量函数刚度
        cpp_cfg.rank_tau = float(self.constraints.get('rank_tau', 0.05))
        cpp_cfg.elim_penalty = float(self.constraints.get('rank_violation_penalty', 1200.0))
        cpp_cfg.jeopardy_penalty = float(self.constraints.get('bottom_two_penalty_weight', 150.0))

        # 3. 动态先验注入
        # 获取先验强度 (Strength)
        alpha_std, strength_std = self._resolve_prior_params(season)
        cpp_cfg.prior_strength = float(strength_std)

        # 4. 机制逻辑开关 (S28+ Judges' Save)
        # 只有在 S28 及以后才开启 enable_judge_save
        mech_cfg = self.cfg.load_config().get('mechanisms', {})
        trans_season = mech_cfg.get('transition_season', 28)
        cpp_cfg.enable_judge_save = (season >= trans_season)

        return cpp_cfg

    def _resolve_prior_params(self, season: int):
        """辅助函数：解析 priors.yaml 中的覆盖逻辑"""
        priors = self.cfg.get_priors_config()
        overrides = priors.get('season_overrides', {})
        defaults = self.constraints

        strength = float(defaults.get('prior_strength', 50.0))
        alpha = 1.2

        s_key = season if season in overrides else str(season)
        if s_key in overrides:
            spec = overrides[s_key]
            if 'prior_strength' in spec: strength = float(spec['prior_strength'])

        return alpha, strength

    def run_week_inference(self, week_df: pd.DataFrame):
        """执行单周推断"""
        if week_df.empty: return None

        season = int(week_df['season'].iloc[0])
        week = int(week_df['week_num'].iloc[0])

        # 赛制判定
        regime_str = self.cfg.get_mechanism(season)
        mech_type = (mcm_core_lib.MechanismType.RANK_BASED
                     if regime_str == "RANK"
                     else mcm_core_lib.MechanismType.PERCENT_BASED)

        # 数据准备 (Zero-Copy)
        judge_signals = np.ascontiguousarray(week_df['week_avg_score'].values, dtype=np.float64)

        # 约束条件
        elim_mask = (week_df['final_status'] == 'Eliminated') & (week_df['eliminated_week'] == week)
        elim_idx = int(np.where(elim_mask)[0][0]) if elim_mask.any() else -1

        winner_mask = week_df['final_status'] == 'Winner'
        winner_idx = int(np.where(winner_mask)[0][0]) if winner_mask.any() else -1

        if 'had_bottom_two_record' in week_df.columns:
            jeopardy_mask = np.ascontiguousarray(week_df['had_bottom_two_record'].values, dtype=np.int32)
        else:
            jeopardy_mask = np.zeros(len(week_df), dtype=np.int32)

        # 先验构建
        prior_mu = np.ascontiguousarray(
            self.prior_generator.informative_prior(week_df), dtype=np.float64
        )

        try:
            cpp_config = self._build_cpp_config(season)
            sampler = mcm_core_lib.MCMCSampler(cpp_config)

            result = sampler.run_parallel_inference(
                judge_signals, elim_idx, jeopardy_mask, prior_mu, mech_type, winner_idx
            )
            return result
        except Exception as e:
            self.logger.error(f"S{season}W{week} 内核崩溃: {e}")
            return None

    def run_batch_inference(self, df_gold: pd.DataFrame) -> pd.DataFrame:
        """全量批处理 (含数据缝合)"""
        self.logger.info(" [STAGE 2] 启动贝叶斯反演生产线...")
        platinum_records = []

        df_sorted = df_gold.sort_values(['season', 'week_num'])
        groups = df_sorted.groupby(['season', 'week_num'])

        # 计数器
        total_batches = len(groups)
        processed = 0

        for (s, w), week_data in groups:
            week_data = week_data.reset_index(drop=True)
            if len(week_data) < 2: continue

            res = self.run_week_inference(week_data)
            if res:
                contestants = week_data['celebrity_name'].values
                for i in range(len(contestants)):
                    platinum_records.append({
                        'season': s,
                        'week_num': w,
                        'celebrity_name': contestants[i],
                        'est_fan_vote_mu': res.posterior_mean[i],
                        'est_fan_vote_sigma': res.posterior_std[i],
                        'r_hat': res.r_hat,
                        'ess': res.ess,
                        'fidelity': res.fidelity_score,
                        'is_converged': res.converged,
                        'acceptance_rate': res.acceptance_rate
                    })

            processed += 1
            if processed % 50 == 0:
                self.logger.info(f" 进度: {processed}/{total_batches} 周已完成...")

        # 数据缝合
        df_inference = pd.DataFrame(platinum_records)
        self.logger.info("正在执行数据缝合 (Joining Metadata)...")

        # Left Join 恢复 final_status, industry 等元数据
        df_platinum = df_gold.merge(
            df_inference,
            on=['season', 'week_num', 'celebrity_name'],
            how='left'
        )

        # 验证完整性
        if 'final_status' not in df_platinum.columns:
            raise RuntimeError("元数据丢失！")

        return df_platinum


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # 模拟 S27 决赛周 (Bobby Bones 夺冠)
    mock_week = pd.DataFrame({
        'season': [27, 27, 27],
        'week_num': [10, 10, 10],
        'celebrity_name': ['Bobby Bones', 'Milo', 'Evanna'],
        'week_avg_score': [24.0, 30.0, 30.0],  # Bobby 分最低
        'final_status': ['Winner', 'RunnerUp', 'RunnerUp'],
        'eliminated_week': [10, 10, 10],
        'had_bottom_two_record': [0, 0, 0],
        'industry_group': ['Music', 'Actor', 'Actor'],
        'partner_alpha': [2.0, 1.0, 1.0],  # 强力舞伴
        'score_delta': [0.5, 0.1, 0.1]  # 进步快
    })

    wrapper = MCMCInferenceWrapper()
    print("\n>>> 正在执行 Bobby Bones 悖论反演测试...")
    res = wrapper.run_week_inference(mock_week)

    if res:
        print(f" [SUCCESS] 反演完成. Converged: {res.converged}")
        print(f" [AUDIT] R-hat: {res.r_hat:.4f}, Fidelity: {res.fidelity_score:.2%}")
        print(f" [RESULT] Bobby Bones 估计得票率: {res.posterior_mean[0]:.2%}")
    else:
        print(" [FAIL] 反演失败")