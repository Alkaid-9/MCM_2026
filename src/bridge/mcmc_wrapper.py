# ==============================================================================
# src/bridge/mcmc_wrapper.py
# Role: MCMC Python Bridge (Industrial Refactor v4.6 - Full-Rank Consistency)
# Function: Bridging Pandas to C++ with Metadata Restoration (The Grand Join).
# ==============================================================================

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# --- 1. 动态加载 C++ 编译内核 ---
# 这里的逻辑确保无论在哪个目录下运行，都能找到 bin 里的 .so/.pyd
_CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CURRENT_DIR.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"

if str(BIN_DIR) not in sys.path:
    sys.path.append(str(BIN_DIR))

try:
    import mcm_core_lib
except ImportError as e:
    logging.critical("❌ [CRITICAL] 核心 C++ 动态库加载失败。")
    logging.critical(f"建议：执行 cd cpp_kernel && mkdir -p build && cd build && cmake .. && make")
    raise e

from src.etl.config_loader import ConfigLoader
from src.core.priors import VotePriors

class MCMCInferenceWrapper:
    """
    贝叶斯推理指挥官：
    负责将 Python 侧的统计特征转化为 C++ 采样引擎可识别的内存视图，
    并在计算结束后，将后验概率分布缝合回原始数据面板。
    """
    def __init__(self):
        self.cfg_loader = ConfigLoader()
        self.logger = logging.getLogger("MCMC_BRIDGE")
        self.prior_engine = VotePriors()

    def _prepare_sampler_config(self) -> Any:
        """从 rules.yaml 中提取并构建 C++ SamplerConfig 对象"""
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
        cpp_cfg.seed = self.cfg_loader.load_config().get('project', {}).get('random_seed', 2026)
        cpp_cfg.return_traces = bool(strategy.get('return_traces', True))

        # --- 似然函数刚度 ---
        cpp_cfg.rank_tau = float(constraints.get('rank_tau', 0.05))
        cpp_cfg.elim_penalty = float(constraints.get('rank_violation_penalty', 1200.0))
        cpp_cfg.jeopardy_penalty = float(constraints.get('bottom_two_penalty_weight', 150.0))

        # --- 贝叶斯先验强度 ---
        cpp_cfg.prior_strength = float(constraints.get('prior_strength', 50.0))

        return cpp_cfg

    def run_week_inference(self, week_df: pd.DataFrame) -> Optional[Any]:
        """执行单周的贝叶斯隐变量反演"""
        if week_df.empty: return None

        season = int(week_df['season'].iloc[0])
        week_num = int(week_df['week_num'].iloc[0])

        mechanism_str = self.cfg_loader.get_mechanism(season)
        mech_type = (mcm_core_lib.MechanismType.RANK_BASED
                     if mechanism_str == "RANK"
                     else mcm_core_lib.MechanismType.PERCENT_BASED)

        # 1. 评委信号准备 (确保内存连续)
        judge_signals = np.ascontiguousarray(
            week_df['week_avg_score'].values, dtype=np.float64
        )

        # 2. 淘汰标签捕捉 (Censorship)
        elim_mask = (week_df['final_status'] == 'Eliminated') & \
                    (week_df['eliminated_week'] == week_num)
        elim_indices = np.where(elim_mask)[0]
        elim_idx = int(elim_indices[0]) if len(elim_indices) > 0 else -1

        # 3. 冠军锚点 (Winner Anchor) - 解决 Bobby Bones 悖论的关键
        winner_mask = (week_df['final_status'] == 'Winner')
        winner_indices = np.where(winner_mask)[0]
        winner_idx = int(winner_indices[0]) if len(winner_indices) > 0 else -1

        # 4. 危险区信号
        if 'had_bottom_two_record' in week_df.columns:
            jeopardy_mask = np.ascontiguousarray(
                week_df['had_bottom_two_record'].values, dtype=np.int32
            )
        else:
            jeopardy_mask = np.zeros(len(week_df), dtype=np.int32)

        # 5. 生成先验分布
        prior_mu = np.ascontiguousarray(
            self.prior_engine.informative_prior(week_df), dtype=np.float64
        )

        # 6. 调用 C++ 执行 MCMC
        try:
            cpp_cfg = self._prepare_sampler_config()
            # 动态开启评委救济逻辑
            if season >= 28:
                cpp_cfg.enable_judge_save = True

            sampler = mcm_core_lib.MCMCSampler(cpp_cfg)
            result = sampler.run_parallel_inference(
                judge_signals, elim_idx, jeopardy_mask,
                prior_mu, mech_type, winner_idx
            )
            return result
        except Exception as e:
            self.logger.error(f"❌ S{season}W{week_num} 推断崩溃: {str(e)}")
            return None

    def run_batch_inference(self, df_gold: pd.DataFrame) -> pd.DataFrame:
        """
        全量批处理：将 Gold 层数据熔炼为 Platinum 后验统计层。
        【核心修复】：在返回前执行数据缝合，防止丢失 final_status 等元数据。
        """
        self.logger.info("🚀 [STAGE 2/3] 启动贝叶斯反演生产线...")

        inference_results = []
        df_sorted = df_gold.sort_values(['season', 'week_num'])
        groups = df_sorted.groupby(['season', 'week_num'])

        for (s, w), week_data in groups:
            week_data = week_data.reset_index(drop=True)
            if len(week_data) < 2: continue

            self.logger.info(f"正在分析 S{s:02d} W{w:02d} (选手:{len(week_data)})")
            res = self.run_week_inference(week_data)

            if res:
                names = week_data['celebrity_name'].values
                for i in range(len(names)):
                    inference_results.append({
                        'season': s,
                        'week_num': w,
                        'celebrity_name': names[i],
                        'est_fan_vote_mu': res.posterior_mean[i],
                        'est_fan_vote_sigma': res.posterior_std[i],
                        'r_hat': res.r_hat,
                        'ess': res.ess,
                        'fidelity': res.fidelity_score,
                        'is_converged': res.converged
                    })

        # 1. 转化为中间 DataFrame
        df_inf = pd.DataFrame(inference_results)

        # 2. 【核心修复】执行 Grand Join (左连接)
        # 将反演出的数值结果合回到含有 Industry, final_status 的黄金表中
        self.logger.info("正在执行‘铂金级’数据缝合，恢复原始元数据...")
        df_platinum = df_gold.merge(
            df_inf,
            on=['season', 'week_num', 'celebrity_name'],
            how='left'
        )

        # 3. 校验关键列完整性
        if 'final_status' not in df_platinum.columns:
            raise KeyError("致命错误：缝合后的 df_platinum 丢失了 'final_status' 列！")

        return df_platinum