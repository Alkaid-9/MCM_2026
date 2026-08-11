"""
MCM 2026 Problem C: Bayesian Inference Orchestrator (Industrial Refactor v5.6)
Role: Mission Commander for Task 1 (Latent Variable Recovery).
Function: Orchestrating the fusion of 'Gold Factors' and 'Zipf Priors' into 'Platinum Posteriors'.
Physics: Minimizing the KL-Divergence between the empirical evidence and social priors.
Standard: Industrial HPC Scheduling / Fault-Tolerant Batch Processing.
"""

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# --- 核心组件导入 ---
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.core.priors import VotePriors
from src.bridge.mcmc_wrapper import MCMCInferenceWrapper
from src.core.uncertainty import UncertaintyQuantifier # 用于实时熵值监控

class BayesianVoteInference:
    """
    贝叶斯投票反演系统 (The BIO Engine Host):
    负责调度 C++ 内核，管理先验注入，并生成铂金层数据 (Platinum Layer)。

    [学术任务对齐]:
    1. Latent Recovery: 恢复不可观测的观众投票 $\mathbf{v}$。
    2. Uncertainty Propagation: 将后验标准差 $\sigma$ 传递给下游归因模型。
    3. Regime Switching: 自动适配 Rank/Percent 赛制切换。
    """

    def __init__(self, df_gold: pd.DataFrame):
        self.cfg_loader = ConfigLoader()
        self.logger = setup_logger("BIO_ORCHESTRATOR")

        # 1. 数据接入 (Gold Layer)
        # 必须包含: season, week_num, celebrity_name, week_avg_score, final_status
        self.df = df_gold.copy()

        # 2. 实例化子系统
        self.prior_engine = VotePriors()          # 先验工厂
        self.bridge = MCMCInferenceWrapper()      # C++ 桥接器
        self.uq_monitor = UncertaintyQuantifier()   # 实时监控器 (轻量级)

        # 3. 结果注册表
        self.results_registry = []

    def _get_season_context(self, season: int) -> Dict[str, Any]:
        """
        [上下文感知]: 获取当前赛季的物理规则与先验超参数。
        """
        # 获取先验形状 (Alpha) 和 强度 (Strength)
        alpha, strength = self.cfg_loader.get_prior_params(season)
        # 获取赛制物理法则 (Rank vs Percent)
        regime = self.cfg_loader.get_mechanism_regime(season)
        return {
            'alpha': alpha,
            'strength': strength,
            'regime': regime
        }

    def run_inference_pipeline(self) -> pd.DataFrame:
        """
        [主程序]: 全量推断流水线。
        遍历所有赛季与周次，驱动 C++ 内核进行并行采样。
        """
        self.logger.info("=" * 60)
        self.logger.info(">>> 启动贝叶斯逆向优化引擎 (Task 1: Latent Recovery) <<<")
        self.logger.info(f"    处理样本规模: {len(self.df)} Rows")
        self.logger.info("=" * 60)

        # 1. 按时序分组 (Season -> Week)
        # 确保时间线性，这对于后续的时间序列分析至关重要
        df_sorted = self.df.sort_values(['season', 'week_num'])
        groups = df_sorted.groupby(['season', 'week_num'])

        # 2. 进度条封装
        total_tasks = len(groups)
        with tqdm(total=total_tasks, desc="BIO Parallel Sampling", unit="week") as pbar:

            for (s, w), week_df in groups:
                # [内存对齐]: 重置索引为 0-based，供 C++ 指针直接定位
                week_data = week_df.reset_index(drop=True)

                # [异常防御]: 跳过单人周或空数据 (数学上无法构成单纯形)
                if len(week_data) < 2:
                    pbar.update(1)
                    continue

                # [核心调用]: 委托 Bridge 执行 C++ 推断
                # 注意：Bridge 内部会处理 Winner Anchor 和 Jeopardy Mask
                cpp_result = self.bridge.run_week_inference(week_data)

                if cpp_result:
                    # [实时监控]: 计算瞬时熵值，用于日志抽样检查
                    # 注意：完整的 UQ 分析在 Stage 3 进行，这里只做简单的健康度检查
                    entropy = -np.sum(
                        cpp_result.posterior_mean * np.log2(cpp_result.posterior_mean + 1e-12)
                    )

                    # [数据归档]: 将 C++ 产出的无名数组映射回 Pandas 实体
                    self._archive_results(s, w, week_data, cpp_result, entropy)
                else:
                    self.logger.warning(f" [KERNEL FAIL] S{s:02d}W{w:02d} 采样未收敛或崩溃。")

                pbar.update(1)

        # 3. 数据合龙 (The Grand Join)
        return self._finalize_platinum_layer()

    def _archive_results(self, season: int, week: int,
                        week_data: pd.DataFrame,
                        res: Any,
                        entropy: float):
        """
        将 C++ 的原始输出 (Raw Output) 结构化归档。
        """
        contestants = week_data['celebrity_name'].values

        for i in range(len(contestants)):
            self.results_registry.append({
                # --- 主键 ---
                'season': season,
                'week_num': week,
                'celebrity_name': contestants[i],

                # --- 潜变量估计 (Latent Estimates) ---
                'est_fan_vote_mu': res.posterior_mean[i],    # 后验均值 (点估计)
                'est_fan_vote_sigma': res.posterior_std[i],  # 后验标准差 (不确定性)

                # --- 科学审计指标 (Scientific Rigor) ---
                'r_hat': res.r_hat,                          # 收敛性 (Gelman-Rubin)
                'ess': res.ess,                              # 有效样本量
                'fidelity': res.fidelity_score,              # 规则还原度
                'acceptance_rate': res.acceptance_rate,      # 采样效率
                'is_converged': res.converged,               # 二元收敛状态

                # --- 辅助指标 ---
                'inference_entropy': entropy                 # 系统熵 (Systemic Confusion)
            })

    def _finalize_platinum_layer(self) -> pd.DataFrame:
        """
        生成最终的 Platinum 层数据。
        将推断结果 (Inference) 与原始特征 (Gold Factors) 进行左连接。
        """
        if not self.results_registry:
            self.logger.critical("❌ 严重错误：推断结果为空！请检查 C++ 内核或数据输入。")
            return pd.DataFrame()

        df_inference = pd.DataFrame(self.results_registry)

        # 统计审计摘要
        conv_rate = df_inference['is_converged'].mean()
        avg_fidelity = df_inference['fidelity'].mean()

        self.logger.info("-" * 60)
        self.logger.info(f"推断流水线结束。")
        self.logger.info(f"  - 全局收敛率 (R-hat < 1.1): {conv_rate:.2%}")
        self.logger.info(f"  - 平均规则保真度 (Fidelity): {avg_fidelity:.4f}")
        self.logger.info("-" * 60)

        return df_inference

    def get_anomaly_report(self, df_platinum: pd.DataFrame) -> pd.DataFrame:
        """
        [Task 2 预处理]: 自动筛选“高分歧”案例。
        物理逻辑：寻找 Model Fidelity 低且最终名次好的选手（即：模型认为他该死，但他活着）。
        """
        self.logger.info("正在扫描历史异常锚点 (Anomaly Anchors)...")
        # 筛选逻辑：Fidelity < 0.8 且 进入决赛圈
        mask = (df_platinum['fidelity'] < 0.8) & \
               (df_platinum['final_status'].isin(['Winner', 'RunnerUp']))
        anomalies = df_platinum[mask].copy()

        if not anomalies.empty:
            self.logger.info(f"发现 {len(anomalies)} 个高分歧数据点（潜在的‘黑幕’或‘强力刷票’）。")

        return anomalies