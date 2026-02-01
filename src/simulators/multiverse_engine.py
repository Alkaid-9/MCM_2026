# ==============================================================================
# src/simulators/multiverse_engine.py
# Role: Temporal Forensics & Multiverse Engine (v5.1 - Schema Fixed Edition)
# Fix: Aligned output columns with SurvivalAnalyst requirements (week_avg_score, final_status).
# Standard: Industrial Grade / Counterfactual Simulation
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from scipy.stats import rankdata
from src.etl.config_loader import ConfigLoader
from src.simulators.mechanism_sandbox import evaluate_elimination



class MultiverseEngine:
    """
    时空重演器：
    基于贝叶斯反演出的“真实民意” (Latent Fan Votes)，在不同赛制宇宙下重演历史。

    【核心修复逻辑】：
    1. Schema 对齐：输出列名必须与 Silver/Gold 层一致，防止下游分析 KeyError。
    2. 全员排位：计算所有人在新规则下的名次，而非仅决定淘汰者。
    3. 动态权重：支持 DAW 机制的 Sigmoid 动态权力移交。
    """

    def __init__(self, df_platinum: pd.DataFrame):
        self.logger = logging.getLogger("MULTIVERSE_ENGINE")

        # 确保输入数据包含必要的元数据列
        # 注意：est_fan_vote_mu 是 Stage 2 的产出
        required_cols = ['season', 'week_num', 'celebrity_name', 'week_avg_score', 'est_fan_vote_mu']
        for col in required_cols:
            if col not in df_platinum.columns:
                raise ValueError(f"MultiverseEngine 输入缺失关键列: {col}")

        self.df = df_platinum.copy()
        self.cfg_loader = ConfigLoader()
        self.cfg = self.cfg_loader.load_config()

    def _get_daw_weight(self, week: int, total_weeks: int, daw_params: Dict = None) -> float:
        """
        计算 DAW (Dynamic Adaptive Weighting) 宇宙下的动态评委权重。
        公式: w(t) = Base + Range * Sigmoid(k * (t - t0))
        """
        # 优先使用传入的优化参数 (Grid Search 用)，否则使用默认配置
        params = daw_params if daw_params else self.cfg.get('task4_mechanism_design', {}).get('dynamic_weighting', {})

        k = float(params.get('sigmoid_k', 10.0))
        t0 = float(params.get('sigmoid_t0', 0.6))

        # 归一化时间进度 (防止除零)
        t = week / (total_weeks + 1e-9)

        # Sigmoid 转换逻辑
        # 范围控制在 [0.3, 0.8]，确保任何阶段评委和观众都有投票权 (避免独裁)
        w_min = 0.3
        w_max = 0.8

        # 核心数学公式
        sigmoid_val = 1.0 / (1.0 + np.exp(-k * (t - t0) * 10))
        w_j = w_min + (w_max - w_min) * sigmoid_val

        return w_j

    def simulate_season(self, season_id: int, mode="RANK", daw_params=None):
        """
        全赛季重赛模拟核心。
        """
        # 提取当前赛季的原始全量数据，用于对照历史事实
        season_data = self.df[self.df['season'] == season_id].copy()
        weeks = sorted(season_data['week_num'].unique())
        total_weeks = max(weeks) if weeks else 0
        season_history = []

        # 初始化当前宇宙的幸存者
        sim_survivors = set(season_data['celebrity_name'].unique())
        history = []

        mode_map = {"PERCENT": 0, "RANK": 1, "DAW": 2}
        m_type = mode_map.get(mode, 1)

        for w in weeks:
            # 1. 提取当前宇宙幸存者在这一周的表现
            week_mask = (season_data['week_num'] == w) & (season_data['celebrity_name'].isin(sim_survivors))
            week_df = season_data[week_mask].reset_index(drop=True)

            if len(week_df) < 2: break  # 决赛或异常，停止迭代

            j_scores = week_df['week_avg_score'].values.astype(np.float64)
            f_votes = week_df['est_fan_vote_mu'].values.astype(np.float64)

            # 2. 确定权重与 Save 规则
            w_j = 0.5
            if mode == "DAW":
                w_j = self._get_daw_weight(w, total_weeks, daw_params)
            enable_save = (season_id >= 28)

            # 3. 调用 Numba 内核判定谁被淘汰
            elim_idx = evaluate_elimination(j_scores, f_votes, m_type, w_j, enable_save)
            loser_name = week_df.loc[elim_idx, 'celebrity_name']

            # ------------------------------------------------------------------
            # 4. 【核心修复】：定义命运背离点 (is_anomaly)
            # ------------------------------------------------------------------
            # 找到历史上真实这一周被淘汰的人
            hist_loser_slice = season_data[(season_data['week_num'] == w) &
                                           (season_data['final_status'] == 'Eliminated')]
            actual_historical_loser = hist_loser_slice['celebrity_name'].iloc[
                0] if not hist_loser_slice.empty else "NONE"

            # 对比：当前模拟宇宙的败者 vs 历史真实败者
            is_anomaly = (loser_name != actual_historical_loser)
            # ------------------------------------------------------------------

            # 5. 计算本周全员排名 (用于后续 SNR 审计)
            # 根据赛制逻辑重新计算得分
            if m_type == 0:  # PERCENT
                sim_scores = (j_scores / (j_scores.sum() + 1e-9)) + f_votes
            elif m_type == 1:  # RANK
                sim_scores = -(rankdata(-j_scores, method='min') + rankdata(-f_votes, method='min'))
            else:  # DAW
                j_r = rankdata(-j_scores, method='min')
                f_r = rankdata(-f_votes, method='min')
                sim_scores = -(w_j * j_r + (1.0 - w_j) * f_r)

            sim_placements = rankdata(-sim_scores, method='min')

            # 6. 记录存档 (Contract Alignment)
            for i in range(len(week_df)):
                record = {
                    'season': season_id,
                    'week_num': w,
                    'celebrity_name': week_df.iloc[i]['celebrity_name'],
                    'est_fan_vote_mu': f_votes[i],
                    'week_avg_score': j_scores[i],
                    'actual_judges_score': j_scores[i],
                    'sim_score': sim_scores[i],
                    'sim_placement': sim_placements[i],
                    'universe': mode,
                    'final_status': week_df.iloc[i]['final_status'],
                    'is_regime_anomaly': is_anomaly  # 现在定义好了
                }
                season_history.append(record)

            # 7. 物理剔除：被淘汰者从当前宇宙消失 (蝴蝶效应)
            sim_survivors.remove(loser_name)

        return season_history

    def run_all_universes(self) -> pd.DataFrame:
        """
        对所有 34 个赛季并行重演三个宇宙（RANK, PERCENT）。
        产出用于 Task 2 审计和 Task 4 优化的全量面板数据。
        """
        self.logger.info(">>> 启动‘平行宇宙’全量审计流水线...")

        all_seasons = sorted(self.df['season'].unique())
        full_history = []

        # 遍历赛季与模式
        # DAW 模式通常在 Stage 5 动态寻优时跑，这里只跑基准对比
        modes = ["RANK", "PERCENT"]

        for s in tqdm(all_seasons, desc="Simulating Universes"):
            for mode in modes:
                season_history = self.simulate_season(s, mode=mode)
                full_history.extend(season_history)

        sim_df = pd.DataFrame(full_history)

        # 基础审计
        self.logger.info(f"模拟完成。生成 {len(sim_df)} 条反事实记录。")
        return sim_df

# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 模拟 Stage 2 产出的 Platinum 数据结构
    try:
        mock_plat = pd.DataFrame({
            'season': [27, 27, 27],
            'week_num': [1, 1, 1],
            'celebrity_name': ['Bobby Bones', 'Pro Dancer', 'Loser'],
            'week_avg_score': [15.0, 25.0, 10.0],
            'est_fan_vote_mu': [0.6, 0.3, 0.1],
            'final_status': ['Winner', 'RunnerUp', 'Eliminated'],
            'eliminated_week': [10, 10, 1],
            'cum_avg_score': [15.0, 25.0, 10.0]
        })

        engine = MultiverseEngine(mock_plat)

        # 测试单赛季模拟
        print("\n--- Testing Season 27 Simulation ---")
        history = engine.simulate_season(27, mode="RANK")
        df_sim = pd.DataFrame(history)

        print(df_sim[['week_num', 'celebrity_name', 'sim_placement', 'final_status']].head())

        # 验证列名契约
        assert 'week_avg_score' in df_sim.columns, "KeyError Fix Failed!"
        assert 'final_status' in df_sim.columns, "Metadata Loss Detected!"

        print("\n✅ [PASS] 数据契约校验通过，Schema 已对齐。")

    except Exception as e:
        print(f"❌ 测试失败: {e}")