# ==============================================================================
# src/simulators/multiverse_engine.py
# Role: Temporal Forensics & Multiverse Engine (v5.1 - Schema Fixed Edition)
# Fix: Aligned output columns with SurvivalAnalyst requirements (week_avg_score, final_status).
# Standard: O-Prize Quality / Counterfactual Rigor.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from scipy.stats import rankdata
from src.etl.config_loader import ConfigLoader

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

    def simulate_season(self, season_id: int, mode: str = "RANK", daw_params: Dict = None) -> List[Dict]:
        """
        [核心算法] 单赛季全时序重赛模拟。

        :param season_id: 目标赛季
        :param mode: "RANK" | "PERCENT" | "DAW"
        :param daw_params: (可选) DAW 优化的 (k, t0) 参数
        :return: 包含该赛季每一位选手、每一周模拟排名的列表 (Dict List)
        """
        season_data = self.df[self.df['season'] == season_id].copy()

        # 确定赛季总周数（用于 DAW 进度计算）
        weeks = sorted(season_data['week_num'].unique())
        total_weeks = max(weeks) if weeks else 0

        season_history = []

        # 遍历每一周 (Step-wise Counterfactual)
        for w in weeks:
            # 提取当周快照
            week_df = season_data[season_data['week_num'] == w].reset_index(drop=True)
            if len(week_df) < 2: continue # 决赛或异常周跳过

            # 1. 准备物理内核输入
            j_scores = week_df['week_avg_score'].values.astype(np.float64)
            f_votes = week_df['est_fan_vote_mu'].values.astype(np.float64)

            # 2. 确定权重
            w_j = 0.5 # 默认对等权重
            if mode == "DAW":
                w_j = self._get_daw_weight(w, total_weeks, daw_params)

            # 3. 计算本周模拟得分 (Simulated Scores)
            # 注意：这里我们计算全员得分，而不仅仅是淘汰者
            sim_scores = np.zeros(len(week_df))

            if mode == "PERCENT":
                # 百分比制：分数占比 + 投票占比 (越大越好)
                j_pct = j_scores / (np.sum(j_scores) + 1e-9)
                sim_scores = j_pct + f_votes

            elif mode == "RANK":
                # 排名制：Rank(J) + Rank(F) (越小越好)
                # 为了统一逻辑，我们取负号，变成“越大越好”
                j_rank = rankdata(-j_scores, method='min')
                f_rank = rankdata(-f_votes, method='min')
                sim_scores = -(j_rank + f_rank)

            elif mode == "DAW":
                # DAW：动态加权排名 (越小越好 -> 取负)
                j_rank = rankdata(-j_scores, method='min')
                f_rank = rankdata(-f_votes, method='min')
                sim_scores = -(w_j * j_rank + (1.0 - w_j) * f_rank)

            # 4. 生成模拟排名 (1=Winner, N=Loser)
            # sim_scores 越大越好，所以降序排列
            sim_placements = rankdata(-sim_scores, method='min')

            # 5. 记录结果 (Data Contract Alignment)
            # 【关键修复】：这里必须使用标准列名，供 SurvivalAnalyst 使用
            for i in range(len(week_df)):
                record = {
                    'season': season_id,
                    'week_num': w,
                    'celebrity_name': week_df.iloc[i]['celebrity_name'],

                    # --- 下游分析必需的标准列 ---
                    'week_avg_score': j_scores[i],       # 原始技术分
                    'est_fan_vote_mu': f_votes[i],       # 反演粉丝票
                    'final_status': week_df.iloc[i]['final_status'], # 生存状态元数据
                    'eliminated_week': week_df.iloc[i].get('eliminated_week', np.nan),

                    # --- 模拟结果 ---
                    'sim_score': sim_scores[i],
                    'sim_placement': sim_placements[i],  # 模拟排名
                    'universe': mode,                    # 平行宇宙标识
                    'judge_weight_applied': w_j,

                    # --- 辅助累积指标 ---
                    'cum_avg_tech_score': week_df.iloc[i].get('cum_avg_score', j_scores[i]),
                    'cum_avg_fan_vote': week_df.iloc[i].get('est_fan_vote_mu', f_votes[i])
                }
                season_history.append(record)

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