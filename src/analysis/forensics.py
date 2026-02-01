"""
Mechanism Forensics & Multiverse Simulator (v5.6 - Final Counterfactual Engine)
Role: Auditing the impact of aggregation rules (Rank vs. Percent) and Judges' Save.
Function: Re-running 34 seasons of DWTS under alternative rule-sets (The "Multiverse").
Physics: Comparing "Low-Pass Filter" (Rank) vs. "Signal Amplifier" (Percent) dynamics.
Standard: Industrial Policy Simulation / O-Prize Forensic Rigor.
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr
import logging
import os


class MechanismForensics:
    """
    机制取证引擎：
    基于贝叶斯反演出的“真实民意”(Platinum Layer)，在不同赛制宇宙下重演历史。

    [平行宇宙设定]:
    1. Universe A (Rank Only): S1-S2 原始规则，强序数约束。
    2. Universe B (Percent Only): S3-S27 规则，高民意敏感度。
    3. Universe C (Rank + Save): S28+ 规则，序数约束 + 专家熔断。
    """

    def __init__(self, df_platinum: pd.DataFrame):
        self.logger = logging.getLogger("FORENSICS_ENGINE")
        # 核心：使用反演出的粉丝投票均值作为“底层民意真值”
        # 必须过滤掉无反演结果的脏数据
        self.data = df_platinum.dropna(subset=['est_fan_vote_mu', 'week_avg_score']).copy()

    def _compute_rank_logic(self, j_scores: np.ndarray, f_shares: np.ndarray) -> np.ndarray:
        """
        [规则 A]: 排名结合法 (Ordinal Aggregation)
        逻辑：Total_Rank = Rank(J) + Rank(F). 数值越小越好。
        物理意义：低通滤波器，过滤掉 f_shares 中的极端长尾脉冲。
        """
        # 1. 评委排名 (分数越高 -> 排名越小/好)
        j_rank = rankdata(-j_scores, method='min')
        # 2. 粉丝排名 (份额越大 -> 排名越小/好)
        f_rank = rankdata(-f_shares, method='min')

        total_rank_sum = j_rank + f_rank
        # 返回最终顺位 (1=Winner, N=Loser)
        return rankdata(total_rank_sum, method='min')

    def _compute_percent_logic(self, j_scores: np.ndarray, f_shares: np.ndarray) -> np.ndarray:
        """
        [规则 B]: 百分比结合法 (Cardinal Aggregation)
        逻辑：Total_Score = J% + F%. 数值越大越好。
        物理意义：线性放大器，完全保留流量信号的幅度信息。
        """
        # 1. 评委分归一化 (Z-Score 转 占比)
        # 注意：这里假设 j_scores 已经是正数（原始分），如果是 Z-score 需先 Sigmoid 或 MinMax
        # 既然上游是 week_avg_score (原始分)，直接归一化即可
        j_sum = np.sum(j_scores) + 1e-9
        j_pct = j_scores / j_sum

        # 2. 粉丝份额 (本身即为占比)
        f_pct = f_shares

        total_score = j_pct + f_pct
        # 返回最终顺位 (分数越高 -> 排名越小/好)
        return rankdata(-total_score, method='min')

    def _apply_judges_save_policy(self, group: pd.DataFrame, placements: np.ndarray) -> np.ndarray:
        """
        [规则 C 补丁]: 评委救济机制 (The Judges' Save / Circuit Breaker)
        逻辑：识别 Bottom Two，强制保留其中技术分更高的一位。
        """
        n = len(placements)
        if n < 2: return placements  # 无法执行二选一

        # 1. 识别危险区：排名数字最大的两个人 (N 和 N-1)
        # argsort 返回的是从小到大的索引，所以最后两个是倒数第一和倒数第二
        sorted_indices = np.argsort(placements)
        bottom_two_indices = sorted_indices[-2:]  # [倒数第二 idx, 倒数第一 idx]

        idx_a, idx_b = bottom_two_indices[0], bottom_two_indices[1]

        # 2. 专家裁决 (Meritocracy Check)
        score_a = group.iloc[idx_a]['week_avg_score']
        score_b = group.iloc[idx_b]['week_avg_score']

        # 3. 修正排名
        # 技术分高者 -> 强制设为倒数第二 (Safe)
        # 技术分低者 -> 强制设为倒数第一 (Eliminated)
        new_placements = placements.copy()

        if score_a >= score_b:
            saved_idx, killed_idx = idx_a, idx_b
        else:
            saved_idx, killed_idx = idx_b, idx_a

        # 模拟排名交换：Killed 变成 N (最大), Saved 变成 N-1
        # 注意：这里只是逻辑上的 rank 修正，不改变其他人的相对顺序
        new_placements[killed_idx] = n
        new_placements[saved_idx] = n - 1

        return new_placements

    def run_multiverse_simulation(self) -> pd.DataFrame:
        """
        [主程序]: 全量执行平行宇宙推演。
        """
        self.logger.info(">>> 启动平行宇宙机制模拟 (Multiverse Simulation)...")

        sim_records = []
        # 按周分组进行独立模拟
        groups = self.data.groupby(['season', 'week_num'])

        for (s, w), group in groups:
            # 必须重置索引以对其 numpy 数组
            group = group.reset_index(drop=True)

            # 提取信号源
            j_scores = group['week_avg_score'].values
            f_shares = group['est_fan_vote_mu'].values

            # 1. 模拟 Rank 宇宙
            rank_pos = self._compute_rank_logic(j_scores, f_shares)

            # 2. 模拟 Percent 宇宙
            pct_pos = self._compute_percent_logic(j_scores, f_shares)

            # 3. 模拟 Rank + Save 宇宙 (S28+ 现状)
            rank_save_pos = self._apply_judges_save_policy(group, rank_pos)

            # 记录结果
            for i in range(len(group)):
                # 计算位移 (Displacement): Rank 对比 Percent 的名次变化
                # 正值表示在 Rank 下排名更靠后（表现更差），负值表示在 Rank 下排名提升
                disp = rank_pos[i] - pct_pos[i]

                # 判定是否在各宇宙中被淘汰 (Rank = N)
                n = len(group)
                is_elim_rank = (rank_pos[i] == n)
                is_elim_pct = (pct_pos[i] == n)
                is_elim_save = (rank_save_pos[i] == n)

                sim_records.append({
                    'season': s,
                    'week_num': w,
                    'celebrity_name': group.loc[i, 'celebrity_name'],
                    'actual_judges_score': j_scores[i],
                    'inferred_fan_vote': f_shares[i],

                    # 模拟结果
                    'sim_rank_placement': rank_pos[i],
                    'sim_pct_placement': pct_pos[i],
                    'sim_save_placement': rank_save_pos[i],

                    # 差异指标
                    'mechanism_displacement': disp,
                    'is_rank_beneficiary': disp < 0,  # 在 Rank 下名次更好 -> 技术流受益
                    'is_percent_beneficiary': disp > 0,  # 在 Percent 下名次更好 -> 流量流受益

                    # 淘汰判定
                    'eliminated_in_rank': is_elim_rank,
                    'eliminated_in_percent': is_elim_pct,
                    'eliminated_in_save': is_elim_save
                })

        return pd.DataFrame(sim_records)

    def analyze_stability_metrics(self, df_sim: pd.DataFrame):
        """
        [Task 2 核心指标]: 机制鲁棒性与技术对齐度。
        计算 Spearman Rho (Merit Correlation)。
        """
        # 1. Rank 机制的技术对齐度
        # 负相关是正确的：分数越高，排名越小(1)
        r_corr, _ = spearmanr(df_sim['actual_judges_score'], df_sim['sim_rank_placement'])

        # 2. Percent 机制的技术对齐度
        p_corr, _ = spearmanr(df_sim['actual_judges_score'], df_sim['sim_pct_placement'])

        # 取绝对值方便比较强度
        r_merit = abs(r_corr)
        p_merit = abs(p_corr)

        self.logger.info("-" * 40)
        self.logger.info(f"🏆 机制效能审计 (Meritocracy Audit):")
        self.logger.info(f"Rank System Merit Correlation:    {r_merit:.4f}")
        self.logger.info(f"Percent System Merit Correlation: {p_merit:.4f}")

        gain = (r_merit - p_merit) / p_merit
        self.logger.info(f"结论: Rank 机制的技术权重比 Percent 高出 {gain:.2%}。")
        self.logger.info("-" * 40)

    def analyze_bobby_bones_butterfly_effect(self, df_sim: pd.DataFrame):
        """
        [案例特写]: Bobby Bones 的平行宇宙生死簿。
        """
        # 筛选 S27 的 Bobby Bones
        case = df_sim[
            (df_sim['season'] == 27) &
            (df_sim['celebrity_name'].str.contains("Bones", na=False))
            ].sort_values('week_num')

        if case.empty:
            self.logger.warning("未找到 Bobby Bones 数据，跳过案例分析。")
            return

        self.logger.info("🦋 蝴蝶效应案例: Bobby Bones (S27)")

        # 检查他在 Rank 宇宙中是否会死
        death_weeks = case[case['eliminated_in_rank'] == True]

        if not death_weeks.empty:
            first_death = death_weeks['week_num'].min()
            self.logger.info(f" [反事实]: 若采用 Rank 机制，Bobby Bones 将在第 {first_death} 周被淘汰。")
            self.logger.info(f" [现实]: 他在 Percent 机制下一直存活到了最后。")
            self.logger.info(" [结论]: Rank 机制成功构筑了‘技术防火墙’，拦截了极端流量。")
        else:
            self.logger.info(" [反事实]: 即便在 Rank 机制下，Bobby 依然存活（其流量优势过于巨大，击穿了低通滤波器）。")

        # 输出他在两种机制下的平均排名差异
        avg_diff = case['mechanism_displacement'].mean()
        self.logger.info(f" 平均排名位移: {avg_diff:.1f} (正值代表 Percent 对他更有利)")