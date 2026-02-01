# ==============================================================================
# src/etl/validators.py
# Role: Data Quality Assurance (QA) & Audit Surgeon (v5.5 - O-Prize Edition)
# Function: Logical consistency checks, survival sanity, and monotonic alignment.
# Physics: Ensuring the "Energy Landscape" for MCMC is logically convex and feasible.
# Standard: Academic Rigor / Data Forensic Auditing / Failure Atomicity.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class DataValidator:
    """
    数据验证引擎：
    在进入 Stage 2 (MCMC 逆向推断) 前执行最后的“逻辑体检”。

    [学术价值]:
    1. Monotonicity: 验证 $Score_A > Score_B \iff Rank_A < Rank_B$。
    2. Censorship: 遵循“死人不会说话”原则，拦截淘汰后的幽灵数据。
    3. Variance: 识别“信号坍缩”周次（评委打分完全一致），预警似然函数失效。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cfg = ConfigLoader()
        self.logger = setup_logger("DATA_VALIDATOR")
        self.validation_passed = True
        self.audit_trail = []  # 存储详细审计日志，用于生成论文附录

    def _record_event(self, check_name: str, status: str, message: str):
        """记录审计条目：PASSED 记录 INFO，FAILED 记录 ERROR 并触发系统熔断。"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            "time": timestamp,
            "check": check_name,
            "status": status,
            "msg": message
        }
        self.audit_trail.append(entry)

        if status == "FAILED":
            self.validation_passed = False
            self.logger.error(f"[{status}] {check_name}: {message}")
        elif status == "WARNING":
            self.logger.warning(f"[{status}] {check_name}: {message}")
        else:
            self.logger.info(f"[{status}] {check_name}: {message}")

    def check_schema_integrity(self):
        """
        审计 1: 关键路径 Schema 完整性。
        物理意义：反演模型需要完整的坐标系（Season/Week）与信号源（Score/Rank）。
        """
        # 对齐聚合后的黄金因子库 (Gold Layer) 核心列
        critical_cols = [
            'celebrity_name', 'season', 'week_num',
            'week_avg_score', 'week_z_sum', 'tech_rank', 'partner_alpha'
        ]

        for col in critical_cols:
            if col not in self.df.columns:
                self._record_event("Schema Check", "FAILED", f"关键因子缺失: {col}")
                continue

            null_count = self.df[col].isna().sum()
            if null_count > 0:
                self._record_event("Null Integrity", "FAILED", f"列 {col} 识别到 {null_count} 处非法缺失点")
            else:
                pass  # Silent success

        self._record_event("Schema Check", "PASSED" if self.validation_passed else "FAILED", "Schema 完整性扫描完成")

    def check_survival_logical_barrier(self):
        """
        审计 2: 生存逻辑一致性 (Censorship Sanity)。
        物理意义：验证选手在被淘汰后不再产生观测信号。
        """
        if 'eliminated_week' not in self.df.columns:
            self._record_event("Survival Logic", "FAILED", "缺失 eliminated_week 标记")
            return

        # 1. 寻找违规点：当前周 (week_num) > 理论淘汰周 (eliminated_week)
        # 填充未淘汰选手（Winners）为无穷大
        temp_elim = self.df['eliminated_week'].fillna(999)
        ghost_records = self.df[self.df['week_num'] > temp_elim]

        if not ghost_records.empty:
            sample_names = ghost_records['celebrity_name'].unique()[:3]
            self._record_event("Ghost Defense", "FAILED",
                               f"检测到 {len(ghost_records)} 条幽灵数据点。样本选手: {sample_names}")
        else:
            self._record_event("Ghost Defense", "PASSED", "生存屏障有效，无死后得分干扰。")

    def check_rank_score_monotonicity(self):
        """
        审计 3: 排名-分数单调性一致性 (Mathematical Monotonicity)。
        学术价值：C++ 似然函数中的约束矩阵极度依赖排名顺序。
        如果出现 $Score_i > Score_j$ 但 $Rank_i > Rank_j$ 的逻辑倒挂，
        会导致 MCMC 的能量景观（Energy Landscape）产生非物理奇点，导致无法收敛。
        """
        # 按 (赛季, 周) 颗粒度进行分组校验
        groups = self.df.groupby(['season', 'week_num'])
        total_violations = 0

        for (s, w), group in groups:
            # 过滤掉无法构成排名的样本
            valid_g = group.dropna(subset=['tech_rank', 'week_avg_score'])
            if len(valid_g) < 2: continue

            # 验证逻辑：分数越高，排名数字应越小 (1st < 2nd)
            # 按照分数降序排列
            sorted_group = valid_g.sort_values('week_avg_score', ascending=False)
            ranks = sorted_group['tech_rank'].values

            # 检查序列是否为非递减 (Non-decreasing)
            # 利用 numpy 向量化比较，处理 ties（并列）
            is_monotonic = np.all(ranks[:-1] <= ranks[1:])

            if not is_monotonic:
                total_violations += 1
                self.logger.warning(
                    f" [FORENSIC ALERT] S{s}W{w} 发生排名倒挂：Rank={ranks}, Score={sorted_group['week_avg_score'].values}")

        if total_violations > 0:
            self._record_event("Monotonicity", "FAILED", f"识别到 {total_violations} 个比赛周存在逻辑倒挂")
        else:
            self._record_event("Monotonicity", "PASSED", "排名与分数单调一致性校验通过。")

    def check_signal_variance(self):
        """
        审计 4: 信号区分度检查 (SNR Baseline)。
        物理意义：如果某周所有选手得分完全一样，说明专家信号发生“维度坍塌”。
        """
        # 统计每场比赛的分数标准差
        episode_stds = self.df.groupby(['season', 'week_num'])['week_avg_score'].std().fillna(0)
        collapsed_episodes = (episode_stds < 1e-4).sum()

        if collapsed_episodes > 0:
            self._record_event("Signal Clarity", "WARNING", f"存在 {collapsed_episodes} 场信号坍缩（评委打分无区分度）")
        else:
            self._record_event("Signal Clarity", "PASSED", "单集信号信噪比处于健康区间。")

    def run_all(self) -> bool:
        """一键启动全方位法医级体检"""
        self.logger.info("=" * 60)
        self.logger.info(">>> 启动黄金因子库 (Gold Layer) 深度逻辑审计 <<<")
        self.logger.info("=" * 60)

        self.check_schema_integrity()
        self.check_survival_logical_barrier()
        self.check_rank_score_monotonicity()
        self.check_signal_variance()

        # 持久化审计结果：用于在论文中展示数据清理的详实度
        log_path = self.cfg.get_path('logs').replace('.log', '_audit_report.json')
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.audit_trail, f, indent=4)
        except:
            pass

        if self.validation_passed:
            self.logger.info("✅ 全量审计通过：数据集满足贝叶斯反演的数学前置条件。")
        else:
            self.logger.error("❌ 审计失败：检测到核心逻辑冲突。请检查上游 ETL 逻辑！")

        return self.validation_passed


# --- 单元测试 (模拟致命逻辑冲突) ---
if __name__ == "__main__":
    # 配置模拟数据
    logging.basicConfig(level=logging.INFO)
    mock_gold = pd.DataFrame({
        'celebrity_name': ['Star_A', 'Star_B', 'Star_C'],
        'season': [1, 1, 1],
        'week_num': [2, 2, 2],
        'week_avg_score': [9.5, 8.0, 7.0],  # A > B > C
        'tech_rank': [2.0, 1.0, 3.0],  # [错误] A 分最高但排名第 2，B 分低却排名第 1 (Monotonicity Fail)
        'eliminated_week': [10.0, 10.0, 1.0],  # [错误] Star_C 第 1 周已淘汰，第 2 周不该有分 (Ghost Fail)
        'partner_alpha': [1.2, 0.9, 1.0],
        'week_z_sum': [2.0, 0.5, -1.0]
    })

    print("\n--- 启动模拟审计测试 ---")
    validator = DataValidator(mock_gold)
    is_ok = validator.run_all()

    print(f"\nFinal Audit Verdict: {'SAFE TO INVERT' if is_ok else 'DATA CORRUPTED'}")
    # 预期输出：DATA CORRUPTED