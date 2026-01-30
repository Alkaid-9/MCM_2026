"""
MCM 2026 Problem C: Data Quality Assurance (QA) & Audit Surgeon
Role: Logical consistency checks, survival sanity, and ranking-score alignment.
Standard: Academic Rigor (Censored Data Integrity) & Engineering Robustness.
"""

import pandas as pd
import numpy as np
import logging
from src.etl.config_loader import ConfigLoader


class DataValidator:
    """
    数据验证引擎：在进入贝叶斯反演引擎前执行最后的‘逻辑体检’。
    核心目标：防止由于‘幽灵数据’或‘逻辑跳变’导致的反演结果偏离真实物理规律。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cfg = ConfigLoader()
        self.validation_passed = True
        self.logs = []
        self.logger = logging.getLogger("VALIDATOR")

    def _log_check(self, name: str, passed: bool, message: str):
        status = "PASSED" if passed else "FAILED"
        log_entry = f"[{status}] {name}: {message}"
        self.logs.append(log_entry)
        if not passed:
            self.validation_passed = False
            self.logger.error(log_entry)
        else:
            self.logger.info(log_entry)

    def check_null_integrity(self):
        """
        审计 1: 关键路径空值检查。
        物理意义：反演模型需要完整的 (Score, Season, Week) 坐标系。
        """
        critical_cols = ['celebrity_name', 'season', 'week_num', 'raw_score', 'score_z']
        for col in critical_cols:
            null_count = self.df[col].isna().sum()
            self._log_check(
                f"Null Integrity - {col}",
                null_count == 0,
                f"识别到 {null_count} 处非法缺失点"
            )

    def check_score_distribution(self):
        """
        审计 2: 分数尺度与通胀审计。
        物理意义：检测是否存在 10.5 或 -1 这种破坏物理意义的打分。
        """
        limit = self.cfg._config['etl']['imputation']['bonus_threshold']
        out_of_bounds = self.df[(self.df['raw_score'] < 1) | (self.df['raw_score'] > limit)]

        self._log_check(
            "Score Boundary Check",
            len(out_of_bounds) == 0,
            f"识别到 {len(out_of_bounds)} 行分数超出物理阈值 [1, {limit}]"
        )

    def check_survival_logical_barrier(self):
        """
        审计 3: 生存逻辑屏障检查（死人不会跳舞）。
        物理意义：如果选手在 Week T 淘汰，Week T+1 不应有任何打分记录。
        这是论文中证明数据预处理严谨性的核心论据。
        """
        # 仅针对 Eliminated 状态的选手
        elim_df = self.df[self.df['final_status'] == 'Eliminated'].copy()
        if elim_df.empty: return

        # 核心逻辑：当前观测周 > 记录的淘汰周 = 逻辑冲突
        conflict = elim_df[elim_df['week_num'] > elim_df['eliminated_week']]

        self._log_check(
            "Survival Logic Barrier",
            len(conflict) == 0,
            f"识别到 {len(conflict)} 条‘幽灵观测’：选手淘汰后仍有评分信号"
        )

    def check_rank_score_consistency(self):
        """
        审计 4: 排名与分数一致性。
        优化：考虑并列情况，并记录具体错误坐标。
        """
        # 找出所有排名第一但分数不是最高的点
        # 或者分数最高但排名不是第一的点
        df_sorted = self.df.sort_values(['season', 'week_num', 'week_avg_score'], ascending=[True, True, False])

        # 核心逻辑：在每一组内，第一行的 tech_rank 必须是 1.0
        errors = df_sorted.groupby(['season', 'week_num']).head(1)
        conflict_cases = errors[errors['tech_rank'] != 1.0]

        if not conflict_cases.empty:
            for _, err in conflict_cases.iterrows():
                self.logger.warning(f"逻辑倒挂检测点: Season {err.season} Week {err.week_num} - {err.celebrity_name}")

        self._log_check(
            "Rank-Score Consistency",
            len(conflict_cases) == 0,
            f"识别到 {len(conflict_cases)} 个周次的排名与最高分不匹配"
        )

    def check_judge_panel_completeness(self):
        """
        审计 5: 评委席位完整性校验。
        物理意义：验证每一行数据是否成功映射到了真实的评委 ID（非 UNKNOWN）。
        """
        unknown_judges = self.df[self.df['judge_id'] == 'UNKNOWN']

        self._log_check(
            "Judge ID Mapping",
            len(unknown_judges) == 0,
            f"识别到 {len(unknown_judges)} 条未识别评委的数据，映射失败"
        )

    def run_all(self) -> bool:
        """一键启动终极体检"""
        self.logger.info("=" * 50)
        self.logger.info(">>> 启动数据质量红线审计 (Silver Layer Audit) <<<")
        self.logger.info("=" * 50)

        self.check_null_integrity()
        self.check_score_distribution()
        self.check_survival_logical_barrier()
        self.check_rank_score_consistency()
        self.check_judge_panel_completeness()

        # 持久化审计日志
        log_path = self.cfg.get_path('logs')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Validation Run @ {pd.Timestamp.now()} ---\n")
            f.write("\n".join(self.logs) + "\n")

        if self.validation_passed:
            self.logger.info("✅ 审计通过：数据集满足贝叶斯反演的数学前置条件。")
        else:
            self.logger.error("❌ 审计失败：检测到核心逻辑冲突，请修复原始数据处理逻辑！")

        return self.validation_passed


# --- 单元测试 (模拟逻辑冲突) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

    # 模拟一个带有‘幽灵观测’的脏数据
    mock_data = pd.DataFrame({
        'celebrity_name': ['Contender_A', 'Contender_A'],
        'season': [1, 1],
        'week_num': [1, 2],
        'raw_score': [8.0, 9.0],
        'score_z': [0.5, 1.2],
        'week_avg_score': [8.0, 9.0],
        'tech_rank': [1.0, 1.0],
        'final_status': ['Eliminated', 'Eliminated'],
        'eliminated_week': [1.0, 1.0],  # 逻辑冲突：第 1 周淘汰了，第 2 周不该有分
        'judge_id': ['CAI', 'LG']
    })

    validator = DataValidator(mock_data)
    validator.run_all()