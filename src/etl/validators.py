# ==============================================================================
# src/etl/validators.py
# Role: Data Quality Assurance (QA) & Audit Surgeon - Aggregated Version
# Function: Logical consistency checks for the Gold Factor Library.
# Standard: Academic Rigor (Causality & Censored Data Integrity).
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from src.etl.config_loader import ConfigLoader

class DataValidator:
    """
    数据验证引擎：
    在进入 Stage 2 (MCMC 逆向推断) 前执行最后的“逻辑体检”。
    针对聚合后的黄金因子库进行 Schema 完整性、生存逻辑一致性和数学单调性校验。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cfg = ConfigLoader()
        self.validation_passed = True
        self.logs = []
        self.logger = logging.getLogger("VALIDATOR")

    def _log_check(self, name: str, passed: bool, message: str):
        """标准化审计日志记录：PASSED 记录 INFO，FAILED 记录 ERROR 并触发熔断。"""
        status = "PASSED" if passed else "FAILED"
        log_entry = f"[{status}] {name}: {message}"
        self.logs.append(log_entry)

        if not passed:
            self.validation_passed = False
            self.logger.error(log_entry)
        else:
            self.logger.info(log_entry)

    def check_schema_integrity(self):
        """
        审计 1: 关键路径 Schema 检查。
        物理意义：反演模型需要完整的坐标系与技术信号。
        """
        # [核心修正]：对齐聚合后的黄金因子库列名
        critical_cols = [
            'celebrity_name', 'season', 'week_num',
            'week_avg_score', 'week_z_sum', 'tech_rank', 'partner_alpha'
        ]

        for col in critical_cols:
            if col not in self.df.columns:
                self._log_check(f"Schema Check - {col}", False, "关键列缺失")
                continue

            null_count = self.df[col].isna().sum()
            self._log_check(
                f"Null Integrity - {col}",
                null_count == 0,
                f"识别到 {null_count} 处非法缺失点"
            )

    def check_survival_logical_barrier(self):
        """
        审计 2: 生存逻辑屏障检查 (Censorship Sanity)。
        物理意义：验证“死人不会说话”原则。当前周次必须小于或等于该选手的淘汰周次。
        """
        if 'eliminated_week' not in self.df.columns:
            self._log_check("Survival Logic", False, "缺失 eliminated_week 标记")
            return

        # 填充未淘汰选手为无穷大 (999)
        temp_elim_week = self.df['eliminated_week'].fillna(999)
        # 寻找违规点：当前周 > 淘汰周
        ghosts = self.df[self.df['week_num'] > temp_elim_week]

        self._log_check(
            "Survival Logic Barrier",
            len(ghosts) == 0,
            f"检测到 {len(ghosts)} 条幽灵数据（被淘汰后仍有记录）"
        )

    def check_rank_score_monotonicity(self):
        """
        审计 3: 排名与分数单调性一致性。
        学术价值：Task 1 的约束矩阵极度依赖排名。如果 9 分的排名比 8 分低，优化器将无法收敛。
        """
        # 按 (赛季, 周) 分组校验
        groups = self.df.groupby(['season', 'week_num'])

        violations = 0
        for (s, w), group in groups:
            # 过滤无效样本
            valid_g = group.dropna(subset=['tech_rank', 'week_avg_score'])
            if len(valid_g) < 2: continue

            # 验证逻辑：分数越高 -> 排名数字应越小
            # 按分值降序排列，排名序列应是非减的（考虑并列情况）
            sorted_group = valid_g.sort_values('week_avg_score', ascending=False)
            ranks = sorted_group['tech_rank'].values

            is_monotonic = np.all(ranks[:-1] <= ranks[1:])

            if not is_monotonic:
                violations += 1
                self.logger.warning(f"S{s}W{w} 排名倒挂：Score={sorted_group['week_avg_score'].values}, Rank={ranks}")

        self._log_check(
            "Rank-Score Monotonicity",
            violations == 0,
            f"识别到 {violations} 个比赛周存在技术排名与得分的逻辑倒挂"
        )

    def check_feature_coverage(self):
        """
        审计 4: 黄金因子库覆盖率。
        重点检查 Partner Alpha 是否成功计算，确保冷启动逻辑没有产生 NaN。
        """
        target_features = ['partner_alpha', 'score_delta']
        for feat in target_features:
            if feat not in self.df.columns: continue

            null_count = self.df[feat].isna().sum()
            self._log_check(
                f"Feature Coverage - {feat}",
                null_count == 0,
                f"因子 {feat} 存在 {null_count} 个空值 (冷启动补全逻辑可能失效)"
            )

    def run_all(self) -> bool:
        """一键启动终极体检流水线"""
        self.logger.info("=" * 50)
        self.logger.info(">>> 启动黄金因子库 (Gold Layer) 逻辑审计 <<<")
        self.logger.info("=" * 50)

        self.check_schema_integrity()
        self.check_survival_logical_barrier()
        self.check_rank_score_monotonicity()
        self.check_feature_coverage()

        # 将审计记录持久化，供论文讨论数据质量时引用
        log_path = self.cfg.get_path('logs')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Audit Run @ {pd.Timestamp.now()} ---\n")
                f.write("\n".join(self.logs) + "\n")
        except:
            pass

        if self.validation_passed:
            self.logger.info("✅ 审计通过：数据集满足贝叶斯反演的数学前置条件。")
        else:
            self.logger.error("❌ 审计失败：检测到核心 Schema 或逻辑冲突，Pipeline 已拦截！")

        return self.validation_passed

# --- 单元测试 (对齐聚合后的 Mock 数据) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

    # 模拟聚合后的脏数据
    mock_gold = pd.DataFrame({
        'celebrity_name': ['A', 'B', 'DeadGuy'],
        'season': [1, 1, 1],
        'week_num': [2, 2, 2],
        'week_avg_score': [9.0, 8.0, 5.0],
        'week_z_sum': [2.5, 1.2, -3.0],
        'tech_rank': [1.0, 2.0, 1.0],      # [错误] B 比 DeadGuy 分高，排名却更后
        'eliminated_week': [10.0, 10.0, 1.0], # [错误] DeadGuy 第 1 周已走，第 2 周不该有记录
        'partner_alpha': [1.2, 0.9, np.nan] # [错误] 因子缺失
    })

    validator = DataValidator(mock_gold)
    result = validator.run_all()

    print(f"\n最终审计结论: {'通过' if result else '不通过'}")
    # 预期：不通过，触发三个 FAILED 报警