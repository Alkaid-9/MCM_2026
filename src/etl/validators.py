"""
MCM 2026 Problem C: Data Quality Assurance (QA) & Audit Surgeon
Role: Logical consistency checks, survival sanity, and ranking-score alignment.
Standard: Academic Rigor (Causality & Censored Data Integrity) & Engineering Robustness.
"""

import pandas as pd
import numpy as np
import logging
from src.etl.config_loader import ConfigLoader

class DataValidator:
    """
    数据验证引擎：在进入贝叶斯反演引擎前执行最后的‘逻辑体检’。
    
    【核心审计逻辑】：
    1. 生存逻辑屏障 (Ghost Protocol):
       验证 Transformer 是否成功拦截了所有“死后得分”的幽灵数据。

    2. 排名单调性 (Monotonicity Check):
       严格验证 (Score_A > Score_B) <==> (Rank_A < Rank_B)。
       防止由于浮点数精度或排序逻辑错误导致的秩逆转。

    3. 评委映射完整性 (Mapping Completeness):
       确保没有任何一行数据的评委 ID 是 'UNKNOWN'，保证 Task 3 归因分析的有效性。

    4. 因子覆盖率 (Feature Coverage):
       检查 Partner Alpha 等核心因子是否存在非法空值 (NaN)，验证冷启动逻辑是否生效。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cfg = ConfigLoader()
        self.validation_passed = True
        self.logs = []
        self.logger = logging.getLogger("VALIDATOR")

    def _log_check(self, name: str, passed: bool, message: str):
        """标准化审计日志记录"""
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
        物理意义：反演模型需要完整的坐标系。
        """
        # 必须存在的列
        critical_cols = ['celebrity_name', 'season', 'week_num', 'raw_score', 'score_z']

        for col in critical_cols:
            if col not in self.df.columns:
                self._log_check(f"Schema Check - {col}", False, "列缺失")
                continue

            null_count = self.df[col].isna().sum()
            self._log_check(
                f"Null Integrity - {col}",
                null_count == 0,
                f"识别到 {null_count} 处非法缺失点"
            )

    def check_survival_logical_barrier(self):
        """
        审计 2: 生存逻辑屏障检查 (死人不会说话)。
        逻辑：验证当前周 (week_num) 是否严格 <= 淘汰周 (eliminated_week)。
        """
        # 填充未淘汰者的淘汰周为无穷大
        temp_elim_week = self.df['eliminated_week'].fillna(999)

        # 寻找违规点：当前周 > 淘汰周
        ghosts = self.df[self.df['week_num'] > temp_elim_week]

        msg = ""
        if not ghosts.empty:
            sample = ghosts.head(3)[['season', 'week_num', 'celebrity_name', 'eliminated_week']].to_dict(orient='records')
            msg = f"发现 {len(ghosts)} 条幽灵数据！样本: {sample}"

        self._log_check(
            "Survival Logic Barrier",
            len(ghosts) == 0,
            msg if msg else "生存屏障有效，无幽灵观测。"
        )

    def check_rank_score_monotonicity(self):
        """
        审计 3: 排名与分数单调性一致性。
        学术价值：Task 1 的约束矩阵依赖于排名。如果 9 分的排名比 8 分低，优化器会崩溃。
        """
        # 按 (赛季, 周) 分组
        groups = self.df.groupby(['season', 'week_num'])

        violations = 0
        for (s, w), group in groups:
            # 过滤掉 tech_rank 为 NaN 的情况 (如无分周)
            valid_g = group.dropna(subset=['tech_rank', 'week_avg_score'])
            if len(valid_g) < 2: continue

            # 检查 Spearman 相关性：分数越高，排名数字应越小 -> 负相关 (-1.0)
            # 或者简单检查：排序分数后，排名是否递增
            sorted_by_score = valid_g.sort_values('week_avg_score', ascending=False)
            ranks = sorted_by_score['tech_rank'].values

            # 检查 rank 数组是否非严格递增 (允许并列)
            is_monotonic = np.all(ranks[:-1] <= ranks[1:])

            if not is_monotonic:
                violations += 1
                self.logger.warning(f"S{s}W{w} 排名倒挂: Score={sorted_by_score['week_avg_score'].values}, Rank={ranks}")

        self._log_check(
            "Rank-Score Monotonicity",
            violations == 0,
            f"识别到 {violations} 个比赛周存在分数与排名的逻辑倒挂"
        )

    def check_judge_panel_completeness(self):
        """
        审计 4: 评委席位映射完整性。
        """
        if 'judge_id' not in self.df.columns:
            self._log_check("Judge Mapping", False, "judge_id 列缺失")
            return

        unknown_judges = self.df[self.df['judge_id'] == 'UNKNOWN']

        self._log_check(
            "Judge ID Mapping",
            len(unknown_judges) == 0,
            f"识别到 {len(unknown_judges)} 条未识别评委的数据 (UNKNOWN)"
        )

    def check_feature_coverage(self):
        """
        审计 5: 黄金因子库覆盖率。
        重点检查 Partner Alpha 是否成功计算（冷启动填充验证）。
        """
        target_features = ['partner_alpha', 'score_delta', 'signal_strength_norm']
        for feat in target_features:
            if feat not in self.df.columns: continue

            null_count = self.df[feat].isna().sum()
            self._log_check(
                f"Feature Coverage - {feat}",
                null_count == 0,
                f"因子 {feat} 存在 {null_count} 个空值 (冷启动逻辑可能失效)"
            )

    def run_all(self) -> bool:
        """一键启动终极体检"""
        self.logger.info("=" * 50)
        self.logger.info(">>> 启动数据质量红线审计 (Audit Phase) <<<")
        self.logger.info("=" * 50)

        self.check_null_integrity()
        self.check_survival_logical_barrier()
        self.check_rank_score_monotonicity()
        self.check_judge_panel_completeness()
        self.check_feature_coverage()

        # 持久化审计日志
        log_path = self.cfg.get_path('logs')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Validation Run @ {pd.Timestamp.now()} ---\n")
                f.write("\n".join(self.logs) + "\n")
        except Exception:
            pass # 日志写入失败不应阻断流程

        if self.validation_passed:
            self.logger.info("✅ 审计通过：数据集满足贝叶斯反演的数学前置条件。")
        else:
            self.logger.error("❌ 审计失败：检测到核心逻辑冲突，请修复上游处理逻辑！")

        return self.validation_passed

# --- 单元测试 (模拟逻辑冲突) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

    # 模拟一个带有‘幽灵观测’和‘排名倒挂’的脏数据
    mock_data = pd.DataFrame({
        'celebrity_name': ['A', 'B', 'DeadGuy'],
        'season': [1, 1, 1],
        'week_num': [2, 2, 2],
        'raw_score': [9.0, 8.0, 5.0],
        'week_avg_score': [9.0, 8.0, 5.0],
        'score_z': [1.0, 0.0, -1.0],
        'tech_rank': [1.0, 2.0, 1.0], # B(8分) 排第2，DeadGuy(5分) 居然排第1？-> 倒挂
        'eliminated_week': [10.0, 10.0, 1.0], # DeadGuy 第1周淘汰，第2周还在？-> 幽灵
        'judge_id': ['CAI', 'LG', 'UNKNOWN'], # -> 映射失败
        'partner_alpha': [1.0, 1.0, float('nan')] # -> 因子缺失
    })

    validator = DataValidator(mock_data)
    result = validator.run_all()

    print(f"\n审计结果: {'通过' if result else '不通过'}")
    # 预期：不通过，触发所有报警