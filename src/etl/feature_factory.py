# ==============================================================================
# src/etl/feature_factory.py
# Role: Strategic Factor Library (The Alpha Generator)
# Function: Constructing high-dimensional features for causal attribution.
# Standard: Academic Rigor (Causality Firewall) & Industrial Scalability.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from src.etl.config_loader import ConfigLoader

class FeatureFactory:
    """
    因子工厂：
    将聚合后的技术信号转化为具有解释力的统计因子。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("FEATURE_FACTORY")

    def build_celebrity_static_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建明星静态背景因子。
        物理意义：捕捉行业背景（流量底池）与年龄（代际偏好）带来的先验差异。
        """
        self.logger.info("正在提取明星背景特征 (Industry & Age)...")

        f_cfg = self.cfg._config.get('features', {})

        # 1. 行业语义映射
        mapping = f_cfg.get('industry_mapping', {})
        # 即使 mapping 没盖全，也给个 Baseline 兜底
        df['industry_group'] = df['celebrity_industry'].map(mapping).fillna('Baseline')

        # 2. 年龄代际分段 (Generation Analysis)
        age_cfg = f_cfg.get('age_segmentation', {})
        # 处理可能的缺失年龄 (Impute by Median)
        median_age = df['celebrity_age_during_season'].median()
        df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(median_age)

        df['age_group'] = pd.cut(
            df['celebrity_age_during_season'],
            bins=age_cfg.get('bins', [0, 25, 40, 60, 100]),
            labels=age_cfg.get('labels', ["GenZ", "Millennial", "GenX", "Senior"])
        )

        # 3. 生成 One-Hot 变量 (为 Task 3 的线性模型与 SHAP 分析做准备)
        # 注意：这里我们不 drop_first，方便后面做完整的 SHAP 归因图
        df = pd.get_dummies(df, columns=['industry_group', 'age_group'],
                            prefix=['ind', 'age'], drop_first=False)

        return df

    def build_performance_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """在聚合表上计算动量因子"""
        self.logger.info("正在计算表现动力学因子 (Aggregated Level)...")

        # 确保时序，防止 diff() 算错对象
        df = df.sort_values(['celebrity_name', 'season', 'week_num'])

        # 1. 进步动量 (Momentum)
        if 'week_avg_score' not in df.columns:
            raise KeyError("致命错误：输入数据中缺失 'week_avg_score' 列，请检查 transformers.py 聚合逻辑！")

        df['score_delta'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(lambda x: x.diff())

        # 2. 累计表现 (Expanding Mean)
        df['cum_avg_score'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.expanding().mean()
        )

        df['score_delta'] = df['score_delta'].fillna(0)
        return df

    def build_contextual_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建赛场环境变量（环境信噪比）。
        物理意义：同样的 8 分，在高手如云的周次和水平接近的周次，吸粉意义完全不同。
        """
        self.logger.info("正在计算竞争环境因子...")

        # 1. 相对技术排名 (Relative Technical Rank)
        # 将绝对分数转化为当周的百分比排名 [0, 1]
        df['relative_rank'] = df.groupby(['season', 'week_num'])['week_avg_score'].transform(
            lambda x: x.rank(pct=True)
        )

        # 2. 存活压力 (Survival Pressure)
        # 剩余选手越少，单票的边际价值越高
        df['n_competitors'] = df.groupby(['season', 'week_num'])['celebrity_name'].transform('nunique')

        # 3. 环境信号强度 (Signal Strength Norm)
        # 如果当周大家打分非常接近 (StdDev 小)，信号模糊，粉丝投票权重会隐式上升。
        # 注意：此处输入已是聚合表，该指标应基于 transformer 阶段算好的 signal_strength_norm
        # 如果列名不存在，我们在此处做一个简易版的防御
        if 'signal_strength_norm' not in df.columns:
            df['signal_strength_norm'] = df.groupby(['season', 'week_num'])['week_avg_score'].transform('std').fillna(0)
            max_val = df['signal_strength_norm'].max() + 1e-9
            df['signal_strength_norm'] /= max_val

        return df

    def build_partner_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        量化舞伴历史红利 (Pro-Partner Alpha) —— 【工业级稳定版】。
        修复：使用 transform 替代 apply 解决索引不兼容问题。
        """
        self.logger.info("正在计算舞伴历史红利因子 (Lagged Alpha)...")

        # 1. 建立舞伴历年战绩索引
        # 必须先重置索引，确保计算过程中的索引是干净的
        partner_history = df[['season', 'ballroom_partner', 'placement']].dropna().drop_duplicates()
        partner_history = partner_history.sort_values(['ballroom_partner', 'season'])

        # 2. 【核心修复】：使用 transform 确保索引对齐
        # transform 会保留 partner_history 的原始索引，直接解决 TypeError
        # lambda x: x.expanding().mean().shift(1) 计算的是：该舞伴在当前赛季之前的平均排名
        partner_history['hist_avg_place'] = (
            partner_history.groupby('ballroom_partner')['placement']
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        # 3. 处理冷启动 (Rookies)
        # 对于新舞伴，给予全局中位排名作为“无偏先验”
        global_median_place = df['placement'].median()
        partner_history['hist_avg_place'] = partner_history['hist_avg_place'].fillna(global_median_place)

        # 4. 转换 Alpha 值 (排名数字越小，Alpha 应该越高)
        # 物理意义：Alpha = 全局水平 / 舞伴水平
        # 例如：全场中位排第 6，某舞伴平均排第 3，其 Alpha = 6/3 = 2.0 (强力舞伴)
        partner_history['partner_alpha'] = global_median_place / (partner_history['hist_avg_place'] + 1e-9)

        # 5. 回填主表
        # 在 merge 前确保连接键的类型一致，防止潜在的类型合并错误
        df['season'] = df['season'].astype(int)
        partner_history['season'] = partner_history['season'].astype(int)

        df = df.merge(
            partner_history[['season', 'ballroom_partner', 'partner_alpha']],
            on=['season', 'ballroom_partner'],
            how='left'
        )

        # 最后的防御：填充新出现的组合
        df['partner_alpha'] = df['partner_alpha'].fillna(1.0)

        return df

    def generate_gold_library(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        因子库生成主流水线。
        """
        # A. 静态背景注入
        df = self.build_celebrity_static_factors(df)

        # B. 表现动力学计算
        df = self.build_performance_dynamics(df)

        # C. 竞争环境模拟
        df = self.build_contextual_factors(df)

        # D. 舞伴能力量化
        df = self.build_partner_alpha(df)

        # E. 最终清理与审计
        # 首周动量 NaN 填 0，代表平稳启动
        df['score_delta'] = df['score_delta'].fillna(0)

        # 删除中间辅助列 (如果有的话)

        self.logger.info(f"黄金因子库构建完毕。维度: {df.shape}")
        return df

if __name__ == "__main__":
    # 模拟聚合后的数据进行测试
    logging.basicConfig(level=logging.INFO)
    mock_agg = pd.DataFrame({
        'season': [1, 1, 1, 2, 2],
        'week_num': [1, 2, 3, 1, 2],
        'celebrity_name': ['A', 'A', 'A', 'B', 'B'],
        'celebrity_industry': ['Singer', 'Singer', 'Singer', 'Athlete', 'Athlete'],
        'celebrity_age_during_season': [25, 25, 25, 34, 34],
        'ballroom_partner': ['P1', 'P1', 'P1', 'P1', 'P1'],
        'week_avg_score': [7.0, 8.0, 9.0, 6.0, 6.5],
        'placement': [1, 1, 1, 5, 5]
    })

    factory = FeatureFactory()
    gold = factory.generate_gold_library(mock_agg)
    print(gold[['celebrity_name', 'season', 'week_num', 'score_delta', 'partner_alpha']])