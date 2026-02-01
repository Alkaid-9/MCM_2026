# ==============================================================================
# src/etl/feature_factory.py
# Role: Strategic Factor Library (The Alpha Generator v5.5)
# Function: Constructing high-dimensional features for LMM and Bayesian Priors.
# Key Logic: Lagged Expanding Windows, Interaction Terms, and Scaled Momentum.
# Standard: Academic Rigor (Causality Firewall) & Industrial Quant Standards.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class FeatureFactory:
    """
    因子工厂：负责将初级清洗后的信号炼制为具有统计显著性的“黄金特征”。

    [核心规范]:
    1. 无偏性: 所有能力类指标必须滞后于当前观测点 (Lagged)。
    2. 可比性: 数值特征需经过标准化，使回归系数 (Beta) 具备跨维度可比性。
    3. 语义性: 保留原始分类标签，支撑后端的因果路径分析。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = setup_logger("FEATURE_FACTORY")
        self.scaler = StandardScaler()

    def build_celebrity_static_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建明星静态背景因子。
        物理意义：捕捉行业背景带来的“原始社会资本”差异 (Prior Distribution Anchor)。
        """
        self.logger.info("提取明星静态特征与行业偏置 (Static Bias)...")

        f_cfg = self.cfg.get_features_config()
        r_cfg = self.cfg._rules  # 直接访问规则字典

        # 1. 行业语义映射与引力偏置 (Industry Gravity)
        mapping = f_cfg.get('industry_mapping', {})
        bias_map = r_cfg.get('factor_anchors', {}).get('industry_base_bias', {})

        df['industry_group'] = df['celebrity_industry'].map(mapping).fillna('Other')
        # 将行业映射为先验的“得票红利”
        df['industry_prior_bias'] = df['industry_group'].map(bias_map).fillna(0.0)

        # 2. 年龄代际分段 (Age Heterogeneity)
        age_cfg = f_cfg.get('age_segmentation', {})

        # Impute missing ages with median to prevent sample dropping
        median_age = df['celebrity_age_during_season'].median()
        df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(median_age)

        df['age_group'] = pd.cut(
            df['celebrity_age_during_season'],
            bins=age_cfg.get('bins', [0, 25, 40, 60, 100]),
            labels=age_cfg.get('labels', ["GenZ", "Millennial", "GenX", "Senior"])
        )

        # 3. 类别特征压实 (One-Hot for SHAP/XGBoost)
        # 关键：保留原始列 'industry_group' 供 LMM 聚类，生成 dummy 供非线性回归
        dummies = pd.get_dummies(
            df[['industry_group', 'age_group']],
            prefix=['ind', 'age'],
            drop_first=False,
            dtype=int
        )
        df = pd.concat([df, dummies], axis=1)

        return df

    def build_performance_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建表现动力学因子 (High-Frequency Dynamics)。
        物理意义：量化“逆袭曲线”。观众通常对“进步最快”的选手有情感溢价。
        """
        self.logger.info("计算表现动量与加速度因子 (Dynamics Logic)...")

        # 严格时序排序，这是 diff 计算的地基
        df = df.sort_values(['celebrity_name', 'season', 'week_num'])

        # 1. 速度 (Velocity): 当周得分相对于上周的变化
        # 这里的 week_avg_score 已经在 transformers 阶段聚合好了
        df['score_delta'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.diff()).fillna(0.0)

        # 2. 加速度 (Acceleration): 进步速度的变化率
        # 物理意义：识别选手的学习爆发期 (Learning Curve inflection)
        df['score_acceleration'] = df.groupby(['celebrity_name', 'season'])['score_delta'].transform(
            lambda x: x.diff()).fillna(0.0)

        # 3. 长期声望 (Expanding Reputation): 累计平均分
        df['cum_avg_score'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.expanding().mean()
        )

        # 4. 发挥稳定性 (Volatility): 历史得分标准差
        # 物理意义：发挥不稳的选手通常争议较大，MCMC 的后验方差也会更大
        df['score_volatility'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.expanding().std()
        ).fillna(0.0)

        return df

    def build_partner_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        量化舞伴历史红利 (Pro-Partner Alpha) —— [因果防火墙版本]。
        学术价值：解决“名师带高徒”的内生性偏置。
        核心公式: Alpha_T = Median(Global_Placement) / Mean(Partner_Placement_{1:T-1})
        """
        self.logger.info("量化舞伴历史溢价 [Causality Firewall Enabled]...")

        # 1. 提取舞伴历史面板 (去重以获得每季最终成绩)
        # 注意：使用 placement (最终排名) 作为衡量标准
        partner_panel = df[['season', 'ballroom_partner', 'placement']].dropna().drop_duplicates()
        partner_panel = partner_panel.sort_values(['ballroom_partner', 'season'])

        # 2. 计算滞后扩张窗口均值 (No Look-ahead Bias)
        # shift(1) 确保第 N 季的 Alpha 评价完全基于之前的历史
        partner_panel['hist_avg_place'] = (
            partner_panel.groupby('ballroom_partner')['placement']
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        # 3. 冷启动防御 (Rookie Handling)
        # 对于新舞伴，使用所有舞伴的历史中位排名作为无偏先验 (Unbiased Prior)
        global_median_place = df['placement'].median()
        partner_panel['hist_avg_place'] = partner_panel['hist_avg_place'].fillna(global_median_place)

        # 4. 映射为 Alpha 因子 (数值越大能力越强)
        # 物理意义：排名越小越好 -> Alpha = 基准 / 历史平均排名
        partner_panel['partner_alpha'] = global_median_place / (partner_panel['hist_avg_place'] + 1e-9)

        # 5. 原子级 Grand Join
        # 确保 join 键类型一致
        df['season'] = df['season'].astype('Int64')  # Nullable Int
        partner_panel['season'] = partner_panel['season'].astype('Int64')

        df = df.merge(
            partner_panel[['season', 'ballroom_partner', 'partner_alpha']],
            on=['season', 'ballroom_partner'],
            how='left'
        )

        # 缺失情况默认中性权重 (1.0)
        df['partner_alpha'] = df['partner_alpha'].fillna(1.0)

        return df

    def build_contextual_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建环境上下文因子。
        """
        self.logger.info("计算环境信噪比与竞争压力...")

        # 1. 相对排名 (0-1)
        df['relative_rank'] = df.groupby(['season', 'week_num'])['week_avg_score'].rank(pct=True)

        # 2. 存活竞争压力 (剩余选手数量)
        df['n_competitors'] = df.groupby(['season', 'week_num'])['celebrity_name'].transform('count')

        # 3. 信号强度归一化 (基于 transformers 算出的 signal_clarity)
        if 'signal_clarity' in df.columns:
            # 归一化到 [0, 1] 区间
            max_clarity = df['signal_clarity'].max() + 1e-9
            df['signal_strength_norm'] = df['signal_clarity'] / max_clarity
        else:
            df['signal_strength_norm'] = 0.5  # 默认中值

        return df

    def generate_gold_library(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一键炼制黄金因子库。产出物将作为 Stage 3-5 的核心输入。
        """
        # 顺序执行，构建特征金字塔
        df = self.build_celebrity_static_factors(df)
        df = self.build_performance_dynamics(df)
        df = self.build_contextual_factors(df)
        df = self.build_partner_alpha(df)

        # 最终特征清洗：处理由于 Expanding window 产生的极少量的 Inf 值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.logger.info(f" 因子库构建成功。特征总维度: {df.shape[1]}")
        return df


# --- 单元测试 (模拟 O 奖级高强度数据环境) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 创建包含多赛季、多选手的复杂模拟数据
    # 模拟 Derek Hough (大神) 连续参赛
    mock_data = pd.DataFrame({
        'season': [1, 1, 2, 2],
        'week_num': [1, 2, 1, 2],
        'celebrity_name': ['Star_A', 'Star_A', 'Star_B', 'Star_B'],
        'celebrity_industry': ['Singer', 'Singer', 'Athlete', 'Athlete'],
        'celebrity_age_during_season': [25, 25, 30, 30],
        'ballroom_partner': ['Derek', 'Derek', 'Derek', 'Derek'],
        'week_avg_score': [8.0, 9.0, 8.5, 8.5],
        'placement': [1, 1, 5, 5],  # S1 夺冠，S2 第五
        'signal_clarity': [1.0, 1.2, 0.8, 0.9]
    })

    factory = FeatureFactory()
    gold = factory.generate_gold_library(mock_data)

    print("\n--- 关键因子核验 (Causality Firewall) ---")
    # 观察 Star_B (S2) 的 partner_alpha 是否基于 Star_A (S1) 的 placement
    # S1 Derek 拿了第 1 -> S2 Derek 的 Alpha 应该很高 (Median/1)
    print(gold[['season', 'celebrity_name', 'ballroom_partner', 'placement', 'partner_alpha', 'score_delta']])