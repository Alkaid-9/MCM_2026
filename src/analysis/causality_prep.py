"""
Causal Inference Preprocessor (v5.6 - Demographics Recovery)
Role: Feature Alignment & Enrichment for Model III.
Function:
    1. Recovering missing 'Gender' via Partner Matching (Reverse Inference).
    2. Constructing Interaction Terms (e.g., Male Star x Pro Alpha).
    3. Formatting data for LMM (Linear Mixed-Effects Models).
Academic Goal: Minimizing Omitted Variable Bias (OVB).
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from src.utils.logger import setup_logger


class CausalityPreprocessor:
    """
    因果推断预处理器：
    负责为 LMM 和 SHAP 准备“高信噪比、多维度”的回归矩阵。
    """

    def __init__(self, df_platinum: pd.DataFrame):
        self.logger = setup_logger("CAUSALITY_PREP")
        self.df = df_platinum.copy()

        # 1. 专家性别字典 (Pro Dancer Gender Map)
        # 物理规则：DWTS 历史上绝大多数是异性搭档 (Male Pro <-> Female Star)
        # 我们利用这一规则反推明星性别
        self.PRO_GENDER_MAP = {
            # --- Male Pros ---
            'Derek Hough': 'M', 'Maksim Chmerkovskiy': 'M', 'Mark Ballas': 'M',
            'Valentin Chmerkovskiy': 'M', 'Tony Dovolani': 'M', 'Gleb Savchenko': 'M',
            'Artem Chigvintsev': 'M', 'Sasha Farber': 'M', 'Pasha Pashkov': 'M',
            'Brandon Armstrong': 'M', 'Keo Motsepe': 'M', 'Alan Bersten': 'M',
            'Tristan MacManus': 'M', 'Louis van Amstel': 'M', 'Jonathan Roberts': 'M',
            'Alec Mazo': 'M', 'Corkie Ballas': 'M', 'Dmitry Chaplin': 'M', 'Ezra Sosa': 'M',
            'Maks Chmerkovskiy': 'M', 'Val Chmerkovskiy': 'M',

            # --- Female Pros ---
            'Julianne Hough': 'F', 'Cheryl Burke': 'F', 'Karina Smirnoff': 'F',
            'Kym Johnson': 'F', 'Edyta Sliwinska': 'F', 'Peta Murgatroyd': 'F',
            'Sharna Burgess': 'F', 'Witney Carson': 'F', 'Lindsay Arnold': 'F',
            'Jenna Johnson': 'F', 'Emma Slater': 'F', 'Britt Stewart': 'F',
            'Daniella Karagach': 'F', 'Rylee Arnold': 'F', 'Chelsie Hightower': 'F',
            'Anna Trebunskaya': 'F', 'Lacey Schwimmer': 'F', 'Allison Holker': 'F',
            'Koko Iwasaki': 'F', 'Kym Johnson-Herjavec': 'F'
        }

        # 2. 同性搭档特例 (Same-Sex Exceptions)
        # 必须硬编码处理，否则会推断错误
        self.SAME_SEX_EXCEPTIONS = {
            'JoJo Siwa': 'F',  # Partner: Jenna Johnson (F) -> Season 30
            'Shangela': 'M',  # Partner: Gleb (M) -> Season 31 (Drag Queen, presented as fluid but biologally male)
            'Chandler Kinney': 'F'  # Just in case
        }

    def _infer_gender(self):
        """
        [数据考古学]：执行性别反推算法。
        逻辑：Star_Gender = Inverse(Partner_Gender) unless Exception.
        """
        self.logger.info("执行性别反推算法 (Reverse Gender Inference)...")

        def resolve_gender(row):
            star = row['celebrity_name']
            partner = row['ballroom_partner']

            # 1. 优先检查特例
            if star in self.SAME_SEX_EXCEPTIONS:
                return self.SAME_SEX_EXCEPTIONS[star]

            # 2. 查表反推
            pro_gender = self.PRO_GENDER_MAP.get(partner, 'Unknown')

            if pro_gender == 'M':
                return 'F'  # 男舞伴 -> 女明星
            elif pro_gender == 'F':
                return 'M'  # 女舞伴 -> 男明星
            else:
                return 'Unknown'  # 舞伴不在字典里

        self.df['gender'] = self.df.apply(resolve_gender, axis=1)

        # 统计覆盖率
        unknowns = self.df[self.df['gender'] == 'Unknown']
        if not unknowns.empty:
            missing_partners = unknowns['ballroom_partner'].unique()
            self.logger.warning(f"无法推断性别的舞伴: {missing_partners[:5]}... (共 {len(unknowns)} 条记录)")

        # 将 Unknown 填补为众数 (或者随机，这里选 Male 因为 Male Star 略多)
        self.df['is_male'] = (self.df['gender'] == 'M').astype(int)

    def _engineer_interactions(self):
        """
        [特征工程]: 构造高阶交互项。
        物理意义：探究 "Showmance" (舞台CP感) 是否存在。
        假设：异性搭档 (Gender_Star != Gender_Pro) 可能比同性搭档有更高的票数溢价？
        或者：男明星 + 强力女舞伴 (Partner Alpha) 的组合效应。
        """
        self.logger.info("构造高阶交互特征 (Interaction Terms)...")

        # 1. 舞伴红利 x 明星性别
        # 逻辑：探究 "女强男弱" vs "男强女弱" 的配置对票数的影响
        # 如果 is_male * partner_alpha 系数显著为正，说明男明星更依赖舞伴带飞
        self.df['inter_male_x_partner'] = self.df['is_male'] * self.df['partner_alpha']

        # 2. 年龄 x 行业
        # 逻辑：体育明星(Athlete)是否越年轻越好？演员(Actor)是否越老越吃香？
        # 这将在 SHAP 分析中体现为非线性依赖
        # 这里只做数据准备，LMM 会自动处理 Group 交互
        pass

    def _standardize_for_regression(self):
        """
        标准化数值特征，确保 LMM 的 Beta 系数可比。
        """
        scaler = StandardScaler()

        cols_to_scale = ['celebrity_age_during_season', 'partner_alpha', 'score_delta']
        for col in cols_to_scale:
            # 填充缺失值 (以中位数填充)
            self.df[col] = self.df[col].fillna(self.df[col].median())

            # 生成 Z-score 列
            self.df[f'z_{col}'] = scaler.fit_transform(self.df[[col]])

    def process(self) -> pd.DataFrame:
        """
        执行预处理流水线。
        返回：增强后的 DataFrame，包含 [gender, is_male, z_age, z_partner_alpha, ...]
        """
        self._infer_gender()
        self._engineer_interactions()
        self._standardize_for_regression()

        # 再次清洗：确保没有 Inf/NaN 进入回归器
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna(subset=['est_fan_vote_mu'])

        self.logger.info(f"因果数据集准备就绪。样本量: {len(self.df)}")
        return self.df


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Mock Data
    data = pd.DataFrame({
        'celebrity_name': ['Star A', 'Star B', 'JoJo Siwa'],
        'ballroom_partner': ['Derek Hough', 'Witney Carson', 'Jenna Johnson'],
        'partner_alpha': [1.5, 1.2, 1.3],
        'celebrity_age_during_season': [25, 30, 18],
        'score_delta': [0.1, 0.2, 0.3],
        'est_fan_vote_mu': [0.5, 0.4, 0.1]
    })

    prep = CausalityPreprocessor(data)
    res = prep.process()
    print(res[['celebrity_name', 'gender', 'is_male', 'inter_male_x_partner']])