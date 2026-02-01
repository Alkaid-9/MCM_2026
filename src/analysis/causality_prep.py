# ==============================================================================
# src/analysis/causality_prep.py
# Role: Pre-processing & Feature Alignment for Causal Inference (Task 3)
# Function: Recovering missing demographics (Gender) & constructing interaction terms.
# Academic Goal: Minimizing Omitted Variable Bias (OVB) for LMM/SHAP.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.etl.config_loader import ConfigLoader


class CausalityPreprocessor:
    """
    因果推断预处理器：
    负责将 Platinum 层的后验数据转化为适合回归分析的“高信噪比”矩阵。
    """

    def __init__(self, df_platinum: pd.DataFrame):
        self.logger = logging.getLogger("CAUSALITY_PREP")
        self.df = df_platinum.copy()
        self.cfg = ConfigLoader()

        # 定义专业舞伴性别字典 (用于反推明星性别)
        # 物理直觉：DWTS 绝大多数组合是异性搭档
        self.PRO_DANCER_GENDER_MAP = {
            # --- Male Pros (Partner -> Female Star) ---
            'Derek Hough': 'M', 'Maksim Chmerkovskiy': 'M', 'Mark Ballas': 'M',
            'Valentin Chmerkovskiy': 'M', 'Tony Dovolani': 'M', 'Gleb Savchenko': 'M',
            'Artem Chigvintsev': 'M', 'Sasha Farber': 'M', 'Pasha Pashkov': 'M',
            'Brandon Armstrong': 'M', 'Keo Motsepe': 'M', 'Alan Bersten': 'M',
            'Tristan MacManus': 'M', 'Louis van Amstel': 'M', 'Jonathan Roberts': 'M',
            'Alec Mazo': 'M', 'Corkie Ballas': 'M', 'Dmitry Chaplin': 'M',
            'Ezra Sosa': 'M',

            # --- Female Pros (Partner -> Male Star) ---
            'Julianne Hough': 'F', 'Cheryl Burke': 'F', 'Karina Smirnoff': 'F',
            'Kym Johnson': 'F', 'Edyta Sliwinska': 'F', 'Peta Murgatroyd': 'F',
            'Sharna Burgess': 'F', 'Witney Carson': 'F', 'Lindsay Arnold': 'F',
            'Jenna Johnson': 'F', 'Emma Slater': 'F', 'Britt Stewart': 'F',
            'Daniella Karagach': 'F', 'Rylee Arnold': 'F', 'Chelsie Hightower': 'F',
            'Anna Trebunskaya': 'F', 'Lacey Schwimmer': 'F', 'Allison Holker': 'F',
            'Koko Iwasaki': 'F'
        }

        # 特殊情况覆盖 (同性搭档)
        self.SAME_SEX_EXCEPTIONS = {
            'JoJo Siwa': 'F',  # Partner: Jenna Johnson (F)
            'Shangela': 'M',  # Drag Queen, usually presented as M/Fluid in data context
            'Chandler Kinney': 'F'  # Just to be safe
        }

    def _infer_celebrity_gender(self):
        """
        [数据考古学]：利用舞伴性别反推明星性别。
        """
        self.logger.info("执行性别反推算法 (Reverse Gender Inference)...")

        def get_gender(row):
            # 1. 优先检查特例名单
            if row['celebrity_name'] in self.SAME_SEX_EXCEPTIONS:
                return self.SAME_SEX_EXCEPTIONS[row['celebrity_name']]

            # 2. 查舞伴字典
            partner = row['ballroom_partner']
            pro_gender = self.PRO_DANCER_GENDER_MAP.get(partner, 'Unknown')

            if pro_gender == 'M':
                return 'F'  # 男舞伴 -> 女明星
            elif pro_gender == 'F':
                return 'M'  # 女舞伴 -> 男明星
            else:
                return 'Unknown'

        self.df['gender'] = self.df.apply(get_gender, axis=1)

        # 统计缺失率
        unknown_count = (self.df['gender'] == 'Unknown').sum()
        if unknown_count > 0:
            self.logger.warning(f"仍有 {unknown_count} 位明星性别无法推断，将标记为 'Unknown'。")
            # 记录未知的舞伴以便后续补充字典
            unknown_partners = self.df[self.df['gender'] == 'Unknown']['ballroom_partner'].unique()
            self.logger.debug(f"未知性别的舞伴列表: {unknown_partners}")

    def _engineer_interactions(self):
        """
        [特征工程]：构造交互项。
        物理意义：探究“男明星+女舞伴”是否存在特定的投票溢价 (The 'Showmance' Effect)。
        """
        self.logger.info("构造高阶交互特征 (Interaction Terms)...")

        # 1. 性别 x 舞伴红利
        # 假设：Partner Alpha 对异性明星的加成更强
        self.df['is_male_star'] = (self.df['gender'] == 'M').astype(int)
        self.df['interaction_gender_alpha'] = self.df['is_male_star'] * self.df['partner_alpha']

        # 2. 年龄 x 行业
        # 假设：运动员(Athlete)吃青春饭(负相关)，演员(Actor)越老越妖(正相关或U型)
        # 这里只做准备，具体非线性关系交给 SHAP 分析
        pass

    def _normalize_features(self):
        """
        标准化数值特征，确保回归系数 (Beta) 可比。
        """
        scaler = StandardScaler()
        num_cols = ['celebrity_age_during_season', 'partner_alpha', 'score_delta']

        # 填充缺失值 (鲁棒性处理)
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                # 生成 Z-score 列，保留原始列用于解释
                self.df[f'z_{col}'] = scaler.fit_transform(self.df[[col]])

    def process(self) -> pd.DataFrame:
        """
        执行完整处理流水线。
        """
        # 1. 过滤无效样本 (无反演结果的)
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['est_fan_vote_mu']).copy()
        if len(self.df) < initial_len:
            self.logger.info(f"过滤了 {initial_len - len(self.df)} 条无反演结果的样本。")

        # 2. 补全性别
        self._infer_celebrity_gender()

        # 3. 构造交互项
        self._engineer_interactions()

        # 4. 标准化
        self._normalize_features()

        # 5. 独热编码行业 (One-Hot Encoding)
        # 这一步通常在 LMM 内部做，但在 prep 阶段做显式转换方便检查
        if 'industry_group' in self.df.columns:
            dummies = pd.get_dummies(self.df['industry_group'], prefix='ind', drop_first=True)
            self.df = pd.concat([self.df, dummies], axis=1)

        self.logger.info(f"因果推断数据集准备就绪。特征维度: {self.df.shape[1]}")
        return self.df


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Mock Data
    data = {
        'celebrity_name': ['Star A', 'Star B', 'JoJo Siwa'],
        'ballroom_partner': ['Derek Hough', 'Witney Carson', 'Jenna Johnson'],
        'industry_group': ['Actor', 'Athlete', 'Music'],
        'celebrity_age_during_season': [25, 30, 18],
        'partner_alpha': [1.5, 1.2, 1.3],
        'score_delta': [0.5, -0.1, 0.2],
        'est_fan_vote_mu': [0.1, 0.2, 0.3]  # 模拟反演结果
    }
    df_mock = pd.DataFrame(data)

    prep = CausalityPreprocessor(df_mock)
    df_clean = prep.process()

    print("\n--- Processed Data Sample ---")
    print(df_clean[['celebrity_name', 'gender', 'is_male_star', 'interaction_gender_alpha']].head())