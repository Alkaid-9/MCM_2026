# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助


# 数据处理
import pandas as pd
import numpy as np

# 机器学习
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

# 可解释性AI (必须会有！画图神器)
import shap

# 因果推断 (备用，用于评估政策/教练效果)
# pip install causalinference
from causalinference import CausalModel

# 绘图
import matplotlib.pyplot as plt
import seaborn as sns
# 设置学术风绘图风格
sns.set_context("paper", font_scale=1.5)
plt.style.use('seaborn-v0_8-whitegrid')