import streamlit as st
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import seaborn as sns
from statannotations.Annotator import Annotator
from pylab import mpl
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


st.set_page_config(page_title="③ OC patients vs. BC patients", page_icon="📈")

st.markdown("# ③ OC patients vs. BC patients")
st.write(
    """This is the result of the population survival curves of ON patients and BP patients, which shows that the log-rank is statistically significant, but the significant relationship, after using multifactorial cox analysis, is still relevant for comparative studies between ON patients and BP patients.
"""
)

from PIL import Image
st.image(Image.open("f3.png"))
st.markdown("### Download [Test Cases](链接：https://pan.baidu.com/s/1bugXjKdSFU1wdxhx-MGhZw?pwd=2558) here")
st.markdown("#### :blue[Example:]")

data = pd.read_csv('NC_data1224.csv')
st.write(data)


file = st.file_uploader("#### Choose a CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file, encoding="utf")

if st.button('**Submit the data**'):
    if file is None:
        st.error('ERROR: No files uploaded!')
    NC_patient = df
    new_patient = NC_patient[NC_patient.iloc[:, -1] == 'OC']
    berlin_patient = NC_patient[NC_patient.iloc[:, -1] == 'BC']

    # 对 day_death 小于等于 0 的所有行增加 1 天
    new_patient.loc[new_patient['day_death'] <= 0, 'day_death'] += 1
    berlin_patient.loc[berlin_patient['day_death'] <= 0, 'day_death'] += 1

    # 假设 'data' 是包含 'day_death'、'survive' 和 'subphenotype' 列的 DataFrame
    # 假设我们对 'I' 和 'II' 亚型感兴趣
    data_I = new_patient
    data_II = berlin_patient

    # 为亚型 I 拟合 Kaplan-Meier 估计器
    kmf_I = KaplanMeierFitter()
    kmf_I.fit(data_I['day_death'], event_observed=data_I['survive'], label='Subphenotype OC')

    # 为亚型 II 拟合 Kaplan-Meier 估计器
    kmf_II = KaplanMeierFitter()
    kmf_II.fit(data_II['day_death'], event_observed=data_II['survive'], label='Subphenotype BC')

    # 创建绘图
    fig, ax = plt.subplots(figsize=(8, 8))

    kmf_I.plot_survival_function(ax=ax, ci_show=False, color='#CE4257', show_censors=True)
    kmf_II.plot_survival_function(ax=ax, ci_show=False, color='#79A3D9', show_censors=True)

    # 添加网格线
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 在 KM 曲线下方添加风险表
    add_at_risk_counts(kmf_I, kmf_II, ax=ax)

    # 添加标题和标签
    plt.title('KM Survival Curve - 30-day analysis')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival probability')

    # 执行 log-rank 检验计算亚型之间的 p 值
    results = logrank_test(data_I['day_death'], data_II['day_death'], event_observed_A=data_I['survive'],
                           event_observed_B=data_II['survive'])
    p = results.p_value

    # 在图中添加 p 值
    plt.figtext(0.5, 0.5, f'p = {p:.4e}', ha="center", fontsize=12,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    # 显示图形
    plt.tight_layout()

    # 在 Streamlit 中显示图形
    st.pyplot(fig)

    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data = df  # 替换为您的数据文件路径
    # 选择的生物标记物列表，包括白蛋白作为炎症反应的指标
    biomarkers = ['7天内pao2中位数', '7天内spo2中位数', '7天内p/f中位数', '7天内s/f中位数', 'resp_rate', 'icu_stay', '7天内fio2中位数', 'sofa']
    # biomarkers = ['pao2', 'spo2', 'pao2fio2ratio', 'spo2fio2ratio', 'resp_rate', 'icu_stay', 'fio2', 'sofa']
    # 定义箱形图的颜色
    colors = ['#CE4257', '#79A3D9']
    # 创建子图布局
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))  # 根据需要调整大小
    axes = axes.flatten()
    # 绘制每个生物标记物的箱形图
    for i, biomarker in enumerate(biomarkers):
        ax = sns.boxplot(x="OC/BC", y=biomarker, data=data, ax=axes[i], palette=colors, showfliers=False,
                         order=["OC", "BC"])
        ax.set_title(biomarker)
        ax.set_xlabel('')

        # 用于Annotator的类别对应该仅包含类别名称
        pairs = [('OC', 'BC')]
        annotator = Annotator(ax, pairs, data=data, x="OC/BC", y=biomarker)
        annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
        annotator.apply_and_annotate()
    plt.subplots_adjust(bottom=0.1)
    # 添加整体x轴标题
    fig.text(0.5, 0.04, '', ha='center')
    # 调整布局
    plt.tight_layout()
    st.pyplot(fig)
    st.balloons()