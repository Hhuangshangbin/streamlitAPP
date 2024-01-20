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


    ards_patient_to_classify = df
    # ards_patient_to_classify=ards_patient_to_classify.drop(columns=["hadm_id","albumin"])
    class_col = [
        'heart_rate',  # 心率
        'sbp',  # 收缩压
        'dbp',  # 舒张压
        'mbp',  # 平均动脉压
        'sbp_ni',  # 非侵入性收缩压（如果适用）
        'dbp_ni',  # 非侵入性舒张压（如果适用）
        'mbp_ni',  # 非侵入性平均动脉压（如果适用）
        'resp_rate',  # 呼吸频率
        '7天内spo2中位数',  # 血氧饱和度
        '7天内pao2中位数',  # 动脉氧分压
        'pco2',  # 动脉二氧化碳分压
        '7天内p/f中位数',  # 动脉氧分压与吸入氧浓度比值
        '7天内s/f中位数',  # 血氧饱和度与吸入氧浓度比值
        '7天内fio2中位数',  # 吸入氧浓度
        'aado2_calc',  # 计算AaDO2
        'ph',  # pH值
        'baseexcess',  # 碱过量
        'bicarbonate',  # 碳酸氢盐
        'totalco2',  # 总二氧化碳
        'chloride',  # 氯化物
        'calcium',  # 钙
        'temperature',  # 体温
        'lactate',  # 血乳酸
        'potassium',  # 钾
        'sodium',  # 钠
        'glucose',  # 葡萄糖
        'aniongap',  # 阴离子间隙
        'bun',  # 尿素氮
        'creatinine',  # 肌酐
        'respiration',  # 呼吸系统评分（如果适用）
        'coagulation',  # 凝血系统评分（如果适用）
        'liver',  # 肝脏评分（如果适用）
        'cardiovascular',  # 心血管系统评分（如果适用）
        'cns',  # 中枢神经系统评分（如果适用）
        'renal'  # 肾脏评分（如果适用）
    ]
    C = 2
    # X = dev_study_data_combined.drop(columns=['ssid', 'label']).values
    X = ards_patient_to_classify[class_col].values
    y = ards_patient_to_classify['survive'].values

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_tprs = []
    cv_aucs = []
    cv_mean_fpr = np.linspace(0, 1, 100)
    j = 1
    for train, test in cv.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        classifier = XGBClassifier(booster='gbtree', objective='binary:logistic', random_state=1)
        classifier.fit(X_train, y_train)

        y_score = classifier.predict_proba(X_test)[:, 1]  # 获取正类的预测概率

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        cv_interp_tpr = np.interp(cv_mean_fpr, fpr, tpr)
        cv_interp_tpr[0] = 0.0
        cv_tprs.append(cv_interp_tpr)
        cv_aucs.append(roc_auc)

        print('Iteraction %d finished...' % j)
        j += 1
    # 计算平均的 TPR 和 AUC
    mean_tpr = np.mean(cv_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(cv_aucs)
    std_auc = np.std(cv_aucs)
    # 绘制 ROC 曲线
    fig = plt.figure(figsize=(6, 6))
    plt.plot(cv_mean_fpr, mean_tpr, color='blue', lw=2,
             label='ROC curve (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
    plt.fill_between(cv_mean_fpr, mean_tpr - np.std(cv_tprs, axis=0), mean_tpr + np.std(cv_tprs, axis=0), color='blue',
                     alpha=.2)
    plt.plot([0, 1], [0, 1], linestyle=':', color='grey', lw=1)
    plt.legend(fontsize=10)
    plt.grid(c='#DCDCDC')
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) curve', fontsize=13, y=1.01)
    st.pyplot(fig)

    final_classifier = XGBClassifier(booster='gbtree', objective='binary:logistic', random_state=1)
    final_classifier.fit(X, y)
    explainer = shap.TreeExplainer(final_classifier)
    shap_values = explainer.shap_values(X)

    shap_sum = (np.abs(shap_values).mean(axis=0))
    importance_df = pd.DataFrame([class_col, shap_sum]).T
    importance_df.columns = ['Feature', 'SHAP Importance']
    importance_df.sort_values('SHAP Importance', ascending=False, inplace=True)

    # Get the top features with the highest mean absolute SHAP values
    top_features = importance_df.head()

    # 绘制 SHAP 总结图
    shap.summary_plot(shap_values, X, feature_names=class_col, show=False, plot_size=(6, 8))
    cmap = plt.get_cmap('RdYlBu_r')  # hot, RdBu_r, RdYlBu_r, copper, plasma
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, 'set_cmap'):
                fcc.set_cmap(cmap)
    plt.tight_layout()
    plt.xlim(-4, 4)
    plt.xlabel('SHAP value')
    st.pyplot(fig)
    st.write("### Top features")
    st.write(top_features)
    st.balloons()