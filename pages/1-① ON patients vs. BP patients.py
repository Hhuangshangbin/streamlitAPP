import streamlit as st
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl

st.set_page_config(page_title="① ON patients vs. BP patients", page_icon="📈")

st.markdown("# ① ON patients vs. BP patients")
st.write(
    """This is the result of the population survival curves of ON patients and BP patients, which shows that the log-rank is statistically significant, but the significant relationship, after using multifactorial cox analysis, is still relevant for comparative studies between ON patients and BP patients.
"""
)
NC_patient=pd.read_csv("ONBP.csv",encoding="gb18030")
new_patient = NC_patient[NC_patient.iloc[:, -1] == 'ON']
from PIL import Image
st.image(Image.open("fig10.png"))
st.markdown("### Download [Test Cases](https://pan.baidu.com/s/1bugXjKdSFU1wdxhx-MGhZw?pwd=2558) here")
st.download_button('OR DOWNLOAD HERE',file_name='ONBP.csv')
st.markdown("#### :blue[Example:]")

data = pd.read_csv('ONBP.csv',encoding="gb18030")
st.write(data)

file = st.file_uploader("#### Choose a CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file,encoding="gb18030")

if st.button('**Submit the data**'):
    if file is None:
        st.error('ERROR: No files uploaded!')
    NC_patient = df
    new_patient = NC_patient[NC_patient.iloc[:, -1] == 'ON']
    berlin_patient = NC_patient[NC_patient.iloc[:, -1] == 'BP']

    # 对 day_death 小于等于 0 的所有行增加 1 天
    new_patient.loc[new_patient['day_death'] <= 0, 'day_death'] = new_patient.loc[
                                                                      new_patient['day_death'] <= 0, 'day_death'] + 1
    berlin_patient.loc[berlin_patient['day_death'] <= 0, 'day_death'] = berlin_patient.loc[berlin_patient[
                                                                                               'day_death'] <= 0, 'day_death'] + 1
    # Assuming 'data' is your DataFrame with the 'day_death', 'survive', and 'subphenotype' columns
    # Let's say we are interested in subphenotypes 'I' and 'II'
    data_I = new_patient
    data_II = berlin_patient
    # Fit the Kaplan-Meier estimator for Subphenotype I
    kmf_I = KaplanMeierFitter()
    kmf_I.fit(data_I['day_death'], event_observed=data_I['survive'], label='Subphenotype ON')
    # Fit the Kaplan-Meier estimator for Subphenotype II
    kmf_II = KaplanMeierFitter()
    kmf_II.fit(data_II['day_death'], event_observed=data_II['survive'], label='Subphenotype BP')
    # Create the plots
    fig, ax = plt.subplots(figsize=(8, 8))
    kmf_I.plot_survival_function(ax=ax, ci_show=False, color='#CE4257', show_censors=True)
    kmf_II.plot_survival_function(ax=ax, ci_show=False, color='#79A3D9', show_censors=True)
    # Add grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Add the at-risk table below the KM curves
    add_at_risk_counts(kmf_I, kmf_II, ax=ax)
    # Add title and labels
    plt.title('KM Survival Curve - 30-day analysis')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival probability')
    # Perform log-rank test to calculate p-value between subphenotypes
    results = logrank_test(data_I['day_death'], data_II['day_death'], event_observed_A=data_I['survive'],
                           event_observed_B=data_II['survive'])
    p = results.p_value
    # Add p-value to the plot
    plt.figtext(0.5, 0.5, f'p = {p:.4e}', ha="center", fontsize=12,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    # Show the plot
    plt.tight_layout()
    st.pyplot(fig)
    st.balloons()
