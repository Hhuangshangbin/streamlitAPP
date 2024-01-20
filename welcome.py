import streamlit as st

st.set_page_config(
    page_title="welcome page",
    page_icon="👋",
)

st.write("# New Acute Respiratory Distress Syndrome (ARDS) Criteria Impact on patients with ARDS in the Berlin criteria! 👋")

st.sidebar.success("""Welcome Page: This page is a preview of our work, and you can get a preliminary look at this study.☝By selecting different tabs, different functions are reproduced!
KM survival curves were analyzed for each patient category and each severity classification, patient population categories, and comparative analyses were done.""")
st.sidebar.success('All rights reserved ©: Huang Shangbin, Li Guohao, Zha Yulong, Cai Yuanhang. !!Reproduction is prohibited without permission!!')
st.markdown("""
    Impact of the New Acute Respiratory Distress Syndrome (ARDS) Criteria in Patients with Berlin Criteria ARDS is a 
    study based on MIMIC-IV v2.2 cohort data. A brief description of this study follows.
    **👈 Select a demo from the sidebar** to see what this study can do!"""
)
st.markdown(
    """
<style>
    .stChatMessage {
        text-align: justify;
    }
</style>
""",
    unsafe_allow_html=True,
)
#背景介绍
with st.chat_message("user"):
    st.markdown("""
        ## Background of the study
        :red[**Acute respiratory distress syndrome (ARDS)**] is a clinical syndrome characterized by acute hypoxic 
        respiratory failure caused by inflammation of the lungs.Over the past 55 years, ARDS criteria have focused 
        on the radiologic manifestations of the syndrome and the severity of the oxygenation defect :red[(PaO2/FiO2)], 
        which reflects the original description and conceptual understanding of ARDS. The Berlin definition, on the other 
        hand, requires that patients receive a :red[positive end-expiratory pressure (PEEP)] of at least 5 cmH2O or they 
        cannot be considered to have ARDS.:red[The European Society of Intensive Care Medicine (ESICM)] updated the ARDS 
        criteria in July 2023. The new criteria use an arterial oxygenation index of PaO2/FiO2 ≤ 300 mmHg or SpO2/FiO2 ≤ 
        315 (SpO2 ≤ 97%) to identify hypoxemia, and add ultrasound as an imaging method to assess lung status. Patients using 
        high-flow nasal cannula oxygen therapy (HFNO) with HFNC ≥30 L/min were included.
        """)
#展示一个示意图
from PIL import Image
img = Image.open("fig1.png")
st.image(img,caption="Acute respiratory distress syndrome ARDS")

st.markdown(
    """
<style>
    .stChatMessage {
        text-align: justify;
    }
</style>
""",
    unsafe_allow_html=True,
)
#背景介绍
with st.chat_message("user"):
    st.markdown("""
        ## Research Significance
        The impact of the new ARDS criteria on the treatment of patients with ARDS is currently unknown; therefore, our goal was to determine the effect of the new ARDS criteria on patients meeting the Berlin ARDS criteria through a multicenter cohort study and performance of the new ARDS criteria. And to study the heterogeneity of patients in the new criteria to provide guidance on the treatment of patients with the new criteria.
        """)

with st.chat_message("user"):
    st.markdown("""
        ## Research Status
        With the updating of global diagnostic criteria for ARDS (Acute Respiratory Distress Syndrome), this study delved into the differences between the old and new diagnostic criteria in patient screening and subgroup categorization with the help of the latest version of MIMIC-IV v2.2 database. Subgroups of patients screened based on the new oxygenation index were also categorized. The new definition aims to optimize the diagnosis and treatment of ARDS by introducing improved oxygenation indices, :red[**such as Pa02/Fi02 and Sp02/Fi02, and their associated disease severity classifications.**]
        """)

st.write("# ☑Presentation of research findings!")
st.markdown("""
**Result1:** The new definition :red[enables earlier diagnosis] of ARDS patients compared to the Berlin definition.""")
st.markdown("""
**Result2:** Patients screened according to the Berlin definition had a greater proportion of mild patients in terms of severity, whereas patients with non-invasive ventilation (NA) and patients with hypoxemia (NC) according to the new definition had a greater proportion of moderate patients.
""")
st.markdown("""
**Result3:** There was a significant survival difference between patients with hypoxemia (NC) who could not be screened by the Berlin definition (OC) and patients with the Berlin definition (BC), with patients with OC having a lower risk of death than patients with BC, i.e., the new definition will include patients with a lower mortality rate compared to the Berlin definition.
""")
st.markdown("""
**Result4:** :red[The prognostic impact of non-invasive ventilation (NIV/CPAP, HFNC)] on patients is positive feedback, and patients with ARDS are in line with the recommended treatment modalities of the new guidelines in terms of therapeutic ventilation strategies.
""")
st.markdown("""
**Result5:** There was a strong significant relationship between patients who died in patients screened according to hypoxemia (NC) in terms of respiratory compared to the survival group.
""")
st.markdown("""
**Result6:** Easily accessible metrics such as respiratory rate (Resp_Rate) based on urea (Bun) can aid in ARDS diagnosis of ARDS high-risk populations in resource-limited areas.
"""
)

st.image(Image.open("fig4.png"))
st.markdown("""
For the survival and death groups, we plotted chordal plots to observe the direct relationship of indicators to subphenotypes and the associated contribution."""
)

st.image(Image.open("fig5.png"),caption='The relationship between different clinical phenotypes and various types of laboratory test results, especially the relative importance or frequency of each test result across phenotypes. Reveal that specific phenotypes exhibit significant differences in laboratory tests')
st.markdown("""
To further assess the robustness and usability of subphenotyping, we trained a predictive model for subphenotyping and used it to predict subphenotypic membership in an external validation cohort. Clinical variables that were cluster analyzed using laboratory tests were used as candidate predictors. The trained predictive model (XGBoost classifier) achieved very high performance in predicting each subphenotype.
""")
st.image(Image.open("fig7.png"))
st.markdown("""
Based on this result, we selected several easily accessible physiological indicators as our prediction model parameters, :red[namely: 'sbp_ni', 'resp_rate', 'spo2', 'temperature', 'potassium', 'sodium', 'glucose', 'bun', 'creatinine']. The seven metrics were used to predict the subtypes.The ROC curve is plotted below:
""")

st.markdown(
    """
<style>
    .stChatMessage {
        text-align: justify;
    }
</style>
""",
    unsafe_allow_html=True,
)
#背景介绍
with st.chat_message("user"):
    st.markdown("""
        As can be seen from the left panel, :red[**the subtype prediction achieved a high accuracy (AUC=0.83±0.02).**] This suggests that the use of these metrics, which can be accessed in resource-limited areas, can also be a better aid in determining subdivided phenotypes of ARDS.
        The following five key factors were derived from SHAP as influencing the model, with respiration rate (resp_rate) being the most influential feature in model prediction. This suggests that changes in respiration rate have a significant effect on model output, and that higher respiration rates may increase the likelihood of model predictions.
        """)
st.image(Image.open("fig9.png"))

st.write("# 📈Prospects for follow-up")

st.markdown("""
In the next study, we will conduct a multi-database inter-validation study to validate our work by performing the same work in the EICU database with the AUMC database as we did with the MIMIC-IV database, and hope to dig out some equally easy-to-access indicators of lower healthcare costs based on the NC patients (Class C in resource-limited areas) to assist physicians to provide some help in the early screening of ARDS patients.""")
st.image(Image.open("fig8.png"))
st.write("##### ASRL= 1.34* log10([bun]) + 2.87* log10([aniongap])-1.46* log10([bicarbonate])")
st.markdown(
    """
<style>
    .stChatMessage {
        text-align: justify;
    }
</style>
""",
    unsafe_allow_html=True,
)
#背景介绍
with st.chat_message("user"):
    st.markdown("""
The previously selected metrics were combined to predict the subphenotypes using a logistic regression model to arrive at the final best performing combination of 'bun', 'aniongap', 'bicarbonate'. And based on the logistic regression model a simple score based on this combination called ASRL score (ARDS Subphenotype In Resource-limited Areas ) was derived which identifies the subphenotype to which each patient belongs.ASRL score ≥ 4.32 indicates that the patient belongs to the subphenotype II.Positive predictive value is 0.83, negative predictive value is 0.79. The positive predictive value was 0.83 and the negative predictive value was 0.79.
        """)

