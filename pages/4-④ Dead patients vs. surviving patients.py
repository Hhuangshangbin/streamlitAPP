import streamlit as st
import pandas as pd


st.set_page_config(page_title="④ Dead patients vs. surviving patients", page_icon="📈")

st.markdown("# ④ Dead patients vs. surviving patients")
st.write(
    """To further assess the robustness and usability of subphenotyping, we trained a predictive model for subphenotyping and used it to predict subphenotypic membership in an external validation cohort. Clinical variables that were cluster analyzed using laboratory tests were used as candidate predictors. The trained predictive model (XGBoost classifier) achieved very high performance in predicting each subphenotype.
    The SHAP identified the above five key factors as influencing the model, with respiration rate (resp_rate) being the most influential feature in the model predictions. This suggests that changes in respiration rate have a significant effect on model output, and that higher respiration rates may increase the likelihood of model predictions.
    """
)

from PIL import Image
st.image(Image.open("f3.png"))
st.markdown("### Download [Test Cases](https://pan.baidu.com/s/1bugXjKdSFU1wdxhx-MGhZw?pwd=2558) here")

st.download_button('OR DOWNLOAD HERE',file_name='NC_data1224.csv')
st.markdown("#### :blue[Example:]")

data = pd.read_csv('NC_data1224.csv')
st.write(data)


file = st.file_uploader("#### Choose a CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file, encoding="utf")

if st.button('**Submit the data**'):
    if file is None:
        st.error('ERROR: No files uploaded!')
    st.image(Image.open("myplot.png"))
    st.image(Image.open("d2.png"))
    st.image(Image.open("d3.png"))
    st.balloons()