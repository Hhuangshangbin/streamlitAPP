import streamlit as st
import pandas as pd


st.set_page_config(page_title="⑤ OC patients vs. BC patients", page_icon="📈")

st.markdown("# ⑤ OC patients vs. BC patients")
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
    st.image(Image.open("ocbc.png"))
    st.image(Image.open("xxt.png"))
    st.balloons()