import streamlit as st
import pandas as pd


st.set_page_config(page_title="⑤ OC patients vs. BC patients", page_icon="📈")

st.markdown("# ⑤ OC patients vs. BC patients")
st.write(
    """Based on the difference in survival between NC and BP patients, we performed an intra-patient comparison in category C i.e. (OC patients and BC patients), and the results of the survival curves are shown on the left, with a significant difference in survival, with OC patients surviving better than BC patients; it is clear that the poorer survival of NC patients than BP patients is a result of the over-representation of BC patients in the population.
Cox analysis showed that patients with severity 3 had a significantly higher risk of death compared to patients with severity 1, whereas patients with OC had a lower risk compared to patients with BC and those using noninvasive ventilation.
"""
)

from PIL import Image
st.image(Image.open("f3.png"))
st.markdown("### Download [Test Cases](https://pan.baidu.com/s/1bugXjKdSFU1wdxhx-MGhZw?pwd=2558) here")
st.markdown("#### :blue[Example:]")
st.download_button('OR DOWNLOAD HERE', 'NC_data1224.csv',
                   file_name=None, mime=None,
                   key=None, help=None,
                   on_click=None, args=None,
                   kwargs=None)

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