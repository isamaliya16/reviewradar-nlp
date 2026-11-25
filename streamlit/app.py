"""ReviewRadar - entry point.  Run:  streamlit run app.py"""
import streamlit as st

from config import DATASET_URL, RISK_BANDS, THRESHOLD_DEFAULT, THRESHOLD_RANGE
from services import system_status
from ui import apply_theme, chips
from views import about, analyze, batch, history, insights, model_lab, words

st.set_page_config(page_title='ReviewRadar', page_icon='📡', layout='wide')

pages = [
    st.Page(analyze.render, title='Analyze', icon=':material/radar:', url_path='analyze', default=True),
    st.Page(batch.render, title='Batch', icon=':material/table_rows:', url_path='batch'),
    st.Page(insights.render, title='Insights', icon=':material/insights:', url_path='insights'),
    st.Page(words.render, title='Word explorer', icon=':material/hub:', url_path='words'),
    st.Page(model_lab.render, title='Model lab', icon=':material/science:', url_path='model-lab'),
    st.Page(about.render, title='How it works', icon=':material/school:', url_path='how-it-works'),
    st.Page(history.render, title='History', icon=':material/history:', url_path='history'),
]
apply_theme()
navigation = st.navigation(pages)

with st.sidebar:
    st.caption('Reads fashion reviews and flags customers who would not recommend the product.')
    st.slider('Decision threshold', *THRESHOLD_RANGE[:2], THRESHOLD_DEFAULT, THRESHOLD_RANGE[2], key='threshold',
              help=f'A review is sent to the quality team when its chance of "not recommended" is above this value. '
                   f'Risk levels: low below {RISK_BANDS[0]}, high from {RISK_BANDS[1]}.')
    st.caption('System status')
    chips(system_status())
    st.markdown(f'[Dataset on Kaggle]({DATASET_URL})')

navigation.run()
