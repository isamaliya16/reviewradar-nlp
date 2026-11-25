"""Small helpers shared by the pages."""
from datetime import datetime

import pandas as pd
import streamlit as st

import ui


def add_history(analysis):
    history = st.session_state.setdefault('history', [])
    history.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'title': analysis.title,
                    'review': analysis.text, 'p_not_recommended': round(analysis.p_not, 3),
                    'flagged': analysis.p_not > analysis.threshold, 'risk': analysis.risk})


def need_data(df):
    """Friendly message + False when the CSV is missing."""
    if df is None:
        ui.empty_state('📂', 'The reviews CSV was not found, so this page cannot be shown. '
                             'Put it in the dataset folder (see README) and reload.')
        return False
    return True


def to_csv(table):
    return pd.DataFrame(table).to_csv(index=False).encode('utf-8')
