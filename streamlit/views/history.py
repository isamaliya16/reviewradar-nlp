"""Page 7: everything analysed in this session."""
import pandas as pd
import streamlit as st

import ui
from views.common import to_csv


def render():
    ui.hero('History', 'Your scan log',
            'Every review you analysed in this browser session. It is not stored on the server.')
    history = st.session_state.get('history', [])
    if not history:
        ui.empty_state('🧾', 'Nothing here yet. Analyse a review and it will appear in this log.')
        return
    table = pd.DataFrame(history)
    flagged = int(table['flagged'].sum())
    ui.kpi_cards([('Reviews analysed', len(table), 'in this session', 'teal', ''),
                  ('Sent to quality team', flagged, 'flagged at the time', 'bad', ''),
                  ('Average risk', table['p_not_recommended'].mean() * 100, 'mean chance of not recommending', 'warn', '%')])
    st.dataframe(table.iloc[::-1], hide_index=True, width='stretch', column_config={
        'p_not_recommended': st.column_config.ProgressColumn('chance of not recommending', min_value=0, max_value=1, format='%.2f'),
        'review': st.column_config.TextColumn(width='large')})
    download, clear = st.columns([2, 5])
    download.download_button('Download CSV', to_csv(table), 'history.csv', 'text/csv', icon=':material/download:')
    if clear.button('Clear history'):
        st.session_state['history'] = []
        st.rerun()
