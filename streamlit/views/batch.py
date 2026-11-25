"""Page 2: analyse many reviews at once and build the quality queue."""
import altair as alt
import pandas as pd
import streamlit as st

import ui
from config import COLORS, MAX_BATCH_ROWS, RISK_NAMES, THRESHOLD_DEFAULT
from services import predict_batch
from views.common import to_csv

TEXT_HINTS = ('review text', 'review', 'text', 'comment', 'message')
TITLE_HINTS = ('title', 'headline', 'subject')


def read_reviews():
    """(titles, texts) from the typed box or the uploaded file."""
    source = st.radio('Input', ['Type or paste reviews', 'Upload a file'], horizontal=True)
    if source == 'Type or paste reviews':
        typed = st.text_area('One review per line', height=190, key='batch_typed',
                             placeholder='Write or paste each review on its own line...')
        lines = [line.strip() for line in typed.splitlines() if line.strip()]
        return [''] * len(lines), lines

    uploaded = st.file_uploader('CSV or TXT file', type=['csv', 'txt'])
    if uploaded is None:
        return [], []
    if uploaded.name.lower().endswith('.txt'):
        lines = [l.strip() for l in uploaded.getvalue().decode('utf-8', errors='replace').splitlines() if l.strip()]
        return [''] * len(lines), lines
    try:
        table = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        try:
            table = pd.read_csv(uploaded, encoding='latin-1')
        except Exception:
            st.error('This file could not be read as a CSV.')
            return [], []
    if table.empty:
        st.error('The file has no rows.')
        return [], []
    names = [str(c) for c in table.columns]
    guess = lambda hints, default: next((i for h in hints for i, n in enumerate(names) if n.lower() == h), default)
    left, right = st.columns(2)
    text_col = left.selectbox('Column with the review text', table.columns, index=guess(TEXT_HINTS, 0))
    title_col = right.selectbox('Column with the title (optional)', ['(none)'] + names,
                                index=(guess(TITLE_HINTS, -1) + 1))
    table = table.dropna(subset=[text_col])
    titles = [''] * len(table) if title_col == '(none)' else table[title_col].fillna('').astype(str).tolist()
    return titles, table[text_col].astype(str).tolist()


def render():
    ui.hero('Batch scan', 'Sort a whole pile of reviews in seconds',
            'Paste reviews or upload a CSV. Every review gets a risk score and the riskiest ones land in a '
            'ready-made queue for the quality team.')
    threshold = st.session_state.get('threshold', THRESHOLD_DEFAULT)
    titles, texts = read_reviews()
    if len(texts) > MAX_BATCH_ROWS:
        st.info(f'Only the first {MAX_BATCH_ROWS} of {len(texts)} reviews will be analysed.')
        titles, texts = titles[:MAX_BATCH_ROWS], texts[:MAX_BATCH_ROWS]
    if texts:
        st.caption(f'{len(texts)} review(s) ready · threshold {threshold:.2f}')

    if st.button('Run scan', type='primary', icon=':material/radar:'):
        if not texts:
            st.warning('Please enter or upload some reviews first.')
        else:
            with st.spinner('Scanning...'):
                st.session_state['batch_result'] = predict_batch(titles, texts, threshold)
                st.session_state['batch_threshold'] = threshold

    result = st.session_state.get('batch_result')
    if result is None:
        ui.empty_state('📦', 'Your results will appear here after the scan.')
        return
    show_result(result, st.session_state.get('batch_threshold', threshold))


def show_result(result, threshold):
    scored = result.dropna(subset=['p_not_recommended'])
    flagged = scored[scored['p_not_recommended'] > threshold].sort_values('p_not_recommended', ascending=False)
    ui.section('Results')
    ui.kpi_cards([
        ('Reviews scanned', len(result), f'{len(result) - len(scored)} without usable words', 'teal', ''),
        ('Sent to quality team', len(flagged), f'above threshold {threshold:.2f}', 'bad', ''),
        ('Share flagged', 100 * len(flagged) / max(len(scored), 1), 'of the scanned reviews', 'warn', '%'),
        ('Average risk', 100 * scored['p_not_recommended'].mean() if len(scored) else 0, 'mean chance of not recommending', 'good', '%'),
    ])
    left, right = st.columns(2)
    with left:
        st.markdown('**Reviews per risk level**')
        counts = scored['risk'].value_counts().reindex(RISK_NAMES).fillna(0).astype(int).reset_index()
        counts.columns = ['risk', 'reviews']
        ui.bar_chart(counts, 'risk', 'reviews', group='risk', sort=list(RISK_NAMES), height=260,
                     colors=dict(zip(RISK_NAMES, [COLORS['good'], COLORS['warn'], COLORS['bad']])))
    with right:
        st.markdown('**How the risk scores are spread**')
        histogram = alt.Chart(scored).mark_bar(cornerRadiusEnd=5, color=COLORS['primary']).encode(
            x=alt.X('p_not_recommended:Q', bin=alt.Bin(step=0.1), title=None), y=alt.Y('count()', title=None))
        ui.show_chart(histogram, 260)

    ui.section('Quality queue (riskiest first)')
    if flagged.empty:
        ui.empty_state('✅', 'No review is above the threshold. Nothing to send to the quality team.')
    else:
        st.dataframe(flagged.drop(columns=['flagged']), hide_index=True, width='stretch', column_config={
            'p_not_recommended': st.column_config.ProgressColumn('chance of not recommending', min_value=0, max_value=1, format='%.2f'),
            'review': st.column_config.TextColumn('review', width='large')})
    download, other = st.columns([1, 1])
    download.download_button('Download all results (CSV)', to_csv(result), 'reviewradar_results.csv', 'text/csv', icon=':material/download:')
    if not flagged.empty:
        other.download_button('Download quality queue (CSV)', to_csv(flagged), 'quality_queue.csv', 'text/csv', icon=':material/download:')
