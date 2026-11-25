"""Page 5: how good is the model, and what does the threshold change?"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

import ui
from config import COLORS, RISK_BANDS, TARGET, THRESHOLD_DEFAULT
from evaluation import metrics_at, threshold_curve
from services import (get_model_comparison, get_reviews, get_test_scores, get_three_approaches, top_model_terms)
from views.common import need_data


def render():
    ui.hero('Model lab', 'How good is the radar?',
            'Everything here is measured live on 4,500 reviews the model never saw during training. '
            'Move the threshold in the sidebar to trade missed customers against false alarms.')
    if not need_data(get_reviews()):
        return
    test, p_not = get_test_scores()
    y = test[TARGET].to_numpy()
    threshold = st.session_state.get('threshold', THRESHOLD_DEFAULT)
    m = metrics_at(y, p_not, threshold)

    ui.kpi_cards([
        ('Recall of unhappy customers', m['recall'] * 100, f'{m["tp"]:,} of {m["tp"] + m["fn"]:,} caught', 'good', '%'),
        ('Precision of the flags', m['precision'] * 100, f'{m["tp"]:,} of {m["flagged"]:,} flags are right', 'teal', '%'),
        ('F1 score (unhappy class)', m['f1'] * 100, 'balance of precision and recall', 'warn', '%'),
        ('Reviews flagged', m['flagged'], f'out of {len(y):,} test reviews', 'bad', ''),
        ('ROC-AUC', f'{roc_auc_score(y, 1 - p_not):.3f}', '1.0 = perfect, 0.5 = guessing', '', ''),
    ])

    ui.section('Threshold explorer')
    left, right = st.columns([3, 2])
    with left:
        grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
        curve = threshold_curve(y, p_not, grid)
        long = curve.melt(id_vars='threshold', value_vars=['precision', 'recall'], var_name='metric', value_name='score')
        ui.line_chart(long, 'threshold', 'score', 'metric', height=300,
                      colors={'precision': COLORS['teal'], 'recall': COLORS['bad']})
        st.caption(f'Lower threshold: more unhappy customers caught, more reviews to check. Current threshold: {threshold:.2f}.')
    with right:
        st.markdown(f'**Confusion matrix at threshold {threshold:.2f}**')
        ui.confusion_matrix(m['tp'], m['fp'], m['fn'], m['tn'])
    table = pd.DataFrame([{'Threshold': t, 'Precision (0)': (r := metrics_at(y, p_not, t))['precision'],
                           'Recall (0)': r['recall'], 'Reviews flagged': r['flagged']}
                          for t in (RISK_BANDS[0], THRESHOLD_DEFAULT, RISK_BANDS[1])])
    st.dataframe(table.round(3), hide_index=True, width='stretch')

    ui.section('Which model is best?')
    if st.button('Train and compare models A-D', icon=':material/model_training:'):
        st.session_state['show_comparison'] = True
    if st.session_state.get('show_comparison'):
        comparison, roc = get_model_comparison()
        st.dataframe(comparison.round(3), hide_index=True, width='stretch')
        a, b = st.columns(2)
        with a:
            st.markdown('**ROC curves**')
            ui.line_chart(roc, 'False positive rate', 'True positive rate', 'Model', height=320)
        with b:
            st.markdown('**Recall and F1 of the unhappy class**')
            bars = comparison.melt(id_vars='Model', value_vars=['Recall (0)', 'F1 (0)'], var_name='metric', value_name='score')
            ui.bar_chart(bars, 'Model', 'score', group='metric', horizontal=True, height=320,
                         colors={'Recall (0)': COLORS['bad'], 'F1 (0)': COLORS['primary']})
        st.caption('Every vectoriser and model is fitted on training reviews only. The deployed model is C: TF-IDF + N-grams.')
    else:
        ui.empty_state('🧪', 'Press the button to train the baseline and models A-D live (about 20 seconds, then cached).')

    ui.section('Heuristic vs VADER vs machine learning')
    approaches, vader_t = get_three_approaches()
    st.dataframe(approaches.round(3), hide_index=True, width='stretch')
    st.caption('Same first 1,000 test reviews for all three. The two simple approaches need no training but ignore context.')

    ui.section('Strongest words of the model')
    towards_not, towards_rec = top_model_terms()
    a, b = st.columns(2)
    with a:
        st.markdown('**Towards “not recommended”**')
        ui.bar_chart(towards_not, 'term', 'weight', horizontal=True, sort='-x', height=380, color=COLORS['bad'])
    with b:
        st.markdown('**Towards “recommends”**')
        ui.bar_chart(towards_rec, 'term', 'weight', horizontal=True, sort='-x', height=380, color=COLORS['good'])
