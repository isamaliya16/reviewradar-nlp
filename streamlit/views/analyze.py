"""Page 1: analyse one review."""
import pandas as pd
import streamlit as st

import ui
from config import MAX_CHARS, RISK_BANDS, TARGET, THRESHOLD_DEFAULT, VADER_THRESHOLD, WARNING_MIN_HITS
from services import analyze_review, get_reviews
from views.common import add_history

RISK_KIND = {'Low': 'good', 'Medium': 'warn', 'High': 'bad'}


def load_random_review():
    """Button callback: put a real review from the dataset into the form."""
    df = get_reviews()
    if df is None:
        return
    row = df.sample(1).iloc[0]
    st.session_state['rv_title'], st.session_state['rv_text'] = row['Title'], row['Review Text']
    st.session_state['rv_truth'] = {'text': row['Review Text'], 'rating': int(row['Rating']),
                                    'recommended': int(row[TARGET]), 'department': row['Department Name']}


def render():
    ui.hero('Review analyser', 'Would this customer recommend the product?',
            'Paste a fashion review. The model reads the words, estimates the chance of a return and tells you '
            'which words made the difference.')
    threshold = st.session_state.get('threshold', THRESHOLD_DEFAULT)
    previous = st.session_state.get('analysis')
    # Stable widget keys; defaults are set only when the widgets are new (e.g. coming back from another page).
    st.session_state.setdefault('rv_title', previous.title if previous else '')
    st.session_state.setdefault('rv_text', previous.text if previous else '')

    with st.form('review_form'):
        title = st.text_input('Review title', key='rv_title', placeholder='A short headline, e.g. what the customer wrote as title')
        text = st.text_area('Review text', key='rv_text', height=150, max_chars=MAX_CHARS,
                            placeholder='Type or paste the customer review here...')
        submitted = st.form_submit_button('Analyse', type='primary', icon=':material/radar:')
    left, right = st.columns([1, 2], vertical_alignment='center')
    left.button('Load a random real review', on_click=load_random_review, icon=':material/casino:')
    right.caption(f'Decision threshold: **{threshold:.2f}** (change it in the sidebar). '
                  'Flag = chance of “not recommended” above the threshold.')

    if submitted:
        if not text.strip():
            st.warning('Please enter a review')
        else:
            try:
                st.session_state['analysis'] = analyze_review(title, text, threshold)
                add_history(st.session_state['analysis'])
                previous = st.session_state['analysis']
            except ValueError as problem:
                st.warning(str(problem))

    if previous is None:
        ui.empty_state('📡', 'Write a review above and press Analyse to scan it.')
        return
    previous.threshold = threshold          # the sidebar slider updates the verdict live
    show_analysis(previous)


def show_analysis(a):
    flagged = a.flagged
    pills = (ui.pill(f'{a.risk} risk', RISK_KIND[a.risk]) +
             (ui.pill('Send to quality team', 'bad', alert=True) if flagged else ui.pill('No action needed', 'good')) +
             ui.pill(f'Threshold {a.threshold:.2f}', 'teal'))
    ui.verdict_card(
        a.p_not, a.threshold, flagged, a.risk,
        'Customer would NOT recommend' if flagged else 'Customer would recommend',
        (f'The model sees a {a.p_not:.0%} chance that this customer is unhappy, above your {a.threshold:.0%} threshold. '
         'This review should be looked at by the quality team.') if flagged else
        (f'The model sees only a {a.p_not:.0%} chance that this customer is unhappy, below your {a.threshold:.0%} threshold.'),
        pills)

    truth = st.session_state.get('rv_truth')
    if truth and truth['text'] == a.text:
        agrees = (truth['recommended'] == 0) == flagged
        st.info(f"Real label of this review: rated {truth['rating']}★ in {truth['department']}, the customer "
                f"{'recommends' if truth['recommended'] else 'does not recommend'} the product. "
                f"The model {'agrees' if agrees else 'disagrees'}.")

    ui.kpi_cards([
        ('Chance of recommending', a.p_rec * 100, 'probability from the model', 'good', '%'),
        ('Chance of not recommending', a.p_not * 100, 'class 0 probability', 'bad', '%'),
        ('Risk level', a.risk, f'low < {RISK_BANDS[0]:.0%} · high ≥ {RISK_BANDS[1]:.0%}', RISK_KIND[a.risk], ''),
        ('Review length', a.stats['words'], 'words (title + text)', 'teal', ''),
    ])

    tabs = st.tabs(['Why this verdict', 'Compare approaches', 'Review statistics', 'Language (POS)', 'Cleaned text'])
    with tabs[0]:
        why_tab(a)
    with tabs[1]:
        compare_tab(a)
    with tabs[2]:
        stats_tab(a)
    with tabs[3]:
        if a.phrases:
            st.markdown('Adjective + noun phrases found with POS tagging:')
            st.html('<div class="pills">' + ''.join(ui.pill(p, 'teal') for p in a.phrases) + '</div>')
        else:
            ui.empty_state('🔎', 'No adjective + noun phrases found in this review.')
    with tabs[4]:
        st.caption('What the model actually sees: lowercase, letters only, stop-words removed (not / no / nor kept), lemmatized.')
        st.code(a.cleaned, language=None, wrap_lines=True)


def why_tab(a):
    ui.section('Your review, colour-coded')
    ui.highlighted_review(a.highlights)
    if a.themes:
        ui.section('Topics mentioned')
        st.html('<div class="pills">' + ''.join(ui.pill(f'{name} · {n}', 'teal') for name, n in a.themes.items()) + '</div>')
    ui.section('Words and phrases that decided it')
    toward_not = a.terms[a.terms['effect'] > 0].head(8)
    toward_rec = a.terms[a.terms['effect'] < 0].tail(8).iloc[::-1]
    left, right = st.columns(2)
    with left:
        st.markdown('**Pushes towards “not recommended”**')
        if toward_not.empty:
            ui.empty_state('✅', 'Nothing in this review points to a return.')
        else:
            ui.meters([(t, e) for t, e in zip(toward_not['term'], toward_not['effect'])], lambda v: 'bad')
    with right:
        st.markdown('**Pushes towards “recommends”**')
        if toward_rec.empty:
            ui.empty_state('⚠️', 'Nothing in this review speaks for the product.')
        else:
            ui.meters([(t, -e) for t, e in zip(toward_rec['term'], toward_rec['effect'])], lambda v: 'good')
    st.caption('Bars show how strongly each term pulled the model’s decision (TF-IDF value × logistic-regression weight). '
               'Two-word terms such as “not worth” are used by the model too.')


def compare_tab(a):
    heuristic_flag = len(a.heuristic_hits) >= WARNING_MIN_HITS
    ml_flag = a.flagged
    rows = [
        {'Approach': 'Heuristic (warning words)', 'Flags the review?': 'Yes' if heuristic_flag else 'No',
         'Evidence': ', '.join(a.heuristic_hits) if a.heuristic_hits else 'no warning word found'},
        {'Approach': 'ML (TF-IDF + N-grams + Logistic Regression)', 'Flags the review?': 'Yes' if ml_flag else 'No',
         'Evidence': f'{a.p_not:.1%} chance of "not recommended" (threshold {a.threshold:.2f})'},
    ]
    if a.vader is not None:
        vader_flag = a.vader < VADER_THRESHOLD
        rows.insert(1, {'Approach': 'VADER (general sentiment)', 'Flags the review?': 'Yes' if vader_flag else 'No',
                        'Evidence': f'compound score {a.vader:+.2f} (flag below {VADER_THRESHOLD})'})
    table = pd.DataFrame(rows)
    flags = set(table['Flags the review?'])
    if len(flags) == 1:
        st.success(f'All approaches agree: {"flag" if "Yes" in flags else "do not flag"} this review.')
    else:
        st.info('The approaches disagree. The heuristic and VADER ignore context; the ML model learned from 18,000 labelled reviews.')
    st.dataframe(table, hide_index=True, width='stretch')


def stats_tab(a):
    s = a.stats
    ui.kpi_cards([('Words', s['words'], 'split on spaces', 'teal', ''),
                  ('Characters', s['characters'], 'including spaces', 'teal', ''),
                  ('Exclamation marks', s['exclamations'], 'regex  !', 'warn', ''),
                  ('Question marks', s['questions'], 'regex  ?', 'warn', ''),
                  ('ALL-CAPS words', s['caps'], 'regex  \\b[A-Z]{3,}\\b', 'bad', '')])
    left, right = st.columns(2)
    with left:
        st.markdown('**Sizes mentioned**')
        st.html('<div class="pills">' + (''.join(ui.pill(x, 'teal') for x in s['sizes']) or ui.pill('none found')) + '</div>')
    with right:
        st.markdown('**Heights mentioned**')
        st.html('<div class="pills">' + (''.join(ui.pill(x, 'teal') for x in s['heights']) or ui.pill('none found')) + '</div>')
