"""Page 3: what do the reviews say? (statistics of the dataset)"""
import pandas as pd
import streamlit as st

import ui
from config import COLORS, TARGET
from data_utils import not_recommended_rate, theme_rates, top_words
from services import analyze_review, get_reviews
from views.common import need_data

VERDICT_COLORS = {'recommends': COLORS['good'], 'does not recommend': COLORS['bad']}


@st.cache_data(show_spinner='Counting words...')
def cached_top_words(recommended, n):
    return top_words(get_reviews(), recommended, n)


@st.cache_data(show_spinner='Looking for topics...')
def cached_theme_rates():
    return theme_rates(get_reviews())


def render():
    ui.hero('Insights', 'What are customers telling us?',
            'Explore the reviews behind the model: who is unhappy, what they talk about and which words give them away.')
    df = get_reviews()
    if not need_data(df):
        return
    unhappy = (1 - df[TARGET].mean()) * 100
    ui.kpi_cards([('Reviews with text', len(df), 'after dropping empty reviews', 'teal', ''),
                  ('Do not recommend', unhappy, 'the minority class we must catch', 'bad', '%'),
                  ('Average rating', f'{df["Rating"].mean():.2f} ★', 'out of 5', 'warn', ''),
                  ('Departments', df['Department Name'].nunique(), f'{df["Class Name"].nunique()} product classes', 'good', '')])
    overview, topics, browse = st.tabs(['Overview', 'Topics and words', 'Browse reviews'])
    with overview:
        overview_tab(df)
    with topics:
        topics_tab(df)
    with browse:
        browse_tab(df)


def overview_tab(df):
    left, right = st.columns(2)
    with left:
        st.markdown('**Recommend or not?**')
        balance = df['verdict'].value_counts().reset_index()
        balance.columns = ['verdict', 'reviews']
        ui.bar_chart(balance, 'verdict', 'reviews', group='verdict', colors=VERDICT_COLORS, height=280)
        st.caption('Always guessing "recommends" would be right most of the time but would catch no unhappy customer at all.')
    with right:
        st.markdown('**Ratings by verdict**')
        ratings = df.groupby(['Rating', 'verdict']).size().reset_index(name='reviews')
        ui.bar_chart(ratings, 'Rating:O', 'reviews', group='verdict', colors=VERDICT_COLORS, height=280)
        st.caption('High ratings are mostly recommended, low ratings mostly not, but the middle is mixed.')
    st.markdown('**Share of reviews that do not recommend, by department**')
    departments = not_recommended_rate(df, 'Department Name').sort_values('not_recommended_%', ascending=False)
    ui.bar_chart(departments, 'Department Name', 'not_recommended_%', horizontal=True, sort='-x', height=260, color=COLORS['bad'])
    left, right = st.columns(2)
    with left:
        st.markdown('**By age group**')
        ages = not_recommended_rate(df.dropna(subset=['age_band']), 'age_band')
        ages['age_band'] = ages['age_band'].astype(str)
        ui.bar_chart(ages, 'age_band', 'not_recommended_%', height=260, color=COLORS['primary'])
    with right:
        st.markdown('**By product class (10 biggest)**')
        classes = not_recommended_rate(df, 'Class Name').nlargest(10, 'reviews').sort_values('not_recommended_%', ascending=False)
        ui.bar_chart(classes, 'Class Name', 'not_recommended_%', horizontal=True, sort='-x', height=260, color=COLORS['teal'])


def topics_tab(df):
    st.markdown('**What do customers talk about?** Share of reviews that do not recommend, among reviews mentioning each topic.')
    themes = cached_theme_rates().sort_values('not_recommended_%', ascending=False)
    ui.bar_chart(themes, 'theme', 'not_recommended_%', horizontal=True, sort='-x', height=250, color=COLORS['bad'])
    st.dataframe(themes.round(1), hide_index=True, width='stretch')
    st.caption('Topics come from the word lists in lexicon/themes.json, so you can edit them without touching the code.')
    amount = st.slider('How many words?', 5, 25, 12)
    left, right = st.columns(2)
    with left:
        st.markdown('**Most frequent words: do not recommend**')
        ui.bar_chart(cached_top_words(False, amount), 'word', 'count', horizontal=True, sort='-x', height=360, color=COLORS['bad'])
    with right:
        st.markdown('**Most frequent words: recommend**')
        ui.bar_chart(cached_top_words(True, amount), 'word', 'count', horizontal=True, sort='-x', height=360, color=COLORS['good'])


def browse_tab(df):
    left, right, third = st.columns([2, 2, 1])
    departments = left.multiselect('Department', sorted(df['Department Name'].unique()))
    ratings = right.multiselect('Rating', [1, 2, 3, 4, 5])
    verdict = third.selectbox('Verdict', ['all', 'recommends', 'does not recommend'])
    keyword = st.text_input('Review contains the word or phrase', placeholder='e.g. a word you want to look for')
    view = df
    if departments:
        view = view[view['Department Name'].isin(departments)]
    if ratings:
        view = view[view['Rating'].isin(ratings)]
    if verdict != 'all':
        view = view[view['verdict'] == verdict]
    if keyword.strip():
        view = view[view['full_text'].str.contains(keyword.strip(), case=False, regex=False)]
    st.caption(f'{len(view):,} review(s) match. Click a row to see what the model thinks.')
    shown = view[['Title', 'Review Text', 'Rating', 'verdict', 'Department Name']].head(500).reset_index(drop=True)
    event = st.dataframe(shown, hide_index=True, width='stretch', height=320, on_select='rerun', selection_mode='single-row',
                         column_config={'Review Text': st.column_config.TextColumn(width='large')})
    if event.selection.rows:
        row = shown.iloc[event.selection.rows[0]]
        threshold = st.session_state.get('threshold', 0.5)
        result = analyze_review(row['Title'], row['Review Text'], threshold)
        agrees = (row['verdict'] == 'does not recommend') == result.flagged
        st.html(f'<div class="card"><div class="pills">{ui.pill(f"Model: {result.p_not:.0%} chance of not recommending", "bad" if result.flagged else "good")}'
                f'{ui.pill("Real label: " + row["verdict"], "teal")}{ui.pill("Model agrees" if agrees else "Model disagrees", "good" if agrees else "warn")}</div></div>')
