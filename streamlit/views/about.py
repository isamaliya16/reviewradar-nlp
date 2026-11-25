"""Page 6: how the project works (pipeline, ideas, tools)."""
import sys
from importlib.metadata import PackageNotFoundError, version

import streamlit as st

import ui
from config import DATASET_URL, TARGET
from evaluation import split_reviews
from services import get_reviews
from views.common import need_data

PIPELINE = [
    ('Collect', 'Load the reviews CSV, keep five columns and drop reviews that have no text.'),
    ('Split first', '80 / 20 stratified split before anything is fitted, so the test reviews stay unseen.'),
    ('Clean', 'Lowercase, expand “n’t” to “not”, keep letters only (RegEx).'),
    ('Tokenise', 'NLTK word_tokenize cuts the text into words.'),
    ('Stop-words', 'Remove filler words but keep “not”, “no” and “nor”, so “not comfortable” never becomes “comfortable”.'),
    ('Lemmatize', 'WordNetLemmatizer reduces words to their base form (“dresses” → “dress”).'),
    ('Features', 'Bag-of-Words, TF-IDF, TF-IDF with bigrams and averaged Word2Vec; fitted on training reviews only.'),
    ('Model', 'Logistic Regression with class_weight = balanced, so the small unhappy class counts more.'),
    ('Evaluate', 'Recall, precision and F1 of the unhappy class, ROC-AUC and the confusion matrix.'),
    ('Deploy', 'The pipeline is saved with joblib and served by this app.'),
]
IDEAS = [
    ('Why recall of the unhappy class?', 'Only about 18% of customers do not recommend, so a model that always says “recommends” is right '
     'about 82% of the time while catching nobody. Recall of the minority class measures what really matters.'),
    ('Why no data leakage?', 'The split happens first. Every vectoriser and Word2Vec model is fitted on training reviews and only transforms the test reviews.'),
    ('What does the threshold do?', 'A lower threshold flags more reviews: more unhappy customers are caught but the quality team has more to check. '
     'The Model lab page shows the trade-off live.'),
    ('Why keep “not”?', 'Negations flip the meaning. Standard stop-word lists remove them, which would hide the clearest signals of a bad review.'),
    ('Why is the ML model better than rules?', 'The heuristic (warning words) and VADER (general sentiment) ignore context. The model learns phrases such as '
     '“not worth” or “wanted to love” from labelled reviews.'),
]
PACKAGES = ['streamlit', 'scikit-learn', 'nltk', 'gensim', 'pandas', 'numpy', 'altair']


def render():
    ui.hero('How it works', 'From raw reviews to a deployed radar',
            'A short tour of the data, the pipeline and the ideas behind this project.')
    df = get_reviews()
    if need_data(df):
        train, test = split_reviews(df)
        ui.kpi_cards([('Reviews', len(df), 'with text', 'teal', ''), ('Training', len(train), 'reviews for fitting', 'good', ''),
                      ('Test', len(test), 'unseen reviews for scoring', 'warn', ''),
                      ('Do not recommend', (1 - df[TARGET].mean()) * 100, 'minority class', 'bad', '%')])
    ui.section('The pipeline')
    ui.timeline(PIPELINE)
    ui.section('Key ideas')
    for question, answer in IDEAS:
        with st.expander(question):
            st.write(answer)
    ui.section('Tools')
    rows = []
    for package in PACKAGES:
        try:
            rows.append((package, version(package)))
        except PackageNotFoundError:
            rows.append((package, 'not installed'))
    st.html('<div class="pills">' + ui.pill(f'Python {sys.version.split()[0]}', 'teal')
            + ''.join(ui.pill(f'{p} {v}') for p, v in rows) + '</div>')
    st.markdown(f'Data: [Women’s E-Commerce Clothing Reviews on Kaggle]({DATASET_URL}) · '
                '[Project on GitHub](https://github.com/isamaliya16/reviewradar-nlp)')
