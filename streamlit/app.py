import re
from pathlib import Path

import joblib
import nltk
import streamlit as st

from text_utils import clean_text, ensure_nltk_data

# Resolve files relative to this script, not the current working directory,
# so the app also works when started from the repo root (e.g. Streamlit Cloud).
BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / 'recommend_pipeline.pkl'


@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


@st.cache_resource
def check_nltk_data():
    """Download missing NLTK data once per server process."""
    return tuple(ensure_nltk_data())


def set_example(text):
    st.session_state['review_text'] = text


def get_phrases(text):
    tagged = nltk.pos_tag(nltk.word_tokenize(text.lower()))
    return [a[0] + ' ' + b[0] for a, b in zip(tagged, tagged[1:])
            if a[1] == 'JJ' and b[1] in ('NN', 'NNS')]


st.title('Fashion Review Recommendation Predictor')
st.write('Paste a customer review to see if the customer would recommend the product.')

missing_nltk = check_nltk_data()
if missing_nltk:
    st.error('Required NLTK data could not be downloaded: ' + ', '.join(missing_nltk)
             + '. Check the internet connection and reload the page.')
    st.stop()

pipeline = load_pipeline()

# ---------- sidebar ----------
st.sidebar.header('Example reviews')
examples = [
    'Absolutely love this dress! The fabric is soft and the fit is perfect.',
    'Very disappointed. The material feels cheap and it is too small. Returning it.',
    'Lovely colour but runs very small.',
]
for i, example in enumerate(examples, 1):
    st.sidebar.button(f'Example {i}', on_click=set_example, args=(example,))

threshold = st.sidebar.slider('Threshold for "not recommended"', 0.1, 0.9, 0.5, 0.05)

st.sidebar.header('About this project')
st.sidebar.write('NLP practical exam (Set C): TF-IDF + Logistic Regression trained on women\'s '
                 'clothing reviews to predict whether a customer recommends a product.')

# ---------- inputs ----------
title = st.text_input('Review title')
review = st.text_area('Review text', key='review_text')

if st.button('Analyse'):
    if not review.strip():
        st.warning('Please enter a review')
    else:
        full_text = title + ' ' + review
        cleaned = clean_text(full_text)
        if not cleaned:
            st.warning('Nothing is left after cleaning (only symbols, numbers or common words). '
                       'Please write a longer review.')
            st.stop()
        prob_not_rec = pipeline.predict_proba([cleaned])[0][0]
        prob_rec = 1 - prob_not_rec

        if prob_not_rec > threshold:
            st.error('Customer would NOT recommend ⚠️')
        else:
            st.success('Customer would recommend ✅')

        st.metric('Probability of recommending', f'{prob_rec:.1%}')
        st.progress(float(prob_rec))

        if prob_not_rec > threshold:
            st.warning('Send to quality team')

        st.subheader('Review statistics')
        col1, col2, col3 = st.columns(3)
        col1.metric('Words', len(full_text.split()))
        col2.metric('Exclamation marks', len(re.findall(r'!', full_text)))
        col3.metric('ALL-CAPS words', len(re.findall(r"\b[A-Z]{3,}\b(?!['’]\w)", full_text)))

        with st.expander('Adjective + noun phrases (POS tagging)'):
            phrases = get_phrases(full_text)
            st.write(phrases if phrases else 'No phrases found')
