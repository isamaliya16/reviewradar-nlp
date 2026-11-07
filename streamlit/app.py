import re
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------- paths and NLTK data ----------
# Use the folder of this file, so the app works from any start folder.
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / 'recommend_pipeline.pkl'

# NLTK data is saved in the "nltk_data" folder, so no internet is needed.
nltk.data.path.insert(0, str(BASE_DIR / 'nltk_data'))

# name -> places where NLTK data can be found (extracted folder or zip file)
NLTK_DATA = {
    'stopwords': ('corpora/stopwords', 'corpora/stopwords.zip/stopwords/'),
    'wordnet': ('corpora/wordnet', 'corpora/wordnet.zip/wordnet/'),
    'punkt_tab': ('tokenizers/punkt_tab', 'tokenizers/punkt_tab.zip/punkt_tab/'),
    'averaged_perceptron_tagger_eng': (
        'taggers/averaged_perceptron_tagger_eng',
        'taggers/averaged_perceptron_tagger_eng.zip/averaged_perceptron_tagger_eng/',
    ),
}


def is_installed(places):
    for place in places:
        try:
            nltk.data.find(place)
            return True
        except LookupError:
            pass
    return False


@st.cache_resource
def get_missing_data():
    """Try to download missing NLTK data. Return the names that are still missing."""
    missing = []
    for name, places in NLTK_DATA.items():
        if not is_installed(places):
            try:
                nltk.download(name, download_dir=str(BASE_DIR / 'nltk_data'), quiet=True)
            except Exception:
                pass
        if not is_installed(places):
            missing.append(name)
    return missing


# ---------- text cleaning ----------
@st.cache_resource
def get_cleaners():
    from nltk.corpus import stopwords
    words = set(stopwords.words('english')) - {'not', 'no', 'nor'}  # keep negations
    return words, WordNetLemmatizer()


def expand_not(text):
    text = text.replace("’", "'")
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    return re.sub(r"n't\b", " not", text)


def clean_text(text):
    stop_words, lemmatizer = get_cleaners()
    text = expand_not(str(text).lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [w for w in word_tokenize(text) if w not in stop_words]
    return ' '.join(lemmatizer.lemmatize(w) for w in words)


def get_phrases(text):
    """Find 'adjective + noun' pairs, like 'soft fabric'."""
    tagged = nltk.pos_tag(nltk.word_tokenize(text.lower()))
    return [a[0] + ' ' + b[0] for a, b in zip(tagged, tagged[1:])
            if a[1] == 'JJ' and b[1] in ('NN', 'NNS')]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def set_example(text):
    st.session_state['review_text'] = text


# ---------- page ----------
st.title('Fashion Review Recommendation Predictor')
st.write('Paste a customer review to see if the customer would recommend the product.')

missing = get_missing_data()
if missing:
    st.error('NLTK data is missing: ' + ', '.join(missing) + '. Please reload the page.')
    st.stop()

model = load_model()

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
st.sidebar.write('NLP practical exam (Set C): TF-IDF + Logistic Regression trained on '
                 "women's clothing reviews to predict if a customer recommends a product.")

# ---------- inputs ----------
title = st.text_input('Review title')
review = st.text_area('Review text', key='review_text')

if st.button('Analyse'):
    if not review.strip():
        st.warning('Please enter a review')
        st.stop()

    full_text = title + ' ' + review
    cleaned = clean_text(full_text)
    if not cleaned:
        st.warning('No useful words found. Please write a longer review.')
        st.stop()

    prob_not_rec = model.predict_proba([cleaned])[0][0]  # class 0 = not recommended
    prob_rec = 1 - prob_not_rec

    if prob_not_rec > threshold:
        st.error('Customer would NOT recommend ⚠️')
        st.warning('Send to quality team')
    else:
        st.success('Customer would recommend ✅')

    st.metric('Probability of recommending', f'{prob_rec:.1%}')
    st.progress(float(prob_rec))

    st.subheader('Review statistics')
    col1, col2, col3 = st.columns(3)
    col1.metric('Words', len(full_text.split()))
    col2.metric('Exclamation marks', full_text.count('!'))
    col3.metric('ALL-CAPS words', len(re.findall(r"\b[A-Z]{3,}\b(?!['’]\w)", full_text)))

    with st.expander('Adjective + noun phrases (POS tagging)'):
        phrases = get_phrases(full_text)
        st.write(phrases if phrases else 'No phrases found')
