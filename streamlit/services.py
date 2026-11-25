"""Model loading and every prediction the pages need (cached where it is slow)."""
import re
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import data_utils
import evaluation
from config import (MAX_CHARS, MODEL_FILE, RISK_BANDS, RISK_NAMES, TOP_TERMS, VADER_THRESHOLD, WARNING_MIN_HITS,
                    WORD2VEC_FILE, LEXICON_DIR)
from text_utils import (clean_text, data_status, get_phrases, lemma_key, review_stats, split_tokens)


# ---- Cached resources ------------------------------------------------------------
@st.cache_resource
def ensure_nltk_data():
    return data_status(download=True)


@st.cache_resource(show_spinner='Loading model...')
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_resource(show_spinner='Loading word vectors...')
def load_word_vectors():
    """Skip-gram vectors, or None if gensim or the model file is missing."""
    try:
        from gensim.models import Word2Vec
        return Word2Vec.load(str(WORD2VEC_FILE)).wv
    except Exception:
        return None


@st.cache_resource
def load_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


@st.cache_data(show_spinner='Loading reviews...')
def get_reviews():
    return data_utils.load_reviews()


@st.cache_data
def warning_words():
    lines = (LEXICON_DIR / 'warning_words.txt').read_text(encoding='utf-8').splitlines()
    return [w.strip().lower() for w in lines if w.strip()]


@st.cache_resource
def themes():
    return data_utils.load_themes()


def system_status():
    ready = ensure_nltk_data()
    return {'ML model': MODEL_FILE.exists(), 'NLTK data': all(ready[k] for k in ready if k != 'vader_lexicon'),
            'VADER': ready.get('vader_lexicon', False), 'Word2Vec': load_word_vectors() is not None,
            'Dataset': get_reviews() is not None}


# ---- The three approaches ---------------------------------------------------------
def warning_hits(text):
    """Heuristic: the warning words / phrases found in the text."""
    low = text.lower()
    return [w for w in warning_words() if w in low]


def vader_compound(text):
    analyzer = load_vader()
    return None if analyzer is None else analyzer.polarity_scores(text)['compound']


def risk_level(p_not):
    low, high = RISK_BANDS
    return RISK_NAMES[0] if p_not < low else RISK_NAMES[1] if p_not < high else RISK_NAMES[2]


def detect_themes(text):
    """{theme: number of mentions} for the themes found in the text."""
    found = {name: len(pattern.findall(text)) for name, pattern in themes().items()}
    return {k: v for k, v in found.items() if v}


# ---- One review -------------------------------------------------------------------
@dataclass
class Analysis:
    title: str
    text: str
    full_text: str
    cleaned: str
    p_not: float
    threshold: float
    terms: pd.DataFrame                      # term, effect (+ pushes towards NOT recommended)
    highlights: list = field(default_factory=list)   # [(piece, effect or None)]
    phrases: list = field(default_factory=list)
    themes: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    heuristic_hits: list = field(default_factory=list)
    vader: float | None = None

    @property
    def p_rec(self):
        return 1 - self.p_not

    @property
    def flagged(self):
        return self.p_not > self.threshold

    @property
    def risk(self):
        return risk_level(self.p_not)


def explain(model, cleaned):
    """Terms of the review with their push: positive = towards 'not recommended'."""
    vectorizer, classifier = model.named_steps['vec'], model.named_steps['clf']
    row = vectorizer.transform([cleaned])
    names = vectorizer.get_feature_names_out()[row.indices]
    effect = -(row.data * classifier.coef_[0][row.indices])
    return pd.DataFrame({'term': names, 'effect': effect}).sort_values('effect', ascending=False, ignore_index=True)


def analyze_review(title, text, threshold):
    """Everything about one review. Raises ValueError when it has no usable words."""
    full_text = f'{title.strip()} {text.strip()}'.strip()[:MAX_CHARS]
    cleaned = clean_text(full_text)
    if not cleaned:
        raise ValueError('No useful words found. Please write a longer review.')
    model = load_model()
    terms = explain(model, cleaned)
    single = dict(zip(terms['term'], terms['effect']))
    highlights = [(piece, single.get(lemma_key(piece)) if is_word else None) for piece, is_word in split_tokens(full_text)]
    return Analysis(title=title.strip(), text=text.strip(), full_text=full_text, cleaned=cleaned,
                    p_not=float(model.predict_proba([cleaned])[0][0]), threshold=threshold, terms=terms,
                    highlights=highlights, phrases=get_phrases(full_text), themes=detect_themes(full_text),
                    stats=review_stats(full_text), heuristic_hits=warning_hits(full_text), vader=vader_compound(full_text))


# ---- Many reviews -----------------------------------------------------------------
def predict_batch(titles, texts, threshold):
    """DataFrame with one row per review: probability, flag, risk and the two simple approaches."""
    full = [f'{t.strip()} {r.strip()}'.strip()[:MAX_CHARS] for t, r in zip(titles, texts)]
    cleaned = [clean_text(t) for t in full]
    usable = [i for i, c in enumerate(cleaned) if c]
    p_not = np.full(len(full), np.nan)
    if usable:
        p_not[usable] = load_model().predict_proba([cleaned[i] for i in usable])[:, 0]
    table = pd.DataFrame({'title': list(titles), 'review': list(texts), 'p_not_recommended': p_not.round(3)})
    table['flagged'] = table['p_not_recommended'] > threshold
    table['risk'] = [risk_level(p) if not np.isnan(p) else 'No text' for p in p_not]
    table['warning_words'] = [len(warning_hits(t)) for t in full]
    compound = [vader_compound(t) for t in full]
    table['vader_compound'] = [None if c is None else round(c, 3) for c in compound]
    return table


# ---- Evaluation (cached) ------------------------------------------------------------
@st.cache_data(show_spinner='Scoring the test reviews...')
def get_test_scores():
    df = get_reviews()
    test, p_not = evaluation.deployed_scores(df, load_model())
    return test, p_not


@st.cache_data(show_spinner='Training and scoring models A-D (about 20 seconds)...')
def get_model_comparison():
    return evaluation.compare_models(get_reviews(), load_word_vectors())


@st.cache_data(show_spinner='Comparing the three approaches...')
def get_three_approaches():
    test, p_not = get_test_scores()
    scorer = vader_compound if load_vader() is not None else None
    return evaluation.three_approaches(get_reviews(), test, p_not, warning_hits, scorer)


def top_model_terms(n=TOP_TERMS * 2):
    """Strongest words of the deployed model in both directions."""
    model = load_model()
    names = model.named_steps['vec'].get_feature_names_out()
    coef = model.named_steps['clf'].coef_[0]
    order = coef.argsort()
    towards_not = pd.DataFrame({'term': names[order[:n]], 'weight': -coef[order[:n]]})
    towards_rec = pd.DataFrame({'term': names[order[::-1][:n]], 'weight': coef[order[::-1][:n]]})
    return towards_not, towards_rec


# ---- Word2Vec -----------------------------------------------------------------------
def neighbours(word, n):
    vectors = load_word_vectors()
    return pd.DataFrame(vectors.most_similar(word, topn=n), columns=['word', 'similarity'])


def embedding_map(word, n):
    """2-D PCA coordinates of a word and its n nearest neighbours."""
    from sklearn.decomposition import PCA
    vectors = load_word_vectors()
    words = [word] + [w for w, _ in vectors.most_similar(word, topn=n)]
    points = PCA(n_components=2, random_state=42).fit_transform(np.vstack([vectors[w] for w in words]))
    return pd.DataFrame({'word': words, 'x': points[:, 0], 'y': points[:, 1],
                         'kind': ['searched word'] + ['neighbour'] * n})
