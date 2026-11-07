import re
from functools import lru_cache

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK resource name -> path used by nltk.data.find() to check if it is installed
NLTK_RESOURCES = {
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'punkt_tab': 'tokenizers/punkt_tab',
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
}


def _installed(path):
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False


def ensure_nltk_data():
    """Download any missing NLTK data and return the names that are still missing.

    nltk.download() returns False instead of raising when it fails, so
    availability is always re-checked afterwards.
    """
    missing = []
    for name, path in NLTK_RESOURCES.items():
        if not _installed(path):
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass
        if not _installed(path):
            missing.append(name)
    return missing


@lru_cache(maxsize=1)
def _stop_words():
    from nltk.corpus import stopwords
    return frozenset(set(stopwords.words('english')) - {'not', 'no', 'nor'})


@lru_cache(maxsize=1)
def _lemmatizer():
    return WordNetLemmatizer()


def expand_not(text):
    text = text.replace("’", "'")
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    return re.sub(r"n't\b", " not", text)


def clean_text(text):
    text = str(text).lower()
    text = expand_not(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = _stop_words()
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = _lemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)
