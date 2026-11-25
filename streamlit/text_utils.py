"""Text cleaning and RegEx helpers, shared by the notebook and the app (no Streamlit in here)."""
import re
from functools import lru_cache

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from config import CAPS_PATTERN, HEIGHT_PATTERN, KEEP_STOPWORDS, NLTK_DATA, NLTK_DIR, SIZE_PATTERN

nltk.data.path.insert(0, str(NLTK_DIR))


# ---- NLTK data -----------------------------------------------------------------
def is_installed(places):
    for place in places:
        try:
            nltk.data.find(place)
            return True
        except LookupError:
            pass
    return False


def data_status(download=False):
    """{name: available}. With download=True, missing data is downloaded into nltk_data/."""
    status = {}
    for name, places in NLTK_DATA.items():
        if download and not is_installed(places):
            try:
                nltk.download(name, download_dir=str(NLTK_DIR), quiet=True)
            except Exception:
                pass
        status[name] = is_installed(places)
    return status


@lru_cache(maxsize=1)
def _tools():
    from nltk.corpus import stopwords
    return set(stopwords.words('english')) - KEEP_STOPWORDS, WordNetLemmatizer()


# ---- Cleaning -------------------------------------------------------------------
def expand_not(text):
    text = text.replace('’', "'")
    text = re.sub(r"\bcan't\b", 'can not', text)
    text = re.sub(r"\bwon't\b", 'will not', text)
    return re.sub(r"n't\b", ' not', text)


def clean_text(text):
    """lowercase -> expand n't -> letters only -> tokenise -> drop stop-words (keep not/no/nor) -> lemmatize."""
    stop_words, lemmatizer = _tools()
    text = re.sub(r'[^a-z\s]', ' ', expand_not(str(text).lower()))
    return ' '.join(lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in stop_words)


def lemma_key(token):
    """The model feature a written word turns into (used to colour words in a review)."""
    low = token.lower().replace('’', "'")
    if low.endswith("n't") or low == 'cannot':
        return 'not'
    return _tools()[1].lemmatize(low)


def split_tokens(text):
    """[(piece, is_word)] - the review cut into words and the text between them, so it can be rebuilt."""
    return [(m.group(), m.group()[0].isalpha()) for m in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*|[^A-Za-z]+", text)]


# ---- RegEx statistics and POS phrases ---------------------------------------------
def review_stats(text):
    return {
        'words': len(text.split()),
        'characters': len(text),
        'exclamations': text.count('!'),
        'questions': text.count('?'),
        'caps': len(re.findall(CAPS_PATTERN, text)),
        'sizes': sorted({m.lower() for m in re.findall(SIZE_PATTERN, text)}),
        'heights': [h.strip() for h in re.findall(HEIGHT_PATTERN, text)],
    }


def get_phrases(text):
    """'adjective + noun' pairs such as 'soft fabric'."""
    tagged = nltk.pos_tag(nltk.word_tokenize(text.lower()))
    return [a[0] + ' ' + b[0] for a, b in zip(tagged, tagged[1:]) if a[1] == 'JJ' and b[1] in ('NN', 'NNS')]
