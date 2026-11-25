"""Central settings. Change values here instead of editing the app code."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


def first_existing(*paths):
    """First path that exists (works from the repo root, from this folder and on Streamlit Cloud)."""
    return next((p for p in paths if p.exists()), paths[0])


# ---- Files -----------------------------------------------------------------
DATASET_NAME = 'Womens Clothing E-Commerce Reviews.csv'
DATASET_FILE = first_existing(BASE_DIR / 'dataset' / DATASET_NAME, ROOT_DIR / 'dataset' / DATASET_NAME)
MODEL_FILE = first_existing(BASE_DIR / 'recommend_pipeline.pkl', ROOT_DIR / 'model' / 'recommend_pipeline.pkl')
WORD2VEC_FILE = first_existing(BASE_DIR / 'word2vec_skipgram.model', ROOT_DIR / 'model' / 'word2vec_skipgram.model')
LEXICON_DIR = BASE_DIR / 'lexicon'
ASSETS_DIR = BASE_DIR / 'assets'
DATASET_URL = 'https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews'

# ---- NLTK data (shipped in `nltk_data/`, downloaded only if missing) ----------
NLTK_DIR = BASE_DIR / 'nltk_data'
NLTK_DATA = {   # name -> places where NLTK can find it (folder or zip file)
    'stopwords': ('corpora/stopwords', 'corpora/stopwords.zip/stopwords/'),
    'wordnet': ('corpora/wordnet', 'corpora/wordnet.zip/wordnet/'),
    'punkt_tab': ('tokenizers/punkt_tab', 'tokenizers/punkt_tab.zip/punkt_tab/'),
    'averaged_perceptron_tagger_eng': ('taggers/averaged_perceptron_tagger_eng',
                                       'taggers/averaged_perceptron_tagger_eng.zip/averaged_perceptron_tagger_eng/'),
    'vader_lexicon': ('sentiment/vader_lexicon', 'sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt'),
}
KEEP_STOPWORDS = {'not', 'no', 'nor'}      # negations are never removed

# ---- Data columns ------------------------------------------------------------
TARGET = 'Recommended IND'                 # 1 = recommends, 0 = does not recommend
COLUMNS = ['Title', 'Review Text', 'Rating', TARGET, 'Department Name', 'Class Name', 'Age']
AGE_BANDS = [18, 25, 35, 45, 55, 65, 100]

# ---- Decision threshold and risk levels -----------------------------------------
THRESHOLD_DEFAULT = 0.5
THRESHOLD_RANGE = (0.1, 0.9, 0.05)          # min, max, step
RISK_BANDS = (0.3, 0.7)                     # P(not recommended) below / above these
RISK_NAMES = ('Low', 'Medium', 'High')

# ---- The three approaches ------------------------------------------------------
WARNING_MIN_HITS = 1        # heuristic: flag when at least this many warning words are found
VADER_THRESHOLD = 0.4       # VADER: flag when compound score is below this (tuned in the notebook)
VADER_GRID = [0, 0.2, 0.4]  # values tried when tuning on the training subset
SUBSET_ROWS = 1000          # reviews used for the three-approach comparison

# ---- Evaluation (same split as the notebook) --------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---- Inputs and limits -----------------------------------------------------------
MAX_CHARS = 3000
MAX_BATCH_ROWS = 2000
TOP_TERMS = 8               # terms shown in "why" tables
NEIGHBOURS_DEFAULT = 12

# ---- RegEx patterns (from the notebook) -------------------------------------------
SIZE_PATTERN = r'(?i)\b(?:xxs|xs|small|medium|large|xl|xxl)\b'
HEIGHT_PATTERN = r"\b[4-6]['’]\s?\d{0,2}"
CAPS_PATTERN = r"\b[A-Z]{3,}\b(?!['’]\w)"

# ---- Colours (also used by charts) --------------------------------------------------
COLORS = {
    'primary': '#1E88E5', 'navy': '#0B2545', 'teal': '#00ACC1',
    'good': '#12B76A', 'bad': '#F04438', 'warn': '#F79009', 'muted': '#667085',
}

# ---- Part-of-speech descriptions ----------------------------------------------------
POS_DESCRIPTIONS = {'JJ': 'adjective', 'JJR': 'adjective (comparative)', 'JJS': 'adjective (superlative)',
                    'NN': 'noun', 'NNS': 'noun (plural)'}
