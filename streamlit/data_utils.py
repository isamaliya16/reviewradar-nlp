"""Loading the reviews and simple statistics (no Streamlit in here)."""
import json
import re
from collections import Counter

import pandas as pd

from config import AGE_BANDS, COLUMNS, DATASET_FILE, KEEP_STOPWORDS, LEXICON_DIR, TARGET


def load_reviews():
    """Reviews prepared exactly like the notebook. Returns None if the CSV is missing."""
    if not DATASET_FILE.exists():
        return None
    df = pd.read_csv(DATASET_FILE, index_col=0)[COLUMNS]
    df = df.dropna(subset=['Review Text']).copy()
    df['Title'] = df['Title'].fillna('')
    df['full_text'] = (df['Title'] + ' ' + df['Review Text']).str.strip()
    df['Class Name'] = df['Class Name'].fillna('Unknown')
    df['Department Name'] = df['Department Name'].fillna('Unknown')
    df['verdict'] = df[TARGET].map({1: 'recommends', 0: 'does not recommend'})
    labels = [f'{lo}-{hi - 1}' if hi < 100 else f'{lo}+' for lo, hi in zip(AGE_BANDS, AGE_BANDS[1:])]
    df['age_band'] = pd.cut(df['Age'], bins=AGE_BANDS, right=False, labels=labels)
    return df


def not_recommended_rate(df, by):
    """Share (0-100) of reviews that do not recommend, per group, plus the group size."""
    grouped = df.groupby(by, observed=True)[TARGET]
    table = pd.DataFrame({'not_recommended_%': (1 - grouped.mean()) * 100, 'reviews': grouped.size()})
    return table.reset_index()


def load_themes():
    """{theme: compiled regex} from lexicon/themes.json."""
    raw = json.loads((LEXICON_DIR / 'themes.json').read_text(encoding='utf-8'))
    return {name: re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')\b', re.I)
            for name, words in raw.items()}


def theme_rates(df):
    """For each theme: how many reviews mention it and how many of those do not recommend."""
    rows = []
    for name, pattern in load_themes().items():
        mentions = df['full_text'].str.contains(pattern)
        rows.append({'theme': name, 'reviews': int(mentions.sum()),
                     'not_recommended_%': float((1 - df.loc[mentions, TARGET].mean()) * 100)})
    return pd.DataFrame(rows)


def top_words(df, recommended, n=15):
    """Most frequent words (stop-words removed, negations kept) in one class of reviews."""
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english')) - KEEP_STOPWORDS
    counter = Counter()
    for text in df.loc[df[TARGET] == int(recommended), 'full_text']:
        counter.update(w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop and len(w) > 2)
    return pd.DataFrame(counter.most_common(n), columns=['word', 'count'])
