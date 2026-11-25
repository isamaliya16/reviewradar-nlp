"""Scores of the deployed model and of the other approaches (same split as the notebook)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (RANDOM_STATE, SUBSET_ROWS, TARGET, TEST_SIZE, VADER_GRID, WARNING_MIN_HITS)
from text_utils import clean_text


def split_reviews(df):
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET])
    return train.copy(), test.copy()


def deployed_scores(df, model):
    """Probability of 'not recommended' for every test review, from the deployed pipeline."""
    _, test = split_reviews(df)
    test['clean_text'] = test['full_text'].apply(clean_text)
    return test, model.predict_proba(test['clean_text'])[:, 0]


def metrics_at(y, p_not, threshold):
    """Precision / recall / F1 of the 'not recommended' class when flagging p_not > threshold."""
    flagged = p_not > threshold
    bad = np.asarray(y) == 0
    tp, fp = int((flagged & bad).sum()), int((flagged & ~bad).sum())
    fn, tn = int((~flagged & bad).sum()), int((~flagged & ~bad).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'flagged': tp + fp, 'accuracy': (tp + tn) / len(bad),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def threshold_curve(y, p_not, grid):
    rows = [{'threshold': float(t), **{k: m[k] for k in ('precision', 'recall', 'flagged')}}
            for t in grid for m in [metrics_at(y, p_not, t)]]
    return pd.DataFrame(rows)


def _row(name, y, pred, score_rec):
    return {'Model': name, 'Accuracy': accuracy_score(y, pred),
            'Precision (0)': precision_score(y, pred, pos_label=0, zero_division=0),
            'Recall (0)': recall_score(y, pred, pos_label=0, zero_division=0),
            'F1 (0)': f1_score(y, pred, pos_label=0, zero_division=0),
            'ROC-AUC': roc_auc_score(y, score_rec)}


def compare_models(df, word_vectors=None):
    """Baseline and models A-D, trained on the training reviews only. Returns (table, roc_curves)."""
    train, test = split_reviews(df)
    train['clean_text'] = train['full_text'].apply(clean_text)
    test['clean_text'] = test['full_text'].apply(clean_text)
    y_train, y_test = train[TARGET], test[TARGET]

    def make_lr():
        return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)

    rows = [_row('Baseline (always "recommends")', y_test, np.ones(len(y_test), dtype=int), np.zeros(len(y_test)))]
    curves = []
    features = {'A: Bag-of-Words': CountVectorizer(max_features=5000),
                'B: TF-IDF': TfidfVectorizer(max_features=5000),
                'C: TF-IDF + N-grams': TfidfVectorizer(ngram_range=(1, 2), max_features=10000)}
    fitted = {}
    for name, vectorizer in features.items():
        fitted[name] = Pipeline([('vec', vectorizer), ('clf', make_lr())]).fit(train['clean_text'], y_train)
    scores = {n: (m.predict(test['clean_text']), m.predict_proba(test['clean_text'])[:, 1]) for n, m in fitted.items()}

    if word_vectors is not None:
        def average(text):
            vectors = [word_vectors[w] for w in text.split() if w in word_vectors]
            return np.mean(vectors, axis=0) if vectors else np.zeros(word_vectors.vector_size)
        x_train = np.vstack(train['clean_text'].apply(average))
        x_test = np.vstack(test['clean_text'].apply(average))
        model_d = make_lr().fit(x_train, y_train)
        scores['D: Averaged Word2Vec'] = (model_d.predict(x_test), model_d.predict_proba(x_test)[:, 1])

    for name, (pred, score) in scores.items():
        rows.append(_row(name, y_test, pred, score))
        fpr, tpr, _ = roc_curve(y_test, score)
        curves.append(pd.DataFrame({'Model': name, 'False positive rate': fpr, 'True positive rate': tpr}))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def three_approaches(df, test_with_scores, p_not, warning_matcher, vader_scorer):
    """Heuristic vs VADER vs ML on the same first SUBSET_ROWS test reviews (as in the notebook).

    VADER's threshold is tuned on the first SUBSET_ROWS training reviews. vader_scorer may be None.
    """
    train, _ = split_reviews(df)
    test = test_with_scores.head(SUBSET_ROWS)
    y = test[TARGET]
    rows = []

    def add(name, needs_training, pred):
        rows.append({'Approach': name, 'Needs training data?': needs_training,
                     'Accuracy': accuracy_score(y, pred),
                     'Recall (0)': recall_score(y, pred, pos_label=0, zero_division=0),
                     'F1 (0)': f1_score(y, pred, pos_label=0, zero_division=0)})

    hits = test['full_text'].apply(lambda t: len(warning_matcher(t)))
    add(f'Heuristic (warning words, ≥ {WARNING_MIN_HITS} hit)', 'No', (hits < WARNING_MIN_HITS).astype(int))
    best_t = None
    if vader_scorer is not None:
        small = train.head(SUBSET_ROWS)
        train_scores = small['full_text'].apply(vader_scorer)
        best_t = max(VADER_GRID, key=lambda t: f1_score(small[TARGET], (train_scores >= t).astype(int), pos_label=0, zero_division=0))
        test_scores = test['full_text'].apply(vader_scorer)
        add(f'VADER (threshold {best_t} tuned on {len(small):,} train reviews)', 'No', (test_scores >= best_t).astype(int))
    add('ML (TF-IDF + N-grams + Logistic Regression)', 'Yes', (p_not[:len(test)] <= 0.5).astype(int))
    return pd.DataFrame(rows), best_t
