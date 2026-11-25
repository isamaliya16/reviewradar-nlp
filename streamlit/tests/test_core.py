"""Logic tests with hand-written reviews (no example data taken from the app)."""
import pandas as pd

import services
import text_utils
from evaluation import metrics_at

UNHAPPY = 'The stitching came apart after one wash and the fabric is so cheap. Very disappointed, I am sending it back.'
HAPPY = 'Gorgeous dress, the fabric is soft and the fit is perfect. I will buy it in another color!'


def test_clean_text_keeps_negations_and_lemmatizes():
    assert text_utils.clean_text("It doesn't fit and the dresses can't be returned") == 'not fit dress not returned'
    assert text_utils.clean_text('12345 !!! ???') == ''


def test_review_stats_regex():
    stats = text_utils.review_stats("I'm 5'4\" and the SMALL was perfect! Really? Size XL runs big!")
    assert stats['exclamations'] == 2 and stats['questions'] == 1
    assert stats['heights'] == ["5'4"] and set(stats['sizes']) == {'small', 'xl'}
    assert stats['caps'] == 1                                         # SMALL is the only all-caps word
    assert text_utils.review_stats('LOVE this')['caps'] == 1


def test_phrases_find_adjective_noun_pairs():
    assert 'soft fabric' in text_utils.get_phrases('The soft fabric feels lovely')


def test_unhappy_review_is_flagged_and_happy_is_not():
    bad = services.analyze_review('Awful', UNHAPPY, 0.5)
    good = services.analyze_review('Wonderful', HAPPY, 0.5)
    assert bad.flagged and bad.risk == 'High' and bad.p_not > 0.8
    assert not good.flagged and good.risk == 'Low' and good.p_rec > 0.8


def test_threshold_changes_the_flag_but_not_the_probability():
    review = services.analyze_review('', UNHAPPY, 0.5)
    p = review.p_not
    review.threshold = 0.999
    assert review.p_not == p and not review.flagged


def test_explanation_matches_the_probability():
    review = services.analyze_review('', UNHAPPY, 0.5)
    assert review.terms.iloc[0]['effect'] > 0                       # strongest term pushes towards "not recommended"
    assert any(effect for _, effect in review.highlights)           # words are coloured in the text
    assert 'Fabric & quality' in review.themes and 'Returns' in review.themes


def test_review_without_words_is_rejected():
    try:
        services.analyze_review('', '!!! ... 123', 0.5)
    except ValueError as problem:
        assert 'No useful words' in str(problem)
    else:
        raise AssertionError('expected ValueError')


def test_heuristic_and_vader_agree_with_common_sense():
    assert services.warning_hits('Poor quality and cheap, returned it')
    assert not services.warning_hits('Lovely and comfortable')
    assert services.vader_compound('This is wonderful and I love it') > 0.5


def test_batch_marks_empty_reviews():
    table = services.predict_batch(['', ''], [UNHAPPY, '!!!'], 0.5)
    assert bool(table.loc[0, 'flagged']) and table.loc[1, 'risk'] == 'No text'


def test_metrics_at_counts():
    y = pd.Series([0, 0, 1, 1]).to_numpy()
    scores = pd.Series([0.9, 0.2, 0.6, 0.1]).to_numpy()
    m = metrics_at(y, scores, 0.5)
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (1, 1, 1, 1) and m['flagged'] == 2


def test_word_vectors_neighbours():
    table = services.neighbours('small', 5)
    assert len(table) == 5 and table['similarity'].is_monotonic_decreasing
