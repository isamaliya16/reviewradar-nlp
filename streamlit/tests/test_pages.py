"""Run every page headlessly and type our own reviews."""
from streamlit.testing.v1 import AppTest

MIXED = 'Cute top, but the sleeves are tight and I might return it.'
UNHAPPY = 'The seams ripped on the first wear, the material is thin and cheap. Total waste of money, returning it.'
HAPPY = 'I adore this sweater. So cozy, flattering and the color is stunning. Highly recommend!'


def page(module):
    return AppTest.from_string(f'from views.{module} import render\nrender()', default_timeout=240).run()


def submit(at, title, text):
    at.text_input(key='rv_title').set_value(title)
    at.text_area(key='rv_text').set_value(text)
    [b for b in at.button if b.label == 'Analyse'][0].click().run()
    return at


def test_empty_review_asks_for_text():
    at = page('analyze')
    [b for b in at.button if b.label == 'Analyse'][0].click().run()
    assert at.warning and 'enter a review' in at.warning[0].value


def test_unhappy_review_goes_to_quality_team():
    at = submit(page('analyze'), 'Terrible', UNHAPPY)
    assert not at.exception
    assert at.session_state['analysis'].flagged and at.session_state['history'][0]['flagged']


def test_happy_review_is_not_flagged():
    at = submit(page('analyze'), 'Love it', HAPPY)
    assert not at.session_state['analysis'].flagged


def test_second_submit_uses_the_new_text():
    at = submit(page('analyze'), 'First', UNHAPPY)
    at = submit(at, 'Second', HAPPY)
    assert at.session_state['analysis'].text == HAPPY and len(at.session_state['history']) == 2


def test_sidebar_threshold_changes_the_verdict():
    flagged_by_default = submit(page('analyze'), 'Mixed', MIXED).session_state['analysis'].flagged
    at = page('analyze')
    at.session_state['threshold'] = 0.9
    not_flagged_when_strict = not submit(at, 'Mixed', MIXED).session_state['analysis'].flagged
    assert flagged_by_default and not_flagged_when_strict


def test_random_real_review_fills_the_form():
    at = page('analyze')
    [b for b in at.button if 'random' in b.label][0].click().run()
    assert at.text_area(key='rv_text').value.strip() and 'rv_truth' in at.session_state


def test_batch_scan_with_typed_reviews():
    at = page('batch')
    at.text_area(key='batch_typed').set_value(f'{UNHAPPY}\n{HAPPY}\n@@@ ###')
    [b for b in at.button if b.label == 'Run scan'][0].click().run()
    result = at.session_state['batch_result']
    assert not at.exception and len(result) == 3
    assert bool(result.loc[0, 'flagged']) and not bool(result.loc[1, 'flagged']) and result.loc[2, 'risk'] == 'No text'


def test_batch_without_reviews_warns():
    at = page('batch')
    [b for b in at.button if b.label == 'Run scan'][0].click().run()
    assert at.warning


def test_insights_model_lab_and_about_render():
    for module in ('insights', 'model_lab', 'about'):
        assert not page(module).exception, module


def test_word_explorer_known_and_unknown_words():
    at = page('words')
    at.text_input(key='w_word').set_value('small').run()
    assert not at.exception and not at.warning
    at.text_input(key='w_word').set_value('zzzzqqq').run()
    assert at.warning and 'not in the vocabulary' in at.warning[0].value


def test_history_empty_state():
    assert not page('history').exception
