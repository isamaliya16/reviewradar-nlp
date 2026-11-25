"""Page 4: explore the Word2Vec (Skip-gram) word vectors."""
from difflib import get_close_matches

import streamlit as st

import ui
from config import COLORS, NEIGHBOURS_DEFAULT
from services import embedding_map, load_word_vectors, neighbours


def render():
    ui.hero('Word explorer', 'Words that live close together',
            'The Skip-gram model learned a vector for every word from the reviews. Type a word to see its neighbours: '
            'words customers use in the same situations.')
    vectors = load_word_vectors()
    if vectors is None:
        ui.empty_state('🧬', 'The word-vector model could not be loaded (gensim or word2vec_skipgram.model is missing).')
        return
    ui.kpi_cards([('Words in the vocabulary', len(vectors), 'seen often enough in training reviews', 'teal', ''),
                  ('Numbers per word', vectors.vector_size, 'dimensions of each vector', 'good', '')])

    left, right = st.columns([3, 2], vertical_alignment='bottom')
    word = left.text_input('Type a word', key='w_word', placeholder='a word from a fashion review').strip().lower()
    amount = right.slider('Neighbours', 3, 30, NEIGHBOURS_DEFAULT)
    if not word:
        ui.empty_state('✍️', 'Type a word to explore its neighbourhood.')
    elif word not in vectors:
        st.warning(f'“{word}” is not in the vocabulary.')
        close = get_close_matches(word, vectors.index_to_key, n=6)
        if close:
            st.html('<div class="pills">Did you mean ' + ''.join(ui.pill(c, 'teal') for c in close) + '</div>')
    else:
        table = neighbours(word, amount)
        a, b = st.columns([2, 3])
        with a:
            st.markdown(f'**Closest to “{word}”**')
            ui.meters([(w, s) for w, s in zip(table['word'], table['similarity'])], lambda v: 'blue')
        with b:
            st.markdown('**Map of the neighbourhood** (2-D view of the vectors)')
            ui.word_map(embedding_map(word, amount))
        st.caption('Closer on the map and a higher similarity mean the model saw the words in similar contexts.')

    ui.section('How similar are two words?')
    one, two = st.columns(2)
    first = one.text_input('First word', key='w_first').strip().lower()
    second = two.text_input('Second word', key='w_second').strip().lower()
    if first and second:
        missing = [w for w in (first, second) if w not in vectors]
        if missing:
            st.warning('Not in the vocabulary: ' + ', '.join(missing))
        else:
            score = float(vectors.similarity(first, second))
            ui.kpi_cards([(f'“{first}” ↔ “{second}”', max(score, 0) * 100, 'cosine similarity', 'good' if score > .5 else 'warn', '%')])
