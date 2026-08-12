import re
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textwrap import fill

LARGURA_TEXTO_PADRAO = 110
MAX_COLWIDTH_DF = 140


def wrap_texto(texto, largura=LARGURA_TEXTO_PADRAO):
    if pd.isna(texto):
        return ''
    return fill(str(texto), width=largura)


def _build_stopwords():
    nlp = spacy.load('pt_core_news_sm')
    stopwords_spacy = nlp.Defaults.stop_words
    stopwords_nltk = stopwords.words('portuguese')
    return set(stopwords_spacy.union(set(stopwords_nltk)))


STOPWORDS_SET = _build_stopwords()
_lemma = WordNetLemmatizer()


def ensure_text(text):
    return '' if pd.isna(text) else str(text)


def lowercase_text(text):
    return text.lower()


def remove_special_chars(text):
    # Preserva letras acentuadas e dígitos
    return re.sub(r'[-()"#/@;:<>{}`+=~|.!?,]', ' ', text)


def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords_and_lemmatize(text):
    tokens_limpos = []
    for word in text.split():
        if word not in STOPWORDS_SET:
            tokens_limpos.append(_lemma.lemmatize(word))
    return ' '.join(tokens_limpos)


TEXT_CLEANING_PIPELINE = (
    ensure_text,
    lowercase_text,
    remove_special_chars,
    normalize_whitespace,
    remove_stopwords_and_lemmatize,
)


def text_cleaning(text):
    for step in TEXT_CLEANING_PIPELINE:
        text = step(text)
    return text
