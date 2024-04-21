import numpy as np
import pandas as pd
from nltk import *
from nltk.corpus import stopwords
import re


def start_pre_processing(doc):
    doc = re.sub(r'http[s]?://\S+|www\.\S+', '', doc)
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    wpt = WordPunctTokenizer()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    doc = ' '.join(filtered_tokens)
    return doc


def do_pre_processing(doc):
    if isinstance(doc, pd.Series):
        prepared_corpus = doc.apply(lambda x: start_pre_processing(x))
        return prepared_corpus
    elif isinstance(doc, str):
        sentences = doc.split('.')
        prepared_corpus = [start_pre_processing(sentence) for sentence in sentences]
        prepared_corpus = ' '.join(list(filter(None, prepared_corpus)))
        return prepared_corpus

