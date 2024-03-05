import numpy as np
from nltk import *
from nltk.corpus import stopwords


def start_preprocessing(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    wpt = WordPunctTokenizer()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    doc = ' '.join(filtered_tokens)
    return doc


def end_preprocessing(doc):
    prepared_corpus = np.vectorize(start_preprocessing)(corpus)
    return prepared_corpus

