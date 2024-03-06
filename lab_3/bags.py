import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(prepared_corpus):
    cv = CountVectorizer(min_df=0., max_df=1.)
    matrix = cv.fit_transform(prepared_corpus)
    return matrix.toarray()


def bag_of_ngrams(prepared_corpus):
    cv = CountVectorizer(ngram_range=(2, 2))
    matrix = cv.fit_transform(prepared_corpus)
    return matrix


def roman_architecture_vector(prepared_corpus):
    cv = CountVectorizer(ngram_range=(2, 2))
    cv.fit_transform(prepared_corpus)
    query = ["Roman architecture"]
    query_vec = cv.transform(query)
    return query_vec.toarray(), cv.vocabulary_


def to_data_frame(matrix, vocabulary):
    return pd.DataFrame(matrix, columns=vocabulary)
