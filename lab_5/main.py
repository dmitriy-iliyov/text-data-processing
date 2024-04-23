import os
import re
import sys
import gensim
from nltk.corpus import gutenberg
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

import text_pre_processing as tpp

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def prepare_data_frame_for_task_1():
    with open('files/bbc-news-data.csv') as file:
        file_data = [line.strip() for line in file]
    df_column_names = file_data[0].split('\t')
    df_lines = [line.split('\t') for line in file_data[1:10]]
    return pd.DataFrame(df_lines, columns=df_column_names)


def pre_processing_df_for_task_1():
    data_df = prepare_data_frame_for_task_1()
    data_df['clear content'] = tpp.do_pre_processing(data_df['content'])
    return data_df


def pre_processing_data_for_task_2():
    moby_dick = gutenberg.raw('melville-moby_dick.txt')
    return tpp.do_pre_processing(moby_dick[:10000])


# task 1

# bigram = gensim.models.Phrases(norm_papers, min_count=20, threshold=20, delimiter='_')
# bigram_model = gensim.models.phrases.Phraser(bigram)
# norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]
# dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
# dictionary.filter_extremes(no_below=20, no_above=0.6)
# bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]

# task 2


