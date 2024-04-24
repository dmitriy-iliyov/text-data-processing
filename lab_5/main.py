import os
import sys
from operator import itemgetter

import gensim
import nltk
import pandas as pd
from nltk.collocations import TrigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.corpus import gutenberg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

import text_pre_processing as tpp

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def prepare_data_frame_for_task_1():
    with open('files/bbc-news-data.csv') as file:
        file_data = [line.strip() for line in file]
    df_column_names = file_data[0].split('\t')
    df_lines = [line.split('\t') for line in file_data[1:100]]
    return pd.DataFrame(df_lines, columns=df_column_names)


def pre_processing_df_for_task_1():
    data_df = prepare_data_frame_for_task_1()
    data_df['clear content'] = tpp.do_pre_processing(data_df['content'])
    return data_df


def pre_processing_data_for_task_2(doc):
    return tpp.do_pre_processing(doc[:10000])


def compute_ngrams(sequence, n):
    return list(zip(*(sequence[index:] for index in range(n))))


def get_top_ngrams(corpus, ngram_val=1, limit=5):
    tokens = nltk.word_tokenize(corpus)
    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq) for text, freq in sorted_ngrams]
    return sorted_ngrams


# task 1
df = pre_processing_df_for_task_1()
words_from_clear_content = [nltk.word_tokenize(line) for line in df['clear content'].values]
bigram = gensim.models.Phrases(words_from_clear_content, min_count=5, threshold=5, delimiter='-')
bigram_model = gensim.models.phrases.Phraser(bigram)
norm_corpus_bigrams = [bigram_model[doc] for doc in df['clear content']]
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
dictionary.filter_extremes(no_below=20, no_above=0.6)
bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
Total_topics = 10
lsi_bow = gensim.models.LsiModel(bow_corpus, id2word=dictionary, num_topics=Total_topics, onepass=True, chunksize=1740, power_iters=1000)
for topic_id, topic in lsi_bow.print_topics(num_topics=10, num_words=20):
    print('Topic #'+str(topic_id+1)+':')
    print(topic)

# task 2
moby_dick = gutenberg.raw('melville-moby_dick.txt')
prepared_corpus_1 = pre_processing_data_for_task_2(moby_dick)
top_10_trigram_1 = get_top_ngrams(corpus=prepared_corpus_1, ngram_val=3, limit=10)
print(top_10_trigram_1)

sentences = moby_dick[:10000].split('.')
prepared_corpus_2 = [nltk.word_tokenize(tpp.start_pre_processing(sentence)) for sentence in sentences]

finder = TrigramCollocationFinder.from_documents(prepared_corpus_2)
trigram_measures = TrigramAssocMeasures()
top_10_trigram_2 = finder.nbest(trigram_measures.raw_freq, 10)
print([' '.join(trigram) for trigram in top_10_trigram_2])


