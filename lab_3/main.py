import text_pre_processing as tpp
import bags
import task_2 as t2


with open('files/doc5.txt', 'r') as file:
    lines = []
    for line in file:
        lines.append(line.strip())
corpus = list(filter(None, lines))
# prepare
prepared_corpus = tpp.end_preprocessing(corpus)
print(prepared_corpus)
# 1
df = bags.bag_of_bigrams(prepared_corpus)
df.to_csv('files/ngrams.csv')
print("Vector for 'Roman architecture':")
print(bags.bigram_phrase_vector(prepared_corpus, "Roman architecture"))
# 2
# df1 = t2.to_tf_idf(prepared_corpus)
# print(df1)
# clusters = t2.clusterizate(prepared_corpus)
# for i, doc in enumerate(corpus):
#     print(f"Document {i+1} belongs to Cluster {clusters[i]}")
# 3

