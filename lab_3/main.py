import text_pre_processing as tpp
import bags
import pandas as pd

with open('files/doc5.txt', 'r') as file:
    lines = []
    for line in file:
        lines.append(line.strip())
corpus = list(filter(None, lines))
prepared_corpus = tpp.end_preprocessing(corpus)

print(prepared_corpus)

df = bags.bag_of_bigrams(prepared_corpus)
df.to_csv('files/ngrams.csv')

print("Vector for 'Roman architecture':")
print(bags.phrase_vector(prepared_corpus, "Roman architecture"))

