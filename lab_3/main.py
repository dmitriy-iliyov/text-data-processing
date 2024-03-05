import text_pre_processing as tpp

with open('files/doc5.txt', 'r') as file:
    lines = []
    for line in file:
        lines.append(line.strip())
corpus = list(filter(None, lines))
prepared_corpus = tpp.end_preprocessing(corpus)
print(prepared_corpus)




