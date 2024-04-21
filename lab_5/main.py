import os
import sys
from nltk.corpus import gutenberg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

import text_pre_processing as tpp

# prepare text
moby_dick = gutenberg.raw('melville-moby_dick.txt')
prepared_corpus = tpp.do_pre_processing(moby_dick[:10000])

print(prepared_corpus)

# task 1

# task 2
