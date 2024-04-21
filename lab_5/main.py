import nltk
from nltk.corpus import gutenberg
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

import text_pre_processing as tpp

nltk.download('gutenberg')
nltk.download('punkt')
moby_dick = gutenberg.raw('melville-moby_dick.txt')

print(moby_dick[:1000])

prepared_corpus = tpp.end_p
