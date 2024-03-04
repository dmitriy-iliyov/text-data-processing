# 2.Використати корпус Brown, десятий текст категорії adventure.
# а) Вивести речення, крім останнього;
# б) Видалити всі прикметники.

from nltk import *
from nltk.corpus import brown


def do():
    # a
    adventure_texts = brown.fileids(categories=['adventure'])
    tenth_adventure_words_list = brown.words(adventure_texts[9])
    tenth_adventure_text = ' '.join(tenth_adventure_words_list)
    tenth_adventure_sentence = sent_tokenize(tenth_adventure_text)
    print(tenth_adventure_sentence[-1] in tenth_adventure_sentence[:-1])
    for sentence in tenth_adventure_sentence[:-1]:
        print(sentence)

    # b
    
