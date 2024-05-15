import nlp as nlp

from task_1 import Task1


# task 1
# t1 = Task1()
# t1.learn_model()
# # task.test_model("The movie Titanic won 11 Academy Awards.")
# t1.test_model("The movie Titanic won 11 Academy Awards, including Best Picture and Best Director. The album 21 by Adele "
#               "received six Grammy Awards. The TV series Game of Thrones won numerous Emmy Awards during its run. "
#               "The book The Great Gatsby was awarded the Pulitzer Prize for Fiction. The film La La Land received six "
#               "Academy Award nominations. The artist Taylor Swift has won several American Music Awards.")

# task 2

import random
import spacy
from spacy.training import Example
nlp = spacy.load("en_core_web_md")
trainset = [
    ("navigate home", {"entities": [(9,13, "GPE")]}),
    ("navigate to office", {"entities": [(12,18, "GPE")]}),
    ("navigate", {"entities": []}),
    ("navigate to Oxford Street", {"entities": [(12, 25, "GPE")]})]
epochs = 20
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.create_optimizer()
    for i in range(epochs):
        random.shuffle(trainset)
        for text, annotation in trainset:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotation)
            nlp.update([example], sgd=optimizer)

ner = nlp.get_pipe("ner")
ner.to_disk("new_ner")
doc = nlp("navigate to my house")
print(doc.ents)
