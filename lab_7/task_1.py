import random

import spacy
from spacy.training import Example


class Task1:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.trainset = self.add_trainset()

    def learn_model(self):
        ner = self.nlp.get_pipe("ner")
        ner.add_label("AWARD")

        epochs = 20
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.create_optimizer()
            for i in range(epochs):
                random.shuffle(self.trainset)
                for text, annotation in self.trainset:
                    doc = self.nlp(text)
                    example = Example.from_dict(doc, annotation)
                    self.nlp.update([example], sgd=optimizer)
            self.nlp.to_disk("new_ner")

    def add_trainset(self):
        return [
            ("The movie Inception won the Oscar for Best Cinematography.", {"entities": [(25, 31, "AWARD")]}),
            ("The album Thriller by Michael Jackson received eight Grammy Awards.", {"entities": [(54, 67, "AWARD")]}),
            ("The TV series Breaking Bad won multiple Emmy Awards.", {"entities": [(39, 51, "AWARD")]}),
            ("The book To Kill a Mockingbird was awarded the Pulitzer Prize.", {"entities": [(54, 68, "AWARD")]}),
            ("The film Schindler's List won seven Academy Awards.", {"entities": [(39, 55, "AWARD")]}),
            ("The artist Beyonce has won numerous MTV Video Music Awards.", {"entities": [(40, 63, "AWARD")]})
        ]

    def test_model(self, text):
        doc = self.nlp(text)
        for ent in doc.ents:
            print(ent.text, ent.label_)
