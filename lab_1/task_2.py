# 2. За допомогою регулярних виразів знайти і видалити такі помилки, як подвійні (або більше) пробіли, знаки пунктуації
# та зайві великі літери.
import re


def do(data):
    print(data)
    data = re.sub(r'[^\w\s]', ' ', data)
    data = re.sub(r'\s+', ' ', data)
    data = re.sub(r'([A-Z])([a-zA-Z]{1,100})', lambda x: x.group(1) + x.group(2).lower(), data)
    print(data)
    return data
