import gensim.downloader as api
import numpy as np

# Load pre-trained GloVe model
model = api.load("glove-wiki-gigaword-50")

def get_synonyms(word, model, top_n=3):
    if word in model:
        return [synonym for synonym, _ in model.most_similar(word, topn=top_n)]
    else:
        return []

def extend_text_with_synonyms(text, model, top_n=3):
    words = text.split()
    extended_text = []
    
    for word in words:
        extended_text.append(word)
        synonyms = get_synonyms(word, model, top_n)
        extended_text.extend(synonyms)
    
    return " ".join(extended_text)

print(extend_text_with_synonyms("football goal", model))