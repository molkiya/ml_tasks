import numpy as np


# 1. Построение признаков через Word2Vec
def document_vector(model, tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)