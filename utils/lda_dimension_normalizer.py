import numpy as np


def document_vector(model, tokens):
    valid_tokens = [token for token in tokens if token in model.wv]
    if valid_tokens:
        return np.mean(model.wv[valid_tokens], axis=0)
    else:
        print(f"No valid tokens found for tokens: {tokens}")
        return np.zeros(model.vector_size)  # model.vector_size should be 300 for CBOW embeddings



# Для всех трёх моделей:
def dimension_normalizer(df, model_cbow, model_sg, model_sg_ns):
    print("Starting transformation of tokens to vectors...")
    df['vec_cbow'] = df['tokens'].apply(lambda x: document_vector(model_cbow, x))
    print(f"Model vector size: {model_cbow.vector_size}")

    # Check the shape of the first few 'vec_cbow' vectors to ensure they are 1D arrays of length 300
    print(f"Shape of the first few 'vec_cbow' vectors before stacking: {[vec.shape for vec in df['vec_cbow'].head()]}")

    # Explicitly convert each 'vec_cbow' vector to a NumPy array, then stack them
    X_topics = np.vstack([np.array(vec) for vec in df['vec_cbow']])

    # Check the shape of X_topics after stacking
    print(f"X_topics shape after stacking: {X_topics.shape}")

    # Print the first few entries of the 'vec_cbow' column to check the vectors
    print(f"Sample 'vec_cbow' after transformation:\n{df['vec_cbow'].head()}")

    df['vec_sg'] = df['tokens'].apply(lambda x: document_vector(model_sg, x))
    df['vec_sg_ns'] = df['tokens'].apply(lambda x: document_vector(model_sg_ns, x))

    # Преобразуем в матрицы признаков
    # Convert to numpy array for model training
    print("Stacking vec_cbow into X_topics array...")
    X_cbow = np.vstack(df['vec_cbow'].apply(np.array).values)
    # Print the shape of X_topics to ensure it's correct
    print(f"X_topics shape: {X_cbow.shape}")
    X_sg = np.vstack(df['vec_sg'].values)
    X_sg_ns = np.vstack(df['vec_sg_ns'].values)

    return X_cbow, X_sg, X_sg_ns
