from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def term_document_matrix_show(df, max_features=20):
    # Преобразуем токены обратно в текст
    texts = df['tokens'].apply(lambda tokens: ' '.join(tokens))

    # Создаём терм-документную матрицу
    vectorizer = CountVectorizer(max_features=max_features)
    X_counts = vectorizer.fit_transform(texts)

    # Преобразуем в DataFrame для удобства
    terms = vectorizer.get_feature_names_out()
    tdm_df = pd.DataFrame(X_counts.toarray(), columns=terms)

    # Визуализируем heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(tdm_df.head(10), cmap="YlGnBu", annot=True, fmt="d")
    plt.title("Фрагмент терм-документной матрицы (первые 10 документов)")
    plt.xlabel("Термы")
    plt.ylabel("Документы")
    plt.tight_layout()
    plt.show()
