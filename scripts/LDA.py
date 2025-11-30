import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from utils.lda_compress import preprocess
import spacy.cli
from utils.lda_dimension_normalizer import dimension_normalizer
from utils.lda_topic_classification_from_vectors import lda_topic_classification_from_vectors, LDA_CLASSIFIERS
from utils.metrics import lda_plot_metrics

# Загрузка необходимых данных
spacy.cli.download("en_core_web_sm")
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

# Загрузка данных
df = pd.read_csv(Path(__file__).parent.parent / "data/imdb/IMDB Dataset.csv").head(10000)  # Можно изменить на полный датасет
df['tokens'] = df['review'].apply(preprocess)
df['joined_tokens'] = df['tokens'].apply(lambda x: ' '.join(x))

# Преобразуем метки
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Обучение моделей Word2Vec
model_cbow = Word2Vec(sentences=df['tokens'], vector_size=550, window=5, min_count=2, sg=0)
model_sg = Word2Vec(sentences=df['tokens'], vector_size=550, window=5, min_count=2, sg=1)
model_sg_ns = Word2Vec(sentences=df['tokens'], vector_size=550, window=5, min_count=2, sg=1, negative=5)

# Основной запуск
X_cbow, X_sg, X_sg_ns = dimension_normalizer(df, model_cbow, model_sg, model_sg_ns)
print("Shape of X_cbow:", X_cbow.shape)
print("Shape of X_sg:", X_sg.shape)
print("Shape of X_sg_ns:", X_sg_ns.shape)

X_dimensions = [X_cbow, X_sg, X_sg_ns]

# LDA классификация для CBOW (с токенами)
for classifier in LDA_CLASSIFIERS:
    print(f"Classifier: {classifier}")
    cbow_metrics = lda_topic_classification_from_vectors(X_cbow, y, num_epochs=500, num_topics=550, name='CBOW',
                                                         classifier_type=classifier)
    print("cbow_metrics:", {k: len(v) for k, v in cbow_metrics.items()})
    lda_plot_metrics(cbow_metrics, f"{classifier}_CBOW")
    print(f"Classifier: {classifier}, CBOW is over")

    sg_metrics = lda_topic_classification_from_vectors(X_sg, y, num_epochs=500, num_topics=550, name='SG',
                                                       classifier_type=classifier)
    lda_plot_metrics(sg_metrics, f"{classifier}_SG")
    print(f"Classifier: {classifier}, SG is over")

    sg_ns_metrics = lda_topic_classification_from_vectors(X_sg_ns, y, num_epochs=500, num_topics=550, name='SG_NS',
                                                          classifier_type=classifier)
    lda_plot_metrics(sg_ns_metrics, f"{classifier}_SG_NS")
    print(f"Classifier: {classifier}, SG_NS is over")
