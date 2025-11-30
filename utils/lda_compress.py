import re
import spacy
import nltk
from nltk.corpus import stopwords

# Подготовка
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_sm')


def preprocess(text):
    # Убираем HTML теги
    text = re.sub(r'<.*?>', '', text)  # Убираем все теги HTML

    # Приводим к нижнему регистру
    text = text.lower()

    # Токенизация и удаление стоп-слов
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    return tokens
