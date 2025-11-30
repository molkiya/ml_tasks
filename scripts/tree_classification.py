import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils.decision_tree import DecisionTree
from utils.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

# 1. Загрузка набора данных "Ирисы Фишера"
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Фильтруем только два класса (бинарная классификация)
mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]

# Добавляем белый шум A * rnd(0,1), где A = 5
A = 5
noise = A * np.random.rand(*X.shape)
X_noisy = X + noise

# Преобразуем в DataFrame
columns = iris.feature_names
df = pd.DataFrame(X_noisy, columns=columns)
df['target'] = y

# Построение корреляционной диаграммы
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')
plt.show()

# Выбираем два признака с наибольшей корреляцией
# Получаем 2 признака, наиболее коррелирующих с 'target' (без неё самой)
top_features = df.corr().abs().nlargest(3, 'target').index.drop('target')

# Добавляем явно 'sepal width (cm)' если его нет
if 'petal width (cm)' not in top_features:
    selected_features = list(top_features[:1]) + ['petal width (cm)']
else:
    selected_features = list(top_features[:2])

# Добавляем 'target' в конец
selected_features.append('target')

X_selected = df[selected_features]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.4, random_state=42)

print(X_train, X_test)

# 2. Обучение дерева решений
clf = DecisionTree(max_depth=15)
clf.fit(X_train.to_numpy(), y_train)

# Предсказание
y_pred = clf.predict(X_test.to_numpy())

# 3. Оценка модели (По эпохам)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print('Confusion Matrix:\n', conf_matrix)

# Визуализация матрицы ошибок
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Setosa', 'Versicolor'],
            yticklabels=['Setosa', 'Versicolor'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()