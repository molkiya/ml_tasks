import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

from utils.affinity_propagation import affinity_propagation
from utils.manhattan_distance import manhattan_distance

# 1. Загрузка и зашумление данных
iris = datasets.load_iris()
X = iris.data
y_true = iris.target

A = 1  # амплитуда шума
noise = A * np.random.rand(*X.shape)
X_noisy = X + noise

columns = iris.feature_names

df = pd.DataFrame(X_noisy, columns=columns)
df['target'] = y_true

X_selected = df.drop(df.columns[-1], axis=1)

print(X_selected)

# 2. Разделение на обучающую и тестовую выборки с сохранением классов
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y_true, test_size=0.33, random_state=42, stratify=y_true
)

# 3. Кластеризация на обучающей выборке
manhattan_train = manhattan_distance(X_train)
labels_train = affinity_propagation(manhattan_train, damping=0.75, max_iter=100)

print(labels_train)

print("Уникальные кластеры в обучающей выборке:", np.unique(labels_train))

# 4. Назначение кластеров тестовым точкам (через ближайший центр)
unique_clusters = np.unique(labels_train)
cluster_centers = np.array([
    np.mean(X_train[labels_train == cluster], axis=0) for cluster in unique_clusters
])


def assign_to_nearest_cluster(X, centers):
    distances = np.array([[np.sum(np.abs(x - c)) for c in centers] for x in X])
    return np.argmin(distances, axis=1)


labels_test = assign_to_nearest_cluster(X_test, cluster_centers)

# 6. Визуализация кластеров на тестовой части
df_test = pd.DataFrame(X_test, columns=columns)
df_test['Cluster'] = labels_test


# Убедимся, что у нас есть хотя бы 3 признака
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Три признака: columns[0], columns[1], columns[2] — можно указать свои
x = df_test[columns[0]]
y = df_test[columns[1]]
z = df_test[columns[2]]
c = df_test['Cluster']

# Отрисовка точек
sc = ax.scatter(x, y, z, c=c, cmap='tab10', s=50)

ax.set_xlabel(columns[0])
ax.set_ylabel(columns[1])
ax.set_zlabel(columns[2])
ax.set_title('Кластеры на тестовой выборке (3D)')

# Цветовая легенда
legend1 = ax.legend(*sc.legend_elements(), title="Кластер", loc="best")
ax.add_artist(legend1)

plt.tight_layout()
plt.show()
