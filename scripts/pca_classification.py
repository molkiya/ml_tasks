import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_curve, auc
)

project_root = Path(__file__).parent.parent
os.makedirs(project_root / 'models', exist_ok=True)

# 1. Загрузка данных
data = pd.read_csv(project_root / 'data/fraud/creditcard.csv')
print("Первые семь строк набора данных:")
print(data.head(7))

# 2. Выделение признаков и целевой переменной
X = data.drop('Class', axis=1)
y = data['Class']

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (7 компонент)
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_scaled)

results = []

# Основной цикл
for i, x_data in enumerate([X, X_pca]):
    compressed = i == 1
    print('compressed', compressed)
    title_suffix = "with PCA" if compressed else "without PCA"

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(x_data, y, train_size=0.5, random_state=42)

    # Стандартизация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Truncated SVD
    svd = TruncatedSVD(n_components=6, random_state=42)
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)

    # --- GaussianNB ---
    print("\n--- GaussianNB ---")
    nb = GaussianNB()
    nb.fit(X_train_svd, y_train)
    y_pred_nb = nb.predict(X_test_svd)

    # --- DecisionTreeClassifier ---
    print("\n--- DecisionTreeClassifier ---")
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train_svd, y_train)
    y_pred_tree = tree.predict(X_test_svd)

    # Метрики
    precision_nb = precision_score(y_test, y_pred_nb)
    recall_nb = recall_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb)

    precision_tree = precision_score(y_test, y_pred_tree)
    recall_tree = recall_score(y_test, y_pred_tree)
    f1_tree = f1_score(y_test, y_pred_tree)

    # ROC и AUC
    y_scores_nb = nb.predict_proba(X_test_svd)[:, 1]
    y_scores_tree = tree.predict_proba(X_test_svd)[:, 1]

    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_scores_nb)
    roc_auc_nb = auc(fpr_nb, tpr_nb)

    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_scores_tree)
    roc_auc_tree = auc(fpr_tree, tpr_tree)

    # Сравнение
    comparison_df = pd.DataFrame({
        'Model': ['GaussianNB', 'DecisionTree'],
        'Precision': [precision_nb, precision_tree],
        'Recall': [recall_nb, recall_tree],
        'F1-Score': [f1_nb, f1_tree],
        'AUC': [roc_auc_nb, roc_auc_tree]
    })

    results.append(
        [
            [fpr_nb, tpr_nb, roc_auc_nb],
            [fpr_tree, tpr_tree, roc_auc_tree],
            [comparison_df, compressed],
            [
                [nb, X_train_svd, y_train, y_test, y_pred_nb],
                [tree, X_train_svd, y_train, y_test, y_pred_tree]
            ]
        ]
    )

# --- Визуализация и метрики ---
plt.figure(figsize=(8, 6))

for compress_result in results:
    comparison_df = compress_result[2][0]
    print("\nСравнение моделей по метрикам:")
    print(comparison_df.round(3))
    compressed = compress_result[2][1]

    # ROC-кривые
    plt.plot(compress_result[0][0], compress_result[0][1],
             color='green' if compressed else 'blue',
             label=f'{"PCA: " if compressed else ""}GaussianNB (AUC = {compress_result[0][2]:.2f})')

    plt.plot(compress_result[1][0], compress_result[1][1],
             color='cyan' if compressed else 'red',
             label=f'{"PCA: " if compressed else ""}DecisionTree (AUC = {compress_result[1][2]:.2f})')

    # Кросс-валидация и отчёты
    for model_result in compress_result[3]:
        estimator, X_train_svd, y_train, y_test, y_pred = model_result
        model_name = type(estimator).__name__
        cv_scores = cross_val_score(estimator, X_train_svd, y_train, cv=5, scoring='f1')

        print(f"\nF1-оценка по кросс-валидации ({model_name}): {cv_scores}")
        print("Среднее значение F1:", np.mean(cv_scores))
        print(f"Размерность после SVD: {X_train_svd.shape[1]} признаков")
        print(f"Отчет классификации ({model_name}):")
        print(classification_report(y_test, y_pred))
        print("Матрица ошибок:")
        print(confusion_matrix(y_test, y_pred))

# Финальный график
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC-кривые: GaussianNB и DecisionTree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
