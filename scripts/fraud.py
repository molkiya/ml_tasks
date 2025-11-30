import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Загрузка и первичный анализ данных
data = pd.read_csv(Path(__file__).parent.parent / 'data/fraud/creditcard.csv')

# Вывод первых семи строк
print("Первые семь строк набора данных:")
print(data.head(7))

# Статистическая сводка по столбцам
print("\nСтатистическая сводка по столбцам:")
print(data.describe())

# Количество мошеннических транзакций
fraud_count = data['Class'].sum()
print(f"\nКоличество мошеннических транзакций: {fraud_count}")

# 2. Визуализация данных
# Корреляционная матрица
plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Корреляционная матрица')
plt.show()

# # # График парных отношений (pairplot) для выбранных признаков
selected_features = ['V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'Amount', 'Class']
# sns.pairplot(data[selected_features], hue='Class', palette='Set1', diag_kind='kde')
# plt.suptitle('График парных отношений для признаков V6-V15, Amount и Class', y=1.02)
# plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data, palette='Set1')
plt.yscale('log')
plt.title('Распределение классов (логарифмическая шкала)')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.xticks([0, 1], ['Legal', 'Fraud'])
plt.show()

# 3. Подготовка данных для моделирования
# Выбор столбцов V6…V15, Amount и Class
data_selected = data[selected_features]

# Разделение данных на признаки и целевую переменную
X = data_selected.drop('Class', axis=1)
y = data_selected['Class']

# Разделение на обучающую и тестовую выборки (50% на обучение)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.35, random_state=42, stratify=y)

# 4. Построение и оценка моделей
models = {
    'GaussianNB': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

for model_name, model in models.items():
    # Обучение модели
    print(f"\nМодель: {model_name}")
    model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Вероятности для класса 1

    # Оценка модели
    print("Классификационный отчет:")
    print(classification_report(y_test, y_pred))

    # Вычисление матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок:")
    print(cm)

    # Визуализация матрицы ошибок с помощью heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legal', 'Fraud'], yticklabels=['Legal', 'Fraud'])
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title(f'Матрица ошибок для модели: {model_name}')
    plt.show()

    # ROC кривая
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Визуализация ROC кривой
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ложноположительная скорость (FPR)')
    plt.ylabel('Истинноположительная скорость (TPR)')
    plt.title(f'ROC кривая для модели: {model_name}')
    plt.legend(loc='lower right')
    plt.show()
