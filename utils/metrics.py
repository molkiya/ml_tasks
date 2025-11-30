import matplotlib.pyplot as plt
import numpy as np


# Реализация accuracy
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# Реализация precision
def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0


# Реализация recall
def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0


# Реализация confusion matrix
def confusion_matrix(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[true_negative, false_positive],
                     [false_negative, true_positive]])

def evaluate_metrics(y_true, y_pred):
    # Получаем confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Вычисляем метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Confusion Matrix': (tn, fp, fn, tp)
    }

def mean_squared_error(y_true, y_pred):
    # Проверим, что входные данные являются массивами NumPy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Вычислим квадрат разницы между реальными и предсказанными значениями
    squared_errors = (y_true - y_pred) ** 2

    # Среднее значение квадратичных ошибок
    mse = np.mean(squared_errors)

    return mse

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def mean_absolute_error(y_true, y_pred):
    # Проверяем, что длина массивов одинакова
    if len(y_true) != len(y_pred):
        raise ValueError("Длины входных списков должны совпадать")

    # Вычисляем абсолютные ошибки
    absolute_errors = [abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)]

    # Возвращаем среднее значение абсолютных ошибок
    return sum(absolute_errors) / len(absolute_errors)



def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae


def lda_plot_metrics(metrics_dict, model_name):
    epochs = metrics_dict['epoch']

    fig, ax = plt.subplots(1, 3, figsize=(21, 6))

    # 1. Confusion Matrix (последняя эпоха)
    tp = metrics_dict['TP'][-1]
    fp = metrics_dict['FP'][-1]
    fn = metrics_dict['FN'][-1]
    tn = metrics_dict['TN'][-1]

    matrix = np.array([[tp, fn],
                       [fp, tn]])

    ax[0].imshow(matrix, cmap='Blues')
    ax[0].set_title(f"{model_name} — Confusion Matrix (Final Epoch)")
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(['Positive', 'Negative'])
    ax[0].set_yticklabels(['Pred Positive', 'Pred Negative'])
    ax[0].set_xlabel("Actual Label")
    ax[0].set_ylabel("Predicted Label")

    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, matrix[i, j], ha='center', va='center', color='black', fontsize=16)

    # 2. Графики метрик по эпохам
    colors = {
        'Accuracy': '#4c72b0', 'Precision': '#55a868', 'Recall': '#c44e52',
        'F1': '#8172b2', 'ROC_AUC': '#ccb974'
    }

    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        ax[1].plot(epochs, metrics_dict[metric], label=metric, color=colors[metric], marker='o')

    ax[1].set_title(f"{model_name} — Classification Metrics by Epoch")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Metric Value")
    ax[1].legend()
    ax[1].grid(True)

    # 3. ROC AUC по эпохам
    ax[2].plot(epochs, metrics_dict['ROC_AUC'], color=colors['ROC_AUC'], marker='o')
    ax[2].set_ylim([0, 1])
    ax[2].set_title(f"{model_name} — ROC AUC by Epoch")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("ROC AUC")
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()



