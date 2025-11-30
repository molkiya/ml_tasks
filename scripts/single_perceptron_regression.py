import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from utils.generate_time_series import generate_time_series
from utils.create_windows import create_windows
from utils.single_perceptron import SinglePerceptron
from utils.metrics import normalize

# Параметры временного ряда
a, b = -3, 3
m = 20

# Генерация временного ряда: x(t) = t^3 - 8t
t, x = generate_time_series(m, a=a, b=b)  # t: shape (m,), y: shape (m,)

# Визуализация временного ряда
plt.figure(figsize=(8, 6))
plt.plot(t, x, marker='o', color='b', label='x(t) = t^3 - 8t')
plt.title('Визуализация временного ряда')
plt.xlabel('Время (t)')
plt.ylabel('Значение (x(t))')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Формирование входов и целевых значений
x_norm = normalize(x)
p = 4
X, Y = create_windows(x_norm, p)

# Обучение
model_tanh = SinglePerceptron(p, activation='tanh')
model_sigmoid = SinglePerceptron(p, activation='sigmoid')

for model in [(model_tanh, 'tanh'), (model_sigmoid, 'sigmoid')]:
    model_name = model[1]
    model = model[0]
    errors = model.train(X, Y, epochs=5000, eta=0.1)

    # Предсказание и денормализация
    predictions = np.array([model.predict(X[i]) for i in range(len(X))])  # Преобразование в numpy array
    predictions = predictions * (np.max(x) - np.min(x)) + np.min(x)  # Денормализация

    # Визуализация
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(errors)
    plt.title(f'Эволюция ошибки MSE - ({model_name})')
    plt.xlabel('Эпохи')
    plt.ylabel('MSE')

    plt.subplot(1, 3, 2)
    plt.plot(t, x, 'o-', label='Реальные значения')
    plt.plot(t[p:], predictions, 'x-', label=f'Предсказания')
    plt.title(f'Сравнение ряда и предсказаний - ({model_name})')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(t[p:], x[p:] - predictions, 'r-')
    plt.title(f'Ошибки предсказания - ({model_name})')
    plt.tight_layout()
    plt.show()

    # Вывод значений
    print("\nРеальные значения vs Предсказания:")
    print("t\t\tРеальное\tПредсказание\tОшибка")
    for ti, real, pred in zip(t[p:], x[p:], predictions):
        print(f"{ti:.4f}\t{real:.6f}\t{pred:.6f}\t{real - pred:.6f}")
