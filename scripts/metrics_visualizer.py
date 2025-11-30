import matplotlib.pyplot as plt

# Данные
model_name = "GNN (Chebyshev)"
metrics = ["Precision", "Recall", "F1", "F1 Micro AVG", "Accuracy"]
values = [0.966, 0.797, 0.873, 0.978, 0.978]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']  # разные цвета

# Создание графика
plt.figure(figsize=(6, 8))
bars = plt.barh(metrics, values, color=colors)

# Добавление значений рядом с полосками
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', va='center')

# Настройка графика
plt.xlim(0, 1.1)
plt.title(f'Метрики модели:\n{model_name}', fontsize=14)
plt.xlabel('Значение')
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Показать график
plt.tight_layout()
plt.show()
