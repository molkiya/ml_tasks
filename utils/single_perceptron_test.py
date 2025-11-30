import numpy as np
import matplotlib.pyplot as plt

# Генерация временного ряда
def generate_time_series(m, a=-3, b=3):
    t = np.linspace(a, b, m)
    y = t**3 - 8 * t
    return t, y

# Скользящее окно
def create_windows(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

# Однослойный персептрон
class SimplePerceptron:
    def __init__(self, input_size, activation='tanh', learning_rate=0.01, max_epochs=1000, epsilon=1e-4):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = epsilon

    def activate(self, x):
        return np.tanh(x) if self.activation == 'tanh' else 1 / (1 + np.exp(-x))

    def activate_derivative(self, output):
        return 1 - output ** 2 if self.activation == 'tanh' else output * (1 - output)

    def fit(self, X, y):
        for epoch in range(self.max_epochs):
            total_error = 0
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                output = self.activate(linear_output)
                error = target - output
                d_error = error * self.activate_derivative(output)
                self.weights += self.learning_rate * d_error * xi
                self.bias += self.learning_rate * d_error
                total_error += error ** 2
            if total_error < self.epsilon:
                print(f"Stopped at epoch {epoch}, total_error = {total_error:.6f}")
                break

    def predict(self, X):
        return np.array([self.activate(np.dot(xi, self.weights) + self.bias) for xi in X])

# Метрики
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def classification_metrics(y_true, y_pred, threshold=0):
    y_true_cls = [1 if y >= threshold else 0 for y in y_true]
    y_pred_cls = [1 if y >= threshold else 0 for y in y_pred]
    TP = sum(p == 1 and t == 1 for p, t in zip(y_pred_cls, y_true_cls))
    TN = sum(p == 0 and t == 0 for p, t in zip(y_pred_cls, y_true_cls))
    FP = sum(p == 1 and t == 0 for p, t in zip(y_pred_cls, y_true_cls))
    FN = sum(p == 0 and t == 1 for p, t in zip(y_pred_cls, y_true_cls))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    return TP, FP, FN, TN, accuracy, precision, recall

# Основной блок
m = 20
t, series = generate_time_series(m)

# Окна
window_size = 3
X_all, y_all = create_windows(series, window_size)

# Разделение на обучающую (13) и тестовую (7)
split_index = 13 - window_size
X_train, y_train = X_all[:split_index], y_all[:split_index]
X_test, y_test = X_all[split_index:], y_all[split_index:]

from sklearn.preprocessing import MinMaxScaler

scaler_y = MinMaxScaler(feature_range=(0, 1))  # для sigmoid
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Обучение
model_tanh = SimplePerceptron(input_size=window_size, activation='tanh', learning_rate=0.01, max_epochs=1000, epsilon=1e-4)
model_tanh.fit(X_train, y_train)

model_sigmoid = SimplePerceptron(input_size=window_size, activation='sigmoid', learning_rate=0.01, max_epochs=1000, epsilon=1e-4)
model_sigmoid.fit(X_train, y_train)

for model in [model_sigmoid, model_tanh]:
    # Предсказание
    # y_pred = model.predict(X_test)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    print(y_pred)

    # Метрики
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    TP, FP, FN, TN, acc, prec, rec = classification_metrics(y_test, y_pred)

    # Вывод
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")

    # Графики
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, series, label='Исходный ряд')
    plt.title('Временной ряд')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(y_test)), y_test, label='Истинные значения')
    plt.plot(range(len(y_pred)), y_pred, '--', label='Предсказания')
    plt.title('Предсказание vs Истина')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
