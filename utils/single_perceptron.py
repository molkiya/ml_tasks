import numpy as np


class SinglePerceptron:
    def __init__(self, p, activation='tanh'):
        self.weights = np.random.randn(p + 1) * 0.1
        self.activation = activation

    def predict(self, X):
        net = np.dot(X, self.weights[1:]) + self.weights[0]
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-net))
        elif self.activation == 'tanh':
            return np.tanh(net)
        else:
            return net

    def train(self, X, Y, epochs=1000, eta=0.01):
        errors = []
        for epoch in range(epochs):
            error = 0
            for xi, yi in zip(X, Y):
                pred = self.predict(xi)
                delta = yi - pred
                self.weights[1:] += eta * delta * xi
                self.weights[0] += eta * delta
                error += delta ** 2
            errors.append(error / len(Y))
            if epoch > 100 and abs(errors[-2] - errors[-1]) < 1e-6:
                break
        return errors
