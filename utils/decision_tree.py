import numpy as np

# Реализация дерева решений
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    @staticmethod
    def entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        # 𝑆 = − ∑ (𝑖 от 1 до N) 𝑝𝑖 log2 𝑝𝑖, где:
        # 𝑝𝑖 – вероятность нахождения системы в 𝑖-м состоянии,
        # N – количество возможных состояний системы.
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain(self, X, y, threshold):
        parent_entropy = self.entropy(y)
        left_y, right_y = y[X <= threshold], y[X > threshold]
        n, n_left, n_right = len(y), len(left_y), len(right_y)
        return parent_entropy - (n_left / n) * self.entropy(left_y) + (n_right / n) * self.entropy(right_y)

    def best_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, 0
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            # 𝐼𝐺(𝑄) = 𝑆0 (parent_entropy) − ∑ (𝑖 от 1 до q) 𝑁𝑖/𝑁 * 𝑆𝑖, где:
            # S0 – энтропия всей системы,
            # q – число групп разбиения,
            # 𝑁𝑖 - число элементов выборки, у которых признак Q имеет i-е значение.
            for threshold in thresholds:
                gain = self.information_gain(X[:, feature_idx], y, threshold)
                # На каждом шаге выбирается тот признак, при разделении по которому прирост информации оказывается наибольшим
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature_idx, threshold, gain
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return np.bincount(y).argmax()

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return np.bincount(y).argmax()

        left_mask, right_mask = X[:, feature] <= threshold, X[:, feature] > threshold
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, sample, tree):
        if isinstance(tree, dict):
            if sample[tree['feature']] <= tree['threshold']:
                return self.predict_sample(sample, tree['left'])
            else:
                return self.predict_sample(sample, tree['right'])
        return tree

    def predict(self, X):
        return np.array([self.predict_sample(sample, self.tree) for sample in X])