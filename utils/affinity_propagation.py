# Реализация Affinity Propagation
import numpy as np
import matplotlib.pyplot as plt

def affinity_propagation(dist_matrix,
                         damping=0.75,
                         max_iter=100,
                         pref_quantile=10):
    """
    dist_matrix: квадратная матрица манхэттеновских расстояний (n×n)
    damping: коэффициент дампинга [0.5–1.0)
    max_iter: число итераций
    pref_quantile: перцентиль похожести для preference (0–100)
    """
    # 1) similarity = –distance
    S = -dist_matrix.copy()

    # 2) preference на диагонали = перцентиль similarity
    pref = np.percentile(S, pref_quantile)
    np.fill_diagonal(S, pref)

    n = S.shape[0]
    R = np.zeros_like(S)
    A = np.zeros_like(S)

    # Для отладки — чтобы строить графики
    manh_r = []
    manh_a = []

    for it in range(max_iter):
        R_old, A_old = R.copy(), A.copy()

        # --- Update responsibilities R ---
        for i in range(n):
            # сумма availability+similarity по всем k
            tmp = A[i] + S[i]
            # для каждого k
            for k in range(n):
                # максимальное значение по j ≠ k
                max_others = np.max(np.delete(tmp, k))
                R[i, k] = S[i, k] - max_others

        # --- Update availabilities A ---
        for i in range(n):
            for k in range(n):
                if i == k:
                    A[i, k] = np.sum(np.maximum(0, R[:, k])) - R[k, k]
                else:
                    A[i, k] = min(
                        0,
                        R[k, k] +
                        np.sum(np.maximum(0, R[:, k])) -
                        np.maximum(0, R[i, k])
                    )

        # --- Damping ---
        R = (1 - damping) * R + damping * R_old
        A = (1 - damping) * A + damping * A_old

        # --- Считаем «манхэттеновские» изменения для мониторинга ---
        manh_r.append(np.sum(np.abs(R - R_old)))
        manh_a.append(np.sum(np.abs(A - A_old)))

    # График изменения R и A
    plt.figure(figsize=(10, 5))
    plt.plot(manh_r, label='Манх. расстояние R')
    plt.plot(manh_a, label='Манх. расстояние A')
    plt.xlabel('Итерация')
    plt.ylabel('Манх. расстояние')
    plt.title('Изменение R и A по итерациям')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Выбираем кластеры: для каждой i – argmax_k (R[i,k] + A[i,k])
    clusters = np.argmax(R + A, axis=1)
    return clusters


