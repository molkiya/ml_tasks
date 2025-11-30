import numpy as np

def generate_time_series(m, a=-3, b=3):
    """
    Генерирует временной ряд на интервале [a, b] с m точками.
    Функция: x(t) = t^3 - 8t

    Parameters:
    - m: количество точек
    - a: начало интервала
    - b: конец интервала

    Returns:
    - t: массив временных точек (shape: [m])
    - y: значения функции в эти моменты времени (shape: [m])
    """
    t = np.linspace(a, b, m)
    y = t**3 - 8 * t

    return t, y
