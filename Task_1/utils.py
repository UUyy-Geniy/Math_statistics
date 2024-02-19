import numpy as np

types = ['normal', 'cauchy', 'student_t', 'poisson', 'uniform']


def get_distribution(name, size):
    if name == 'normal':
        return np.random.normal(0, 1, size)
    elif name == 'cauchy':
        return np.random.standard_cauchy(size)
    elif name == 'student_t':
        return np.random.standard_t(3, size)
    elif name == 'poisson':
        return np.random.poisson(lam=10, size=size)
    elif name == 'uniform':
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), size)
    else:
        return []


def get_density(name, array):
    if name == 'normal':
        return [1 / (np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2) for x in array]
    elif name == 'cauchy':
        return [1 / (np.pi * (x ** 2 + 1)) for x in array]
    elif name == 'student_t':
        return [(np.math.gamma((3 + 1) / 2) / (np.sqrt(3 * np.pi) * np.math.gamma(3 / 2)) *
                 (1 + x ** 2 / 3) ** (-((3 + 1) / 2))) for x in array]
    elif name == 'poisson':
        return [10 ** float(x) * np.exp(-10) / np.math.gamma(x+1) for x in array]
    elif name == 'uniform':
        return [1 / (2 * np.sqrt(3)) if abs(x) <= np.sqrt(3) else 0 for x in array]
    else:
        return []


def compute_trimmed_mean(array):
    r = int(len(array) / 4)
    sorted_array = np.sort(array)
    summ = 0.0
    for i in range(r + 1, len(array) - r):
        summ += sorted_array[i]
    return summ / (len(array) - 2 * r)
