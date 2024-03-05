import math
import numpy as np
import utils as c
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#INTERVAL CONSTANTS
SIGNIFICANT = 0.05
CONFIDENTIAL_LEVEL = 1 - SIGNIFICANT/2


def get_interval(name: str, fun, *args):
    data = {"name": name,
            "n": [],
            "mean_min": [],
            "mean": [],
            "mean_max": [],
            "dev_min": [],
            "dev": [],
            "dev_max": []}
    figure, axes = plt.subplots(1, 2 * 2, figsize=(20, 10))
    for index, amount in enumerate([20, 100]):
        sample = fun(*args, size=amount)

        mean = sample.mean()
        dev = np.std(sample)

        student_value = scipy.stats.t.ppf(CONFIDENTIAL_LEVEL, amount - 1)
        chi_square_value_min = scipy.stats.chi2.ppf(CONFIDENTIAL_LEVEL, amount - 1)
        chi_square_value_max = scipy.stats.chi2.ppf(SIGNIFICANT / 2, amount - 1)

        mean_min = mean - dev * student_value / math.sqrt(amount - 1)
        mean_max = mean + dev * student_value / math.sqrt(amount - 1)

        dev_min = math.sqrt(amount) * dev / math.sqrt(chi_square_value_min)
        dev_max = math.sqrt(amount) * dev / math.sqrt(chi_square_value_max)

        data['n'].append(amount)
        data['mean_min'].append(mean_min)
        data['mean_max'].append(mean_max)
        data['dev_min'].append(dev_min)
        data['dev_max'].append(dev_max)
        data['mean'].append(mean)
        data['dev'].append(dev)

        sns.histplot(sample, ax=axes[index], stat='density', color='#FFEFD0')
        axes[index].axvline(x=mean_min, color='#D400C9', linewidth=2)
        axes[index].axvline(x=mean_max, color='#D400C9', linewidth=2)
        axes[index].axvline(x=mean_min - dev_max, color='#0007FF', linewidth=2)
        axes[index].axvline(x=mean_max + dev_max, color='#0007FF', linewidth=2)

    sns.lineplot(x='x', y='y', data={
        'x': np.array([data['mean_min'][0], data['mean_max'][0]]),
        'y': np.array([1, 1]),
    }, ax=axes[2], color='#D400C9', label='m interval n=20', linewidth=2)
    sns.lineplot(x='x', y='y', data={
        'x': np.array([data['mean_min'][1], data['mean_max'][1]]),
        'y': np.array([1.1, 1.1]),
    }, ax=axes[2], color='#0007FF', label='m interval n=100', linewidth=2)
    axes[2].set_ylim(0.9, 1.4)
    axes[2].legend()

    sns.lineplot(x='x', y='y', data={
        'x': np.array([data['dev_min'][0], data['dev_max'][0]]),
        'y': np.array([1, 1]),
    }, ax=axes[3], color='#D400C9', label='sigma interval n=20', linewidth=2)
    sns.lineplot(x='x', y='y', data={
        'x': np.array([data['dev_min'][1], data['dev_max'][1]]),
        'y': np.array([1.1, 1.1]),
    }, ax=axes[3], color='#0007FF', label='sigma interval n=100', linewidth=2)
    axes[3].set_ylim(0.9, 1.4)
    axes[3].legend()

    plt.show()
    result_df = pd.DataFrame(data)
    print(result_df)


ns = [20, 100]


def get_interval_any(name: str, fun, *args):
    data = {
        'n': [],
        'mean_min': [],
        "mean": [],
        'mean_max': [],
        'dev_min': [],
        "dev": [],
        'dev_max': [],
    }

    for index, n in enumerate(ns):
        sample = fun(*args, size=n)
        mean = sample.mean()
        data["mean"].append(mean)
        dev = np.std(sample)
        data["dev"].append(dev)
        s = math.sqrt(((sample - mean) ** 2 / n).sum())
        m4 = ((sample - mean) ** 4).sum() / n
        e = m4 / s ** 4 - 3

        normal_quantile = scipy.stats.norm.ppf(CONFIDENTIAL_LEVEL)

        dev_min = s * (1 - 0.5 * normal_quantile * math.sqrt((e + 2) / n))
        dev_max = s * (1 + 0.5 * normal_quantile * math.sqrt((e + 2) / n))

        mean_min = mean - dev * normal_quantile / math.sqrt(n)
        mean_max = mean + dev * normal_quantile / math.sqrt(n)

        data['n'].append(n)
        data['mean_min'].append(mean_min)
        data['mean_max'].append(mean_max)
        data['dev_min'].append(dev_min)
        data['dev_max'].append(dev_max)

    result_df = pd.DataFrame(data)
    print(result_df)


get_interval("Normal", np.random.normal, 0, 1)
get_interval_any("poisson", np.random.poisson, 10)