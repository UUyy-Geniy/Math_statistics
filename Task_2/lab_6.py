import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import math


def get_least_squares_coefs(x, y):
    xy_m = np.mean(np.multiply(x, y))
    x_m = np.mean(x)
    x_2_m = np.mean(np.multiply(x, x))
    y_m = np.mean(y)
    b1_mnk = (xy_m - x_m * y_m) / (x_2_m - x_m * x_m)
    b0_mnk = y_m - x_m * b1_mnk

    return b0_mnk, b1_mnk


def abs_dev_val(b_arr, x, y):
    return np.sum(np.abs(y - b_arr[0] - b_arr[1] * x))


def get_linear_approx_coefs(x, y):
    init_b = np.array([0, 1])
    res = minimize(abs_dev_val, init_b, args=(x, y), method='COBYLA')

    return res.x


def draw(lsm_0, lsm_1, lam_0, lam_1, x, y, title, fname):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='pink', s=6, label='Выборка')
    y_lsm = np.add(np.full(20, lsm_0), x * lsm_1)
    y_lam = np.add(np.full(20, lam_0), x * lam_1)
    y_real = np.add(np.full(20, 2), x * 2)
    ax.plot(x, y_lsm, color='pink', label='МНК')
    ax.plot(x, y_lam, color='red', label='МНМ')
    ax.plot(x, y_real, color='orange', label='Модель')
    ax.set(xlabel='X', ylabel='Y',
       title=title)
    ax.legend()
    ax.grid()
    fig.savefig(fname + '.png', dpi=200)


def task2():
    x = np.arange(-1.8, 2.1, 0.2)
    eps = np.random.normal(0, 1, size=20)
    # y = 2 + 2x + eps
    y = np.add(np.add(np.full(20, 2), x * 2), eps)
    # y2 with noise
    y2 = np.add(np.add(np.full(20, 2), x * 2), eps)
    y2[0] += 10
    y2[19] += -10

    if not os.path.exists("task2_data"):
        os.mkdir("task2_data")

    lsm_0, lsm_1 = get_least_squares_coefs(x, y)
    lam_0, lam_1 = get_linear_approx_coefs(x, y)

    lsm_02, lsm_12 = get_least_squares_coefs(x, y2)
    lam_02, lam_12 = get_linear_approx_coefs(x, y2)

    with open("task2_data/no_noise.tex", "w") as f:
        f.write("\\begin{enumerate}\n")
        f.write("\\item LSM without disturbances:\n")
        f.write("$\hat{a}\\approx " + str(lsm_0) + "$, $\hat{b}\\approx " + str(lsm_1) + "$\n")
        f.write("\\item LASSO without disturbances:\n")
        f.write("$\hat{a}\\approx " + str(lam_0) + "$, $\hat{b}\\approx " + str(lam_1) + "$\n")
        f.write("\\end{enumerate}\n")

    with open("task2_data/noise.tex", "w") as f:
        f.write("\\begin{enumerate}\n")
        f.write("\\item LSM with disturbances:\n")
        f.write("$\hat{a}\\approx " + str(lsm_02) + "$, $\hat{b}\\approx " + str(lsm_12) + "$\n")
        f.write("\\item LASSO with disturbances:\n")
        f.write("$\hat{a}\\approx " + str(lam_02) + "$, $\hat{b}\\approx " + str(lam_12) + "$\n")
        f.write("\\end{enumerate}\n")

    draw(lsm_0, lsm_1, lam_0, lam_1, x, y, 'Выборка без возмущений', 'task2_data/no_noise')
    draw(lsm_02, lsm_12, lam_02, lam_12, x, y2, 'Выборка с возмущенями', 'task2_data/noise')


task2()
