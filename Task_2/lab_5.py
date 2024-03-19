import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.stats as stats
import os


def scattering_ellipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def save_ellipses(size, ros):
    if not os.path.exists("task1_data"):
        os.mkdir("task1_data")
    fig, ax = plt.subplots(1, 3)
    size_str = "n = " + str(size)
    titles = [size_str + r', $ \rho = 0$', size_str + r', $\rho = 0.5 $', size_str + r', $ \rho = 0.9$']
    for i in range(len(ros)):
        sample = np.random.multivariate_normal([0, 0], [[1.0, ros[i]], [ros[i], 1.0]], size)
        x, y = sample[:, 0], sample[:, 1]
        scattering_ellipse(x, y, ax[i], edgecolor='pink')
        ax[i].grid()
        ax[i].scatter(x, y, s=5)
        ax[i].set_title(titles[i])
    fig.savefig('task1_data/' + str(size) + '.png', dpi=200)


def save_ellipses_for_mix(sizes):
    if not os.path.exists("task1_data"):
        os.mkdir("task1_data")
    fig, ax = plt.subplots(1, 3)
    for i in range(len(sizes)):
        sample = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], sizes[i]) + \
                 0.1 * np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], sizes[i])
        x, y = sample[:, 0], sample[:, 1]
        scattering_ellipse(x, y, ax[i], edgecolor='pink')
        ax[i].grid()
        ax[i].scatter(x, y, s=5)
        ax[i].set_title('n = ' + str(sizes[i]))
    fig.savefig('task1_data/mix.png', dpi=200)


def quadrant_coefficient(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    n = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(size):
        if x[i] >= med_x and y[i] >= med_y:
            n[1] += 1
        elif x[i] < med_x and y[i] >= med_y:
            n[2] += 1
        elif x[i] < med_x and y[i] < med_y:
            n[3] += 1
        elif x[i] >= med_x and y[i] < med_y:
            n[4] += 1
    return (n[1] + n[3] - n[2] - n[4]) / size


def get_coefficients(size, ro, repeats):
    pearson, quadrant, spearman = [], [], []
    for i in range(repeats):
        sample = np.random.multivariate_normal([0, 0], [[1, ro], [ro, 1]], size)
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spearman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrant_coefficient(x, y))
    return pearson, spearman, quadrant


def get_coefficients_for_mix(size, repeats):
    pearson, quadrant, spearman = [], [], []
    for i in range(repeats):
        sample = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size) + \
                 0.1 * np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], size)
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spearman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrant_coefficient(x, y))
    return pearson, spearman, quadrant


def save_table(ros, size):
    if not os.path.exists("task1_data"):
        os.mkdir("task1_data")
    rows = []
    for ro in ros:
        p, s, q = get_coefficients(size, ro, 1000)
        rows.append(['$\\rho = ' + str(ro) + '~(\\ref{ro})$', '$r ~(\\ref{r})$',
                     '$r_Q ~(\\ref{rQ})$', '$r_S ~(\\ref{rS})$'])
        rows.append(['$E(z)$', np.around(np.mean(p), decimals=3),
                     np.around(np.mean(s), decimals=3),
                     np.around(np.mean(q), decimals=3)])
        rows.append(['$E(z^2)$', np.around(np.mean(np.asarray([el * el for el in p])), decimals=3),
                     np.around(np.mean(np.asarray([el * el for el in s])), decimals=3),
                     np.around(np.mean(np.asarray([el * el for el in q])), decimals=3)])
        rows.append(['$D(z)$', np.around(np.std(p), decimals=3),
                     np.around(np.std(s), decimals=3),
                     np.around(np.std(q), decimals=3)])
    with open("task1_data/" + str(size) + ".tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def save_table_for_mix(sizes):
    if not os.path.exists("task1_data"):
        os.mkdir("task1_data")
    rows = []
    for size in sizes:
        p, s, q = get_coefficients_for_mix(size, 1000)
        rows.append(['$n = ' + str(size) + '$', '$r ~(\\ref{r})$',
                     '$r_Q ~(\\ref{rQ})$', '$r_S ~(\\ref{rS})$'])
        rows.append(['$E(z)$', np.around(np.mean(p), decimals=3),
                     np.around(np.mean(s), decimals=3),
                     np.around(np.mean(q), decimals=3)])
        rows.append(['$E(z^2)$', np.around(np.mean(np.asarray([el * el for el in p])), decimals=3),
                     np.around(np.mean(np.asarray([el * el for el in s])), decimals=3),
                     np.around(np.mean(np.asarray([el * el for el in q])), decimals=3)])
        rows.append(['$D(z)$', np.around(np.std(p), decimals=3),
                     np.around(np.std(s), decimals=3),
                     np.around(np.std(q), decimals=3)])
    with open("task1_data/mix.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def task1():
    sizes = [20, 60, 100]
    ros = [0, 0.5, 0.9]

    for size in sizes:
        save_ellipses(size, ros)
        save_table(ros, size)

    save_ellipses_for_mix(sizes)
    save_table_for_mix(sizes)
