import math

import scipy.stats as stats
from tabulate import tabulate
import numpy as np

AVERAGE_NORMAL = 0
SIGMA_NORMAL = 1
LEFT_UNIF = -math.sqrt(3)
RIGHT_UNIF = -LEFT_UNIF
LABDA_POISSON = 10

DISTRIBUTION_GENERATORS = [("normal", lambda n: np.random.normal(AVERAGE_NORMAL, SIGMA_NORMAL, size=n)),
                           ("uniform", lambda n: np.random.uniform(LEFT_UNIF, RIGHT_UNIF, size=n)),
                           ("poisson", lambda n: np.random.poisson(LABDA_POISSON, size=n)),
                           ("cauchy", lambda n: np.random.standard_cauchy(size=n)),
                           ("student", lambda n: np.random.standard_t(3, size=n))]

DISTRIBUTION_GENERATORS_DICT = {"normal": lambda n: np.random.normal(AVERAGE_NORMAL, SIGMA_NORMAL, size=n),
                                "uniform": lambda n: np.random.uniform(LEFT_UNIF, RIGHT_UNIF, size=n),
                                "poisson": lambda n: np.random.poisson(LABDA_POISSON, size=n),
                                "cauchy": lambda n: np.random.standard_cauchy(size=n),
                                "student": lambda n: np.random.standard_t(3, size=n)}

F_DICT = {"normal": lambda item: stats.norm.cdf(item),
          "student": lambda item: stats.t.cdf(item, 3),
          "uniform": lambda item: stats.uniform.cdf(loc=LEFT_UNIF, scale=(RIGHT_UNIF - LEFT_UNIF), x=item)}


class Task3:
    def get_probability(self, distr, limits, size, name):
        p_arr = np.array([])
        n_arr = np.array([])
        F = F_DICT[name]
        for idx in range(-1, len(limits)):
            prev_cdf = 0 if idx == -1 else F(limits[idx])
            cur_cdf = 1 if idx == len(limits) - 1 else F(limits[idx + 1])
            p_arr = np.append(p_arr, cur_cdf - prev_cdf)

            if idx == -1:
                n_arr = np.append(n_arr, len(distr[distr <= limits[0]]))
            elif idx == len(limits) - 1:
                n_arr = np.append(n_arr, len(distr[distr >= limits[-1]]))
            else:
                n_arr = np.append(n_arr, len(distr[(distr <= limits[idx + 1]) & (distr >= limits[idx])]))
        result = np.divide(np.multiply((n_arr - size * p_arr), (n_arr - size * p_arr)), p_arr * size)
        return n_arr, p_arr, result

    def get_k(self, size, flag):
        return math.ceil(1.72 * (size) ** (1 / 3))

    def create_table(self, n_arr, p_arr, result, size, limits):
        decimal = 3
        cols = ["i", "limits", "n_i", "p_i", "np_i", "n_i - np_i", "/frac{(n_i-np_i)^2}{np_i}"]
        rows = []
        for i in range(0, len(n_arr)):
            if i == 0:
                boarders = [-np.inf, np.around(limits[0], decimals=decimal)]
            elif i == len(n_arr) - 1:
                boarders = [np.around(limits[-1], decimals=decimal), 'inf']
            else:
                boarders = [np.around(limits[i - 1], decimals=decimal), np.around(limits[i], decimals=decimal)]

            rows.append(
                [i + 1, boarders, n_arr[i], np.around(p_arr[i], decimals=decimal),
                 np.around(p_arr[i] * size, decimals=decimal),
                 np.around(n_arr[i] - size * p_arr[i], decimals=decimal),
                 np.around(result[i], decimals=decimal)]
            )

        rows.append([len(n_arr) + 1, "-",
                     np.sum(n_arr),
                     np.around(np.sum(p_arr), decimals=decimal),
                     np.around(np.sum(p_arr * size), decimals=decimal),
                     -np.around(np.sum(n_arr - size * p_arr), decimals=decimal),
                     np.around(np.sum(result), decimals=decimal)])
        print(tabulate(rows, cols, tablefmt="latex"))
        return np.sum(result)
    def get_res(self, res):
        d = ["normal", "student", "uniform"]
        ch = "normal & student & uniform\n"
        for i in range(0, 3):
            ch += d[i] + " "
            for j in range(0, 3):
                ch += str(np.around(res[i][j], decimals=2)) + " & "
            ch += "\\\\ \\\\hline\n"
        print(ch)
    def get_rvs(self, name, size):
        return DISTRIBUTION_GENERATORS_DICT[name](size)

    def calculate(self, distr, p, k):
        mu = np.mean(distr)
        sigma = np.std(distr)

        print('mu = ' + str(np.around(mu, decimals=2)))
        print('sigma = ' + str(np.around(sigma, decimals=2)))

        limits = np.linspace(-1.1, 1.1, num=k - 1)
        chi_2 = stats.chi2.ppf(p, k - 1)
        print('chi_2 = ' + str(np.around(chi_2, decimals=2)))
        return limits

    def run(self):
        distr_names = ["normal", "student", "uniform"]
        alpha = 0.05
        p = 1 - alpha
        flag = True
        for n in [20, 100]:
            res = [[], [], []]
            for i, item in enumerate(distr_names):
                distr = self.get_rvs(item, size=n)
                k = self.get_k(n, flag)
                print(k)
                limits = self.calculate(distr, p, k)
                for item1 in distr_names:
                    n_arr, p_arr, result = self.get_probability(distr, limits, n, item1)
                    print("Name: ", item, "name2: ", item1)
                    res[i].append(self.create_table(n_arr, p_arr, result, n, limits))
            self.get_res(res)
            flag = False


Object = Task3()
Object.run()
