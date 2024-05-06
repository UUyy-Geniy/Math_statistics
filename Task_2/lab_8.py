import math
import os
import matplotlib.pyplot as plt
from scipy.stats import f
import numpy as np

AVERAGE_NORMAL = 0
SIGMA_NORMAL = 1
LEFT_UNIF = -math.sqrt(3)
RIGHT_UNIF = -LEFT_UNIF
LABDA_POISSON = 10

DISTRIBUTION_GENERATORS_DICT = {"normal": lambda n: np.random.normal(AVERAGE_NORMAL, SIGMA_NORMAL, size=n),
                                "uniform": lambda n: np.random.uniform(LEFT_UNIF, RIGHT_UNIF, size=n),
                                "poisson": lambda n: np.random.poisson(LABDA_POISSON, size=n),
                                "cauchy": lambda n: np.random.standard_cauchy(size=n),
                                "student": lambda n: np.random.standard_t(3, size=n)}


class Task4:
    def get_rvs(self, name, size):
        return DISTRIBUTION_GENERATORS_DICT[name](size)

    def calculate(self, distr, n1, i):
        indices1 = np.random.choice(distr.shape[0], n1, replace=False)
        # Получение подвыборки по сгенерированным индексам
        subsample1 = distr[indices1]

        mu1 = np.mean(subsample1)
        sigma1 = np.std(subsample1)

        print(f'$\mu_{i} = ' + str(np.around(mu1, decimals=2)) + "$")
        print(f'$\sigma_{i} = ' + str(np.around(sigma1, decimals=2)) + "$")

        s1 = 0
        for item in subsample1:
            s1 += (item - mu1) ** 2

        s1 = s1 / (n1 - 1)

        return s1, sigma1, subsample1

    def draw_hist(self, data_tot, item,
                  title="Default title"):
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.figure(figsize=(10, 6))

        for data_label in data_tot.keys():
            plt.hist(data_tot[data_label], alpha=0.5, label=data_label, edgecolor = "black")

        plt.title(title)
        plt.legend()
        plt.savefig(f"plots/{item}.png")
        plt.clf()

    def run(self):
        distr_names = ["normal"]
        alpha = 0.05
        p = 1 - alpha / 2
        N = 100

        distr1 = self.get_rvs(distr_names[0], size=N)
        distr2 = self.get_rvs(distr_names[0], size=N)

        num_N = [[20, 40], [20, 100]]

        for ind_N in range(len(num_N)):
            n1, n2 = num_N[ind_N]
            s1, sigma1, subdistr1 = self.calculate(distr1, n1, 1)
            print(f"$s_1 = {s1}$")
            s2, sigma2, subdistr2 = self.calculate(distr2, n2, 2)
            print(f"$s_2 = {s2}$")
            quantile = f.ppf(p, n1 - 1, n2 - 1)

            print(f"$F_B = {(s1 / s2, s2 / s1)[sigma1 < sigma2]}, qu = {quantile}$")

            self.draw_hist({f"Нормальное распределение N={N}": distr1,
                            f"Выборка 1 мощностью {n1}": subdistr1,
                            f"Выборка 2 мощностью {n2}": subdistr2},
                           item=ind_N + 1,
                           title=f"Распределение выборок для случая {ind_N + 1}")


Object = Task4()
Object.run()
