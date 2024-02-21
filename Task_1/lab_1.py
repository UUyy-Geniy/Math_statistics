import matplotlib.pyplot as plt
import numpy as np
from utils import get_distribution, get_density, types

num_bins = 20


def plot_hist():
    sizes = [10, 50, 1000]
    for name in types:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(name)
        for i in range(len(sizes)):
            array = get_distribution(name, sizes[i])
            ax[i].hist(array, num_bins, density=True, edgecolor='blue', alpha=0.2)

            if name == 'cauchy' and (sizes[i] == 50 or sizes[i] == 1000):
                ax[i].set_yscale('log')

            x = np.linspace(min(array), max(array), 1000)
            ax[i].plot(x, get_density(name, x), color='blue', linewidth=1)
            ax[i].set_title("n = " + str(sizes[i]))
        plt.show()


if __name__ == "__main__":
    plot_hist()
