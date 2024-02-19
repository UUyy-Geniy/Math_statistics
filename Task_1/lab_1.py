import matplotlib.pyplot as plt
from utils import get_distribution, get_density, types

num_bins = 20


def plot_hist():
    sizes = [10, 50, 1000]
    for name in types:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(name)
        for i in range(len(sizes)):
            array = get_distribution(name, sizes[i])
            n, bins, patches = ax[i].hist(array, num_bins, density=True, edgecolor='blue', alpha=0.2)
            ax[i].plot(bins, get_density(name, bins), color='blue', linewidth=1)
            ax[i].set_title("n = " + str(sizes[i]))
        plt.show()


if __name__ == "__main__":
    plot_hist()
