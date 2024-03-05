import matplotlib.pyplot as plt
from utils import types, get_distribution


def box_plot_tukey():
    sizes = [20, 100]
    for name in types:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        arr_20 = get_distribution(name, sizes[0])
        arr_100 = get_distribution(name, sizes[1])

        bp = ax.boxplot((arr_20, arr_100), patch_artist=True, vert=False, labels=["n = 20", "n = 100"])
        for whisker in bp['whiskers']:
            whisker.set(color="black", alpha=0.3, linestyle=":", linewidth=1.5)
        for flier in bp['fliers']:
            flier.set(marker="D", markersize=4)
        for box in bp['boxes']:
            box.set(color='red')
            box.set(alpha=0.6)
        for median in bp['medians']:
            median.set(color='black')

        plt.ylabel("n")
        plt.xlabel("X")
        plt.title(name)
        plt.show()


box_plot_tukey()
