import numpy as np
from utils import get_distribution, types, compute_trimmed_mean


def compute_stat_values():
    sizes = [10, 100, 1000]
    repeats = 1000

    for name in types:
        print(f"\nStatistics for {name} distribution:")
        print("{:<8} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "", "Mean", "Median", "z_r", "z_Q", "z_tr"
        ))

        for size in sizes:
            mean, med, zr, zq, ztr = [], [], [], [], []

            for _ in range(repeats):
                array = get_distribution(name, size)
                mean.append(np.mean(array))
                med.append(np.median(array))
                zr.append((max(array) + min(array)) / 2)
                zq.append((np.quantile(array, 0.25) + np.quantile(array, 0.75)) / 2)
                ztr.append(compute_trimmed_mean(array))

            print("{:<8}".format(f"n={size}"))

            print("{:<8} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
                "E(z)", np.mean(mean), np.mean(med), np.mean(zr), np.mean(zq), np.mean(ztr)
            ))

            print("{:<8} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
                "D(z)", np.mean(np.multiply(mean, mean)) - np.mean(mean) * np.mean(mean),
                np.mean(np.multiply(med, med)) - np.mean(med) * np.mean(med),
                np.mean(np.multiply(zr, zr)) - np.mean(zr) * np.mean(zr),
                np.mean(np.multiply(zq, zq)) - np.mean(zq) * np.mean(zq),
                np.mean(np.multiply(ztr, ztr)) - np.mean(ztr) * np.mean(ztr)
            ))


if __name__ == "__main__":
    compute_stat_values()
