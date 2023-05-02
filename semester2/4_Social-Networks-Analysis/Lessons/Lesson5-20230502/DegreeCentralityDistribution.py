import os
import numpy as np
import matplotlib.pyplot as plt


def DegreeCentralityDistribution(Degrees: np.ndarray, OUTPUT_DIR: str, name: str = "DegreeCentralityDistribution_plot"):
    """This function computes and displays the Degree Centrality distribution
    for a given vector of degree centralities.
    """

    min_degree = min(Degrees)
    max_degree = max(Degrees)
    degrees_range = range(min_degree, max_degree)
    plt.figure(3)
    H = plt.hist(Degrees, bins=degrees_range, align='left')
    plt.xticks(degrees_range)
    plt.xlabel('Degrees')
    plt.ylabel('Absolute Frequency')
    plt.grid(True)
    figure_file = os.path.join(
        OUTPUT_DIR, f"{name}.png")
    plt.savefig(figure_file)
    plt.show(block=False)
    return H
