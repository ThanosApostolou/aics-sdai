from sklearn.utils import Bunch
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


def createPlo(iris: Bunch):
    fig, ax = plt.subplots()
    x_index = 3
    colors = ['blue', 'red', 'green']

    for target_index in range(len(iris.target_names)):
        color = colors[target_index % len(colors)]
        ax.hist(iris.data[iris.target == target_index, x_index],
                label=iris.target_names[target_index],
                color=color)

    # for label, color in zip(range(len(iris.target_names)), colors):
    #     ax.hist(iris.data[iris.target == label, x_index],
    #             label=iris.target_names[label],
    #             color=color)

    ax.set_xlabel(iris.feature_names[x_index])
    ax.legend(loc='upper right')
    plt.show()


def main():
    iris: Bunch = load_iris()
    print(type(iris))
    print(iris.keys())
    print(iris["target_names"])
    print(iris.target_names)
    print(iris.data)
    myndarray = np.bincount(iris.target)
    print(myndarray)
    print()
    createPlo(iris)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
