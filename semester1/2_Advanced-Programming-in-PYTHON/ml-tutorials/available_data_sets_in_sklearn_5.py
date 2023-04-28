from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.utils import Bunch


def createPlot(digits: Bunch):
    # set up the figure
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

        # label the image with the target value
        ax.text(0, 7, str(digits.target[i]))

    plt.show()


def available_data_sets_in_sklearn_5():
    digits = load_digits()
    print(digits.keys())
    n_samples, n_features = digits.data.shape
    print((n_samples, n_features))
    print("Shape of an item: ", digits.data[0].shape)
    print("Data type of an item: ", type(digits.data[0]))
    print("Shape of an item: ", digits.images[0].shape)
    print("Data tpye of an item: ", type(digits.images[0]))
    createPlot(digits)
