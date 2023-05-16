# ONE-TO-ONE mapping
# X = {x1, x2, ..., xn}
# Y = {y1, y2, ..., yn}
# X and Y are one dimensional vaalues

import numpy as np
import tensorflow as tf
from tensorflow import keras

def main():
    print("start main")
    # set the length of the sequence
    n = 10
    # define the sequence
    seq = np.array([i/float(n) for i in range(n)])
    # create the input sequence
    X = seq.reshape(n, 1, 1)
    # create the target sequence
    Y = seq[::-1].reshape(n, 1)

    n_neurons = 2 * n
    n_batches = n
    n_epochs = 1000

    # create model
    one_to_one_model = keras.Sequential()
    one_to_one_model.add(keras.layers.LSTM(n_neurons, input_shape=(1, 1)))
    one_to_one_model.add(keras.layers.Dense(1, activation='tanh'))

    one_to_one_model.compile(loss="mean_squared_error", optimizer="adam")
    print(one_to_one_model.summary())

    # train model
    # history = one_to_one_model.fit()

    print("end main")

if __name__ =="__main__":
    main()