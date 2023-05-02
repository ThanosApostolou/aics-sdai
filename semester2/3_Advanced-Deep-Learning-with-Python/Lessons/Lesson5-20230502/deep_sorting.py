# This classs file provides fundamental computation functionality for sorting
# a given sequence of integers through the utilization of a deep sequence
# neural model.


# Import all required python libraries.

import logging, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, Dropout, Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
import warnings
# Ignore keras-related warnings.
warnings.filterwarnings("ignore")

class DeepSorting:
    """batch_size = πλήθος σειρών που δίνουμε στο μοντέλο για εκπαίδευση
    sequence_length =  το μέγεθος της κάθε σειράς
    maximum_number = ο μέγιστος ακέραιος που υπάρχει στα inputs μας
    hidden_size = το μέγεθος του κρυφού επιπέδου
    layers_number = 
    epochs = πόσα epochs θα τρέξει
    burning_period = κάθε πότε θα τυπώνουμε μήνυμα
    min_accuracy = η ελάχιστη ακρίβεια
    """

    # Class Constructor:
    def __init__(self, batch_size, sequence_length, maximum_number, hidden_size,
                 layers_number, epochs, burning_period, min_accuracy):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.maximum_number = maximum_number
        self.hidden_size = hidden_size
        self.layers_number = layers_number
        self.epochs = epochs
        self.burning_period = burning_period
        self.min_accuracy = min_accuracy

    # Function for converting integer inputs to one-hot encoded input vectors to
    # the RNN model.
    # X input is a 2 dimnsions array with numbers X.shape = (len(X), sequence_lenght)
    # x output is a 3 dimnsions array with numbers encoded as onehot vector x.shape = (len(X), sequence_lenght, maximum_number)
    def encode(self, X):
        assert X.shape == (len(X), self.sequence_length)
        # Initialize one-hot encoded input matrix.
        x = np.zeros((len(X), self.sequence_length, self.maximum_number),
                     dtype=np.float32)
        # Loop through the various elements of X.
        for batch_index, batch in enumerate(X):
            for sequence_index, element_index in enumerate(batch):
                x[batch_index, sequence_index, element_index] = 1.0
        
        assert x.shape == (len(X), self.sequence_length, self.maximum_number)
        return x

    # Generator function providing an infinite stream of inputs for training.
    def batch_generator(self):
        # Randomly generate a batch of integer sequences X and its sorted
        # counterpart Y.
        x = np.zeros((self.batch_size, self.sequence_length, self.maximum_number),
                     dtype=np.float32)
        y = np.zeros((self.batch_size, self.sequence_length, self.maximum_number),
                     dtype=np.float32)
        while True:
            # Generate a batch of input X.
            X = np.random.randint(self.maximum_number, size=(self.batch_size,
                                                             self.sequence_length))
            # Generate the corresponding batch of ouput Y.
            Y = np.sort(X, axis=1)
            # Loop through the various elements of the input X.
            x = self.encode(X)
            # Loop through the various elements of the output Y.
            y = self.encode(Y)

            # Return the current version of the 3-dimensional matrices x and y.
            yield x, y
            x.fill(0.0)
            y.fill(0.0)

    # Model creation function.
    def create_model(self):
        print("Model Creation Process...")
        # Initialize the overall sequencial neural model.
        self.model = Sequential()
        # Add the LSTM encoder layer.
        self.model.add(LSTM(self.hidden_size, input_shape=(self.sequence_length,
                                                           self.maximum_number)))
        # The next layer is a RepeatVector layer which ensures that input
        # will be repeated sequence_length times. That is, at the input level of
        # the decoder, we must repeatedly provide the last output of the RNN
        # for each instance of the input sequence.
        self.model.add(RepeatVector(self.sequence_length))
        # Add the LSTM decoder layers.
        for _ in range(self.layers_number):
            self.model.add(LSTM(self.hidden_size, return_sequences=True))
        # Add a dropout layer which zeroes out half of the existing neural
        # connections.
        self.model.add(Dropout(0.5))
        # The next layer allows to apply a layer to every temporal slice of a
        # given input. In fact, we need to apply the same dense layer to each of
        # the sequence_length timesteps.
        self.model.add(TimeDistributed(Dense(self.maximum_number)))
        # Because TimeDistributed applies the same instance of the Dense layer
        # to each of the timestamps, the same set of weights will be used.
        # Add a softmax activation layer.
        self.model.add(Activation('softmax'))
        # Compile the model.
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

    # Model training function.
    def train_model(self):
        # During the following training loop batches of input and target sequences
        # will be sampled from the previously defined generator in order to be
        # fed into the RNN model for learning purposes.
        print("Model training process...")
        for train_index, (X, Y) in enumerate(self.batch_generator()):
            loss, accuracy = self.model.train_on_batch(X, Y)
            # Check the RNN model performance on a random sequence of integers
            # every burning_period training epochs have been completed.
            if train_index % self.burning_period == 0:
                print("Current Traning Epoch: {}".format(train_index))
                # Sample a random sequence of integers.
                testX = np.random.randint(self.maximum_number,
                                          size=(1, self.sequence_length))
                # Get the one-hot vector encoding for the input sequence.
                x_test = self.encode(testX)
                # Print current accuracy and loss measurements.
                print("Current Accuracy: {}".format(accuracy))
                print("Current Loss: {}".format(loss))
                # Obtain model prediction for the one-hot vector encoded version
                # of the currently sampled input sequence.
                y_test = self.model.predict(x_test, batch_size=1)
                # Get the correctly sorted version of the input sequence.
                testY = np.sort(testX)[0]
                # Get the estimated sorted version of the input sequence.
                testYrnn = np.argmax(y_test, axis=2)[0]
                # Check whether the rnn-based sorting has been succesfully
                # conducted.
                sort_status = np.array_equal(testY, testYrnn)
                if sort_status:
                    print("CORRECT:")
                else:
                    print("INCORRECT:")
                # Print the input sequence.
                print("Input Sequence = {}".format(testX))
                print("Target Sequence = {}".format(testY))
                print("Predicted Target Sequence = {}".format(testYrnn))
                # Check whether the minimum training accuracy has been reached.
                if accuracy > self.min_accuracy:
                    # Save model and break training loop.
                    self.model.save("sorting_model.h5")
                    break
            # Check whether the maximum number of training epochs has been
            # reached.
            if train_index > self.epochs:
                # Save the trained model and break the ongoing training process.
                self.model.save("sorting_model.h5")
                break

# =============================================================================
#                                          MAIN  PROGRAM
# =============================================================================


def main():
    print("Start main")
    # Set values to the global class parameters.
    batch_size = 32        # Number of training patterns per batch.
    sequence_length = 15   # Number of elements in the input sequence to sort.
    maximum_number = 100   # Maximum random value in the input sequence.
    hidden_size = 128      # Number of neurons in each RRN layer.
    layers_number = 2      # Number of RNN layers.
    epochs = 100000        # Number of training epochs.
    buring_period = 500    # Number of training epochs before each testing.
    min_accuracy = 0.995    # Minimum training accyracy required.

    # Instantiate Deep Sorting Class.
    print("Instantiating Deep Sorting Class...")
    deep_sort = DeepSorting(batch_size, sequence_length, maximum_number,
                            hidden_size, layers_number, epochs, buring_period,
                            min_accuracy)
    # Instantiate the neural model.
    deep_sort.create_model()
    # Start the training procees.
    deep_sort.train_model()
    print("End main")


if __name__ == "__main__":
    main()
