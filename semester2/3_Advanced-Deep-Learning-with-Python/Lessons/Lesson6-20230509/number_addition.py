# This script file implements number addition as a sequence to sequence learning
# learning problem.

# Import all required python frameworks.
import tensorflow as tf
from tensorflow import keras
import numpy as np
from classes.character_table import CharacterTable
import matplotlib.pyplot as plt
import numpy.typing
from numpy.typing import NDArray
from numpy import int32

# -----------------------------------------------------------------------------
#                   GLOBAL VARIABLES DEFINITION:
# -----------------------------------------------------------------------------

# Set the global parameters governing the generation of the dataset and the
# training of the model.
CHAR_DIGITS = list("0123456789")  # List of unique digits.
DATA_SIZE = 50000  # Number of available patterns.
DIGITS = 3            # Number of digits representing each summand.
REVERSE = False        # Whether to use sequence order inversion.
# Define the maximum length of input.
MAXLEN = DIGITS + 1 + DIGITS
# Define the dictionary for each character sequence.
DICTIONARY = "0123456789+ "


def data_generation() -> tuple[list[str], list[str]]:
    # Initialize a list storing the available summation questions.
    questions: list[str] = []
    expected: list[str] = []  # Initialize a list storing the expected summation answers.
    # Initialize a set object storing the unique summation pairs.
    seen: set[tuple[int, int]] = set()
    print("Data Generation Process ...")
    while len(questions) < DATA_SIZE:
        # Define the lambda function generating the sequence of digits that will
        # utlimately form each summand.
        def f(): return int(
            "".join(np.random.choice(CHAR_DIGITS)
                    for i in range(np.random.randint(1, DIGITS + 1)))
        )
        # Sample two random summands whose decimal representation requires DIGITS digits.
        a, b = f(), f()
        # Skip any summation questions that have already been processed.
        key: tuple[int, int] = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        # Form the current summation question string.
        q = f"{a}+{b}"
        # Pad the question string with spaces to ensure that is always of MAXLEN characters.
        query = q + " " * (MAXLEN - len(q))
        # Get the string version of the summation result.
        ans = str(a + b)
        # Pad the answer string with spaces to ensure that it is always DIGITS + 1  characters.
        ans += " " * (DIGITS + 1 - len(ans))
        if REVERSE:
            # Reverse the query string. For exapmple '12+345  ' becomes '  543+21'.
            # Note the space used for padding.)
            query = query[::-1]
        # Append the current query string to the questions list.
        questions.append(query)
        # Append the current ansewer string to the expected list.
        expected.append(ans)
        print("Total questions:", len(questions))

    return questions, expected


def data_vectorization(ctable: CharacterTable, questions: list[str], expected: list[str]):
    print("Data Vectorization Process...")

    # Initialize matrix X for storing the vectorized versions of the input sequences.
    X = np.zeros((len(questions), MAXLEN, len(DICTIONARY)), dtype=np.int32)
    # Initialize matrix Y for storing the vectorized versions of the output sequences.
    Y = np.zeros((len(expected), DIGITS + 1, len(DICTIONARY)), dtype=np.int32)
    # Vectorize each sentence within the questions container.
    for i, sentence in enumerate(questions):
        X[i] = ctable.encode(sentence, MAXLEN)
    # Vectorize each sentence within the expected container.
    for i, sentence in enumerate(expected):
        Y[i] = ctable.encode(sentence, DIGITS + 1)

    # Generate a vector of suffled indices in order to rearrange the context of the
    # matrices X and Y.
    indices = np.arange(DATA_SIZE)
    np.random.shuffle(indices)
    X: NDArray[int32] = X[indices]
    Y: NDArray[int32] = Y[indices]

    # Partition the available set of patterns so that 90% is used for training and
    # 10% is used for testing.
    # Define the cutoff point within the [1...DATA_SIZE]
    cutoff = len(X) - len(X) // 10
    # Define the training and testing sequences of patterns.
    Xtrain, Xtest = X[:cutoff], X[cutoff:]
    # Define the training and testing sequences of targets.
    Ytrain, Ytest = Y[:cutoff], Y[cutoff:]
    # Report shapes of training and testing X patterns.
    print("Training Data:")
    print(Xtrain.shape)
    print(Xtest.shape)
    # Report shaapes of training and testing Y patterns.
    print("Testing Data:")
    print(Ytrain.shape)
    print(Ytest.shape)

    return Xtrain, Ytrain, Xtest, Ytest


def main():
    # Instantiate the CharacterTable class.
    ctable = CharacterTable(DICTIONARY)

    # -----------------------------------------------------------------------------
    #                        DATA GENERATION PROCESS:
    # -----------------------------------------------------------------------------
    questions, expected = data_generation()

    # -----------------------------------------------------------------------------
    #                        DATA VECTORIZATION PROCESS:
    # -----------------------------------------------------------------------------
    Xtrain, Ytrain, Xtest, Ytest = data_vectorization(ctable, questions, expected)

    # -----------------------------------------------------------------------------
    #                        MODEL BUILDING PROCESS:
    # -----------------------------------------------------------------------------
    print("Model Building Process...")

    # Set the dimensionality for each encoding LSTM layer. Apparently, the number
    # of entries in the corresponding list designates the total number of encoding
    # LSTM layers. Mind that the last encoding LSTM layers should only be returning
    # each state.
    ENCODE_NEURONS = [128]  # Try ENCODE_NEURONS = [128,128,128]

    # Set the dimensionality for each decoding LSTM layer. Apparently, the number
    # of entries in the coresponding list designates the total number of decoding
    # LSTM layers. Mind that the decoding LSTM layers have to ouput a complete
    # vector of output values.
    DECODE_NEURONS = [128]  # Try DECODE_NEURONS = [128,128,128]

    # Initialize a sequential model.
    model = keras.Sequential()

    # Add the encoding layers:
    for encoding_layer, encoding_neurons in enumerate(ENCODE_NEURONS):
        if encoding_layer == 0:
            # CASE I: The first LSTM layer is added.
            if len(ENCODE_NEURONS) == 1:
                # CASE Ia: The first LSTM layer is the only LSTM layer:
                model.add(keras.layers.LSTM(encoding_neurons,
                          input_shape=(MAXLEN, len(DICTIONARY))))
            else:
                # Case Ib: The first LSTM layer is not the only LSTM layer:
                model.add(keras.layers.LSTM(encoding_neurons, return_sequences=True,
                                      input_shape=(MAXLEN, len(DICTIONARY))))
        elif encoding_layer <= len(ENCODE_NEURONS)-2:
            model.add(keras.layers.LSTM(encoding_neurons, return_sequences=True))
        else:
            model.add(keras.layers.LSTM(encoding_neurons))

    # Add a RepeatVector layer in order to repeat the input according to the number
    # of digits forming the expected answers. That is, at the input level of the
    # decoder, we must repeatedly provide the last output of the RNN for each time
    # step.
    model.add(keras.layers.RepeatVector(DIGITS + 1))

    # Add the decoding layers.
    for decoding_layer in DECODE_NEURONS:
        model.add(keras.layers.LSTM(decoding_layer, return_sequences=True))

    # Apply a dense layer to every temporal slice of the input. For each step of
    # the output sequence, this layer will decide which character will be chosen.
    model.add(keras.layers.Dense(len(DICTIONARY), activation="softmax"))

    # Compile the model.
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    # Provide a summary of the model.
    model.summary()

    # -----------------------------------------------------------------------------
    #                      MODEL TRAINING AND TESTING PROCESS:
    # -----------------------------------------------------------------------------
    print("Model Training and Testing Process...")

    # Set the number of training and testing epochs.
    EPOCHS = 30
    # Set the batch size.
    BATCH_SIZE = 32
    # Set the size of testing data to be used for demonstrating model performance.
    DEMONSTRATION_SIZE = 10

    # Initialize list containers for storing model performance measurements during
    # each epoch.
    training_loss = []
    testing_loss = []
    training_accuracy = []
    testing_accuracy = []

    for epoch in range(EPOCHS):
        print("Current epoch: {}".format(epoch))
        # Perform training and testing for the current epoch.
        history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=1,
                            validation_data=(Xtest, Ytest))
        # According to the current state of the RNN model, record its performance
        # measurements.
        training_loss.append(history.history["loss"])
        testing_loss.append(history.history["val_loss"])
        training_accuracy.append(history.history["accuracy"])
        testing_accuracy.append(history.history["val_accuracy"])

        # Select 10 random samples from the testing set in order to provide a better
        # visualization of the performance of the model.
        for _ in range(DEMONSTRATION_SIZE):
            # Select a random index within the testing subset.
            idx = np.random.randint(0, len(Xtest))
            # Get the associated pattern and target matrices storing the one-hot
            # vector encoding of the corresponding sequences.
            # Mind that idx has to be internaly converted to an np array of a single
            # element vector.
            pattern, target = Xtest[np.array([idx])], Ytest[np.array([idx])]
            # Acquire the one-hot vector version of the estimated target matrix.
            prediction = np.argmax(model.predict(pattern), axis=-1)
            # Get the string representation of the test pattern, i.e. the question
            # string.
            # We need the inner array element.
            question = ctable.decode(pattern[0])
            # Get the string representation of the target pattern, i.e. the answer
            # string.
            answer = ctable.decode(target[0])
            # Get the string representation of the estimated pattern, i.e. the
            # estimated answer string.
            guess = ctable.decode(prediction[0], calc_argmax=False)
            # Print the question string.
            print("QUESTION: {}".format(
                question[::-1] if REVERSE else question))
            # Print the correct answer string.
            print("ANSWER: {}".format(answer))
            # Print the estimated answer string.]
            if answer == guess:
                print("CORRECT RESULT: {}".format(guess))
            else:
                print("INCORRECT RESULT: {}".format(guess))

    # -----------------------------------------------------------------------------
    #                      PERFORMANCE VISUALIZATION PROCESS:
    # -----------------------------------------------------------------------------

    # Visualize training and testing history for loss.
    plt.plot(training_loss)
    plt.plot(testing_loss)
    plt.title('Model Accuracy in terms of Cross-Entropy Loss')
    plt.ylabel('Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper right')
    plt.grid()
    plt.show()

    # Visualize training and testing history for accuracy.
    plt.plot(training_accuracy)
    plt.plot(testing_accuracy)
    plt.title('Model Accuracy in terms of Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='lower right')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
