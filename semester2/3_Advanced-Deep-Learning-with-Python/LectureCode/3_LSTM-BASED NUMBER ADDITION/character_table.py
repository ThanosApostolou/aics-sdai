# This python file implements the CharacterTable class which encapsulates the 
# following functionality: 
#
# Given a set of characters:
#
# [1]: Encode them to a one-hot integer representation.
# [2]: Decode the one-hot or integer representation to their corresponding 
#      character output.
# [3]: Decode a vector of probabilities to their character output.

# Import all required Python modules.
import numpy as np

class CharacterTable:
    
    def __init__(self, chars):
        # This is the class constructor initializing the utilized character 
        # table.
        # ---------------------------------------------------------------------
        # Input Arguments: 
        # chars: List of characters that may appear in the input.    
        # ---------------------------------------------------------------------
        # Get the corresponding list of sorted and unique characters.
        self.chars = sorted(set(chars))
        # Generate the respective dictionary of character-indices pairs.
        self.char_indices = dict((c,i) for i,c in enumerate(self.chars))
        # Generate the respective dictionary of indices-characters pairs.
        self.indices_char = dict((i,c) for i,c in enumerate(self.chars))
        
    def encode(self, C, num_rows):
        # This method provides the one-hot encoding for the given string C.
        # ---------------------------------------------------------------------
        # Input Arguments: 
        # C: The string to be encoded.
        # num_rows: The number of rows in the returned matrix of one-hot vector 
        #           encodings for each character. This parameter ensures that 
        #           the number of rows for each data pattern is the same.
        # ---------------------------------------------------------------------
        # Initialize the output matrix X:
        X = np.zeros((num_rows, len(self.chars)))
        # Loop through the various elements of the input string.
        for i,c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X
    
    def decode(self, X, calc_argmax=True):
        # This method converts the given vector or 2D array to their character
        # output.
        # ---------------------------------------------------------------------
        # Input Arguments:
        # X: When calc_argmax is False, it is a vector or a two-dimmensional 
        #    array of probabilities or one-hot vector representations.
        #    When calc_argmax is True, it is a vector or character indices.
        # calc_argmax: Boolean flag indicating whether to find the character 
        #              index with maximum probability.
        # ---------------------------------------------------------------------
        if calc_argmax:
            # Get the index of the maximum element per each row of matrix X.
            # In this way the matrix of one-hot vectors is transformed into a 
            # row vector containing the indices of the dictionary entries.
            X = X.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in X)    