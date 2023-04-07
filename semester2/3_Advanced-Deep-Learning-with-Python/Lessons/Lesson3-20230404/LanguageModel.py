import torch
from torch import nn
import numpy as np

# function definitions

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Initialize the features array
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    print('features', features)

    # Replace zeros with the corresponding character indices.
    for i in range(batch_size):
        for j in range(seq_len):
            features[i, j, sequence[i,j]] = 1

    

    return features


def main():
    print("Start main")

    # Step 1:
    # Define the fundamental sentences (character sequences) of the model
    text = [
        'hey how are you',
        'good i am fine',
        'have a nice day'
    ]
    ## Από κάθε πρόταση θα πρέπει να προκύψει μια input sequence (ακολουθία εισόδου) και μια target sequence
    ## target sequence είναι one time step ahead

    # Step 2:
    # Define the unique set of characters (language dictionary)
    chars = set(''.join(text))
    print('chars', chars)

    # Step 3:
    # Define a Python dictionary that will map each cunique character to a corresponding integer value
    int2char = dict(enumerate(chars))
    print('int2char', int2char)

    # Step 4:
    # Define the Python dictionary that will provide the inverse mapping
    # That is the mapping of each unique integer to the corresponding character
    char2int = {char: idx for idx,char in int2char.items()}
    print('char2int', char2int)


    # Step 5:
    # We need to determine the maximum sequence length for all sentences in our model.
    # Use this maximum sequence lenght in order to zero pad the contents of the individual sequences
    maxlen = len(max(text, key=len))

    # Perform the actual zero padding process.
    for i in range(len(text)):
        while len(text[i]) < maxlen:
            text[i] += " "
    

    # Step 6:
    # Generate the input and target sequence pairs corresponding to each sentence in our model
    ## Initialize the inpjut and target sequences.
    input_seq = []
    target_seq = []
    ## Loop throught the various sentences in our Language Model
    for i in range(len(text)):
        ## remove the last character from each sentence in order to form the corresponding input sequence
        input_seq.append(text[i][:-1])
        ## remove the first character from each sentence in order to form the corresponding target sequence
        target_seq.append(text[i][1:])


    print('input_seq', input_seq)
    print('target_seq', target_seq)


    # Step 7:
    # We need to transform the previously defined character sequences into sequences of integer values.
    for i in range(len(text)):
        input_seq[i] = [char2int[character] for character in input_seq[i]]
        target_seq[i] = [char2int[character] for character in target_seq[i]]


    print('input_seq', input_seq)
    print('target_seq', target_seq)


    # Step 8:
    seq_len = maxlen - 1
    dict_size = len(char2int)
    batch_size = len(text)
    

    # Step 9:
    # convert numpy arrays to vectors
    input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)

    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.Tensor(target_seq)




    print("End main")


if __name__ == "__main__":
    main()
