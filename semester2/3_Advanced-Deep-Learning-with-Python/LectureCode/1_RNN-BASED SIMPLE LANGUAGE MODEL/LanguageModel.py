# This script file provides fundamental computational functionality for building 
# a simple language model through the utilization of a vanila Recurrent Neural 
# Network

# Import all required Python libraries.
import torch
from torch import nn
import numpy as np
from classes.RecurrentNeuralNetworkModel import RecurrentNeuralNetworkModel

# -----------------------------------------------------------------------------
#                        FUNCTIONS DEFINITION:
# -----------------------------------------------------------------------------
# This function will provide the one - hot vector encoding for a given character
# sequence by providing the triplet of parameters (dict_size,seq_len,batch_size)
# -----------------------------------------------------------------------------
def one_hot_encode(sequence,dict_size,seq_len,batch_size):
    # Initialize a three - dimensional array of zeros with the desired output 
    # shape in order to store the one - hot vector encoded vectors of our 
    # sequence - based features.
    features = np.zeros((batch_size,seq_len,dict_size),dtype=np.float32)
    # Replace zeros at the relevant character indices with ones in order to 
    # represent that character.
    for i in range(batch_size):
        for j in range(seq_len):
            features[i,j,sequence[i][j]] = 1
    return features           
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# This function accepts the input arguments corresponding to the model and 
# current character of the sequence and returns the next character prediction 
# and the next hidden state of the RNN.
def predict_model(model,character):
    # Get the one-hot vector encoding of the input character.
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character,dict_size,character.shape[1],1)
    character = torch.from_numpy(character)
    character = character.to(model.device)
    out,hidden = model(character)
    # Get the probability for each character in the dictionary.
    prob = nn.functional.softmax(out[-1],dim=0).data
    # Output the character with the highest probability.
    char_ind = torch.max(prob,dim=0)[1].item()
    return int2char[char_ind], hidden
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# This function accepts the input arguments corresponding to the output length
# and input characters, returning the produced sequence.
def sample_model(model,out_len,start='hey'):
    # Evaluate the produced neural network model.
    model.eval()
    # Convert input character sequence to lower case.
    start = start.lower()
    # Firsly, run through the sequence of input characters.
    chars = [ch for ch in start]
    # Update the length of the sequence to be generated.
    size = out_len - len(chars)
    # Pass the previous characters through the network model in order to generate
    # the new sentence of predefined length.
    for i in range(size):
        char,h = predict_model(model,chars)
        # Append the predicted character to the existing character sequence.
        chars.append(char)
    return ''.join(chars)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #1
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Define the set of sentences that we will form the basis for this simple 
# language model. In fact, the language model will be trained to output these
# sentences when fed with the first word or the first few characters.
text = ['hey how are you','good i am fine','have a nice day']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #2
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Define a the set of unique characters appearing in the previously
# defined set of sentences.
chars = set(''.join(text))
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                                   STEP #3
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Define a dictionary that maps each unique character to an integer value.
int2char = dict(enumerate(chars))
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #4
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Define the reverse dictionary that maps each unique integer to the 
# corresponding character.
char2int = {char:idx for idx,char in int2char.items()}
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #5
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Input sequences will be padded in order to ensure a common standard length.
# RNNs are typically able to process variably sized input sequences. However, 
# input data will be fed in the network in batches in order to speed up the 
# training process. The utilization of batches during training demands for input
# sequences of the same length. In most cases, padding can be done by filling up 
# the sequences that are shorter than the required length with 0 values or by 
# trimming sequences that are longer than the required length. In this case, 
# padding will be conducted by determining the longest character sequence and 
# adding blank spaces to any sequence of shorter length. 

# Find the length of the longest character sequence in the dataset.
maxlen = len(max(text,key=len))

# Perform the actual zero padding process.
# Loop through the list of fundamental model sentences and add a ' ' whitespace
# character until the length of each sentence matches the length of the longest 
# sequence.
for i in range(len(text)):
    while len(text[i]) < maxlen:
        text[i] += ' '
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #6
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The main task addressed by this script is to predict the next character in the
# sequence at each time step. Therefore, each sentence needs to be divided into
# a pair of input and target / ground truth sequences. The target sequence will
# be formed by considering all characters in the sentence excluding the last one.
# The target sequence (i.e correct answer) will be formed by always considering
# the next character (one time step ahead) of the input sequence. That is, the 
# target sequence will always be one step ahead of the input sequence.
# For example the character sequence S = 'hey how are you' will provide the 
# input and target character sequences Sinput and Starget as follows:
# Sinput = 'hey how are yo' and Starget = 'ey how are you'.

# Initialize the lists holding the input and target character sequences.
input_seq = []
target_seq = []

# Loop through the various sentences in the dataset.
for i in range(len(text)):
    # Remove the last character for the input sequence.
    input_seq.append(text[i][:-1])
    # Remove the first character for the target sequence.
    target_seq.append(text[i][1:])
    # Print the input and target sequence.
    print("Input Sequence: {0} | Target Sequence: {1}".format(input_seq[i],target_seq[i]))
    print(70*"=")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #7
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The previously created dictionary {char2int} will be utilized  in order to 
# transform the input and target sequences to sequences of integers.
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]
    # Print the new forms of the input and target sequences.
    print("Input Sequence: {0}".format(input_seq[i]))
    print("Target Sequence: {0}".format(target_seq[i]))
    print(70*"=")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #8
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The next step is to encode the input sequences into one-hot vectors. This 
# operation requires the definition of the following parameters:
#
# [i] dict_size: is the dictionary size which corresponds to the number of 
#                unique characters in our text sentences. This parameter will 
#                determine the size of the one - hot vectors that will be 
#                encoding each character. That is, each character will have a 
#                unique index assigned in that vector.
#
# [ii] seq_len:  is the length of the sequences that we are going to feed in the 
#                model. Since the length of our text sentences has been 
#                standardize to reflect the maximum sequence length, the value 
#                of this parameter will be set equal to max_len - 1 given that 
#                both input and target sequences have been formed by removing 
#                the first and the last character of the sentence respectively.
#
# [iii] batch_size: the number of sentences defined which are going to be fed in 
#                   the model as a batch.
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

# Get the one - hot vector encoding for the input sequences.
# Input Shape ==> (Batch Size, Sequence Length, One-Hot Encoding Size)
input_seq = one_hot_encode(input_seq,dict_size,seq_len,batch_size)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #9
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Convert numpy arrays to torch tensors.
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)
# Print shapes of input and target sequences before feeding them in the model.
print('Input Sequence Shape: {}'.format(input_seq.shape))
print('Target Sequence Shape: {}'.format(target_seq.shape))
print(70*"=")
# -----------------------------------------------------------------------------
#                                   STEP #10
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The subsequent step is to implement the class RecurrentNeuralNetworkModel which
# will define the neural network architecture. Thus, we need to instantiate this
# class by providing the relevant set of internal parameterss.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #11
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The next step is to instantiate the class which defines the neural network
# architecture by providing the required set of internal hyperparameters. It is
# also required to initialize the execution enviroment for training the model. 
# The additional set of hyperparameters that needs to be specified is the following:
#
# [i] n_epochs: is the number of epochs determining the number of times the model
#               will go through the entire training dataset.
#
# [ii] lr: is the learning rate corresponding to the rate at which the model 
#          will be updating the weights in neural cells each time back-propagation
#          will be conducted.
# 
# Moreover, it is required to define the loss function and optimizer that will 
# utilized during the training process. 

# Instantiate the relevant class by providing the required set of internal parameters.
model = RecurrentNeuralNetworkModel(input_size=dict_size,output_size=dict_size,
                                    hidden_dim=12,n_layers=1,n_epochs=200,lr=0.01)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #12
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Run the actual training process.
model.train_model(input_seq,target_seq)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #13
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Set the test input character sequence.
test_input = 'good'
# Use the trained neural network model to generate a new character sequence.
test_text = sample_model(model,maxlen,'good')
print(70*"=")
print(f"Input Sequence: {test_input} ==> Generated Sequence: {test_text}")
print(70*"=")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   STEP #14
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Report the internal parameters of the model.
model.report_model()