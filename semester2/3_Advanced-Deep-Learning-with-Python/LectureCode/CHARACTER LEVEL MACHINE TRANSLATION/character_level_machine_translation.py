# This script file provides fundamental computational functinality for 
# facilitating the task of machine translation at the character level.

# -----------------------------------------------------------------------------
#                      DOWNLOAD AND UNZIP THE DATA
#------------------------------------------------------------------------------
# Run the following unix commands:
# !!curl -O http://www.manythings.org/anki/fra-eng.zip
# !!unzip fra-eng.zip
#------------------------------------------------------------------------------

# Import all required Python modules.
from classes.data_preparation import DataPreparation
from tensorflow import keras
import os.path
import warnings
import numpy as np

# Ignore keras-related warnings.
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
#                     GLOBAL CONFIGURATION VARIABLES
#------------------------------------------------------------------------------
BATCH_SIZE = 64       # Set the batch size for training.
EPOCHS = 100          # Set the number of training epochs.
LATENT_DIM = 256      # Set the dimensionality of the encoding space.
SAMPLES_NUM = 15000   # Set the number of training samples.
DATA_PATH = "fra.txt" # Set the path to the data file.
TRAINED_MODEL = "translation_model" # Set the name of the saved model.


# -----------------------------------------------------------------------------
#                     FUNCTION IMPLEMENTATION
#------------------------------------------------------------------------------
def decode_sequence(input_sequence):
    # Encode the input sequence as state vectors.
    states_value = encoder_model.predict(input_sequence)
    # Generate empty target sequence of length 1.
    target_sequence = np.zeros((1,1,data_preparator.num_decoder_tokens))
    # Populate the first character of the target sequence with the start 
    # character.
    target_sequence[0,0,data_preparator.target_token_index["\t"]] = 1.0
    # Sampling loop for a batch of sequences. For simplification reasons, here 
    # it is assumed a batch of unit length.
    stop_condition = False
    # Initialize the string storing the decoded sequence.
    decoded_sequence = ""
    while not stop_condition:
        output_tokens,hidden_states,cell_states = decoder_model.predict(
            [target_sequence]+states_value)
        
        # Sample a token.
        sampled_token_index = np.argmax(output_tokens[0,-1,:])
        sampled_char = data_preparator.reverse_target_char_index[sampled_token_index]
        decoded_sequence += sampled_char
        
        # Check the validity of the exit condition.
        if sampled_char == "\n" or len(decoded_sequence) > data_preparator.max_decoder_sequence_length:
            stop_condition = True
        
        # Update the target sequence of length 1.
        target_sequence = np.zeros((1,1,data_preparator.num_decoder_tokens))
        target_sequence[0,0,sampled_token_index] = 1.0
        
        # Update state vectors.
        states_value = [hidden_states,cell_states]
        
        # Return the decoded sequence.
    return decoded_sequence
        

# -----------------------------------------------------------------------------
#                     MAIN PRORGRAM: DATA PREPARATION PROCESS
#------------------------------------------------------------------------------

# Instantiate the data preperator class.
data_preparator = DataPreparation(DATA_PATH,SAMPLES_NUM)
# Run the one-hot vector encoding process for both the input and the target 
# sentences.
data_preparator.vectorize_input_target_sentences()

# -----------------------------------------------------------------------------
#                     MAIN PRORGRAM: MODEL BUILDING PROCESS
#------------------------------------------------------------------------------

# Define the input and lstm layers for the encoder.
encoder_inputs = keras.Input(shape=(None,data_preparator.num_encoder_tokens))
encoder = keras.layers.LSTM(LATENT_DIM,return_state=True)
encoder_outputs, encoder_hidden_state, encoder_cell_state = encoder(encoder_inputs)

# Keep the encoder internal states in a different structure.
encoder_states = [encoder_hidden_state,encoder_cell_state]

# Define the input layer for the decoder.
decoder_inputs = keras.Input(shape=(None,data_preparator.num_decoder_tokens))

# Define the lstm layer for the decoder.
# Take into consideration that the decoder will be returning the full output 
# sequences (hidden states) along with the internal states (cell states) of the 
# network. However, return states will not be used during training but only during
# testing the model.
decoder_lstm = keras.layers.LSTM(LATENT_DIM,return_sequences=True,return_state=True)
decoder_outputs,_,_ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
# Add a dense layer to the overall network structure.
decoder_dense = keras.layers.Dense(data_preparator.num_decoder_tokens,activation="softmax")
# Compute the final output layer tensor for the overall network architecture.
decoder_outputs = decoder_dense(decoder_outputs)

# Define the network model that will transform the encoder_input_data and the
# decoder_input_data into decoder_target_data.
# endoder_input_data + decoder_input_data ===> decoder_target_data.
model = keras.Model([encoder_inputs,decoder_inputs],decoder_outputs)


# Compile network model.
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])

# Check the existence of the MODEL_NAME directory. In case of existence, then the
# neural network model is assumed to be already trained.
model_directory_status = os.path.isdir(TRAINED_MODEL)
if not model_directory_status:
# -----------------------------------------------------------------------------
#                     MAIN PRORGRAM: MODEL TRAINING PROCESS
#------------------------------------------------------------------------------
    print("Training Model....")
    # Train the network model.
    model.fit(
              [data_preparator.encoder_input_data,data_preparator.decoder_input_data],
              data_preparator.decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs = EPOCHS,
              validation_split = 0.2)
    
    # Save the neural network model.
    model.save(TRAINED_MODEL)
else:
# -----------------------------------------------------------------------------
#                     MAIN PRORGRAM: MODEL TESTING PROCESS
#------------------------------------------------------------------------------
    print("Neural network model is already trained!!!")
    # Restore the already trained model.
    model = keras.models.load_model(TRAINED_MODEL)
    print("Trained Model Restored!")
    # Reconstruct the encoder model.
    print("Reconstructing Encoder Model...")
    encoder_inputs = model.input[0]
    encoder_outputs, encoder_hidden_states, encoder_cell_states = model.layers[2].output
    encoder_states = [encoder_hidden_states,encoder_cell_states]
    encoder_model = keras.Model(encoder_inputs,encoder_states)
    # Reconstruct the decoder
    print("Reconstructing Decoder Model...")
    decoder_inputs = model.input[1]
    decoder_hidden_state_inputs = keras.Input(shape=(LATENT_DIM,))
    decoder_cell_state_inputs = keras.Input(shape=(LATENT_DIM,))
    decoder_state_inputs = [decoder_hidden_state_inputs,decoder_cell_state_inputs]
    decoder_lstm = model.layers[3]
    decoder_outputs, decoder_hidden_states, decoder_cell_states = decoder_lstm(
        decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_hidden_states,decoder_cell_states]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model([decoder_inputs]+decoder_state_inputs,
                                [decoder_outputs]+decoder_states)
    
    # Generate decoded sequences.
    for sequence_index in range(20):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_sequence = data_preparator.encoder_input_data[sequence_index : sequence_index + 1]
        decoded_sentence = decode_sequence(input_sequence)
        print("-")
        print("Input sentence:", data_preparator.input_sentences[sequence_index])
        print("Decoded sentence:", decoded_sentence)