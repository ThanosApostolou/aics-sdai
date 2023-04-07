# This class file provides fundamental data preparation functionality.
# In particular, the DataPreparation class encapsulates the following 
# functionality:
#
# [1]: Loads the available english - french sentence pairs from the associated 
#      data file.
# [2]: Performs data vectorization at the character level.     

# Import all required Python modules.
import numpy as np

class DataPreparation:
    
    def __init__(self,data_path,samples_num):
        # ---------------------------------------------------------------------
        # This is the class constructor.
        # ---------------------------------------------------------------------
        # Input Arguments: 
        # data_path:   This is the relative path to the data file containing the 
        #              the english-french sentence pairs.
        # samples_num: Number of samples to be read from data file.
        # ---------------------------------------------------------------------
        # Class Variables: 
        # input_sentences : List storing the english sentences as sequences of 
        #                   words. 
        # target_sentences: List storing the french sentences as sequences of 
        #                   words.
        # input_dictionary:  Set object storing the input language dicionaty.
        # target_dictionary: Set object storing the target language dictionary.            
        # ---------------------------------------------------------------------
        self.data_path = data_path
        self.samples_num = samples_num
        # Initialize input and target sentences lists.
        input_sentences = []
        target_sentences = []
        # Initialize input and target dictionaries.
        input_dictionary = set()
        target_dictionary = set()
        # Open the specified data file:
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        # Set the number of lines to be read from the data file.
        # Mind that the last line of the file is actually an empty string.
        self.num_lines = min(self.samples_num, len(lines)-2)
        # Loop through the data file and collect the first two tab-separated
        # sentences in order 
        for line in lines[: self.num_lines]:
            input_text, target_text, _ = line.split("\t")
            # Set "tab" as the "start sequence" character for the target texts.
            # Set "\n" as the "end sequence" character for the target texts.
            target_text = "\t" + target_text + "\n"
            input_sentences.append(input_text)
            target_sentences.append(target_text)
            # Derive the dictionary for the input sentences.
            for char in input_text:
                if char not in input_dictionary:
                    input_dictionary.add(char)
            # Derive the dictionary for the target sentences.
            for char in target_text:
                if char not in target_dictionary:
                    target_dictionary.add(char)
        
        # Set the input dictionary as a sorted list of the unique input characters.
        self.input_dictionary = sorted(list(input_dictionary))
        # Set the output dictionary as a sorted list of the unique ouput characters.
        self.target_dictionary = sorted(list(target_dictionary))
        # Set class variables storing the input and target sentences.
        self.input_sentences = input_sentences
        self.target_sentences = target_sentences
        # Get the lenghts of the input and target dictionaries.
        self.num_encoder_tokens = len(self.input_dictionary)
        self.num_decoder_tokens = len(self.target_dictionary)
        # Get the maximum input sequence length.
        self.max_encoder_sequence_length = max([len(txt) for txt in input_sentences])
        # Get the maximum target sequence length.
        self.max_decoder_sequence_length = max([len(txt) for txt in target_sentences])
        
        # Report internal variables dimensions.
        print("Number of samples: {}".format(len(self.input_sentences)))
        print("Size of input dictionary: {}".format(len(self.input_dictionary)))
        print("Size of target dictioary: {}".format(len(self.target_dictionary)))
        print("Length of maximum input sequence: {}".format(self.max_encoder_sequence_length))
        print("Length of maximum target sequence: {}".format(self.max_decoder_sequence_length))
        
    def vectorize_input_target_sentences(self):
        # ---------------------------------------------------------------------
        # This function provides the one-hot vector encoding for each character 
        # in the input and target sentences.
        # ---------------------------------------------------------------------
        # Define dictionary objects for acquiring the indices of the characters
        # stored within the input and target dictionaries.
        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_dictionary)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_dictionary)])
        
        # Define reverse-lookup dictionaries in order to decode sequences back
        # to something readable.
        self.reverse_input_char_index = dict((i,char) for char,i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i,char) for char,i in self.target_token_index.items())
        
        # Initialize a matrix for storing the one-hot vector encoded version of 
        # the encoder input data.
        self.encoder_input_data = np.zeros((self.num_lines, 
                                            self.max_encoder_sequence_length, 
                                            self.num_encoder_tokens), 
                                            dtype="float32")
        # Initialize a matrix for storing the one-hot vector encoded version of 
        # the decoder input data.
        self.decoder_input_data = np.zeros((self.num_lines,
                                            self.max_decoder_sequence_length,
                                            self.num_decoder_tokens),
                                            dtype="float32")
        # Intialize a matrix for storing the one-hot vector encoded version of
        # the decoder target data.
        self.decoder_target_data = np.zeros((self.num_lines,
                                            self.max_decoder_sequence_length,
                                            self.num_decoder_tokens),      
                                            dtype="float32")
        # Performe the actual one-hot vector encoding for each pair of input 
        # and output sentences.
        # Loop through the various pairs of input-target sentences.
        for i, (input_sentence,target_sentence) in enumerate(zip(self.input_sentences,self.target_sentences)):
            # Loop through each character of the input sentence.
            for t, char in enumerate(input_sentence):
                self.encoder_input_data[i,t,self.input_token_index[char]] = 1.0
            # The remaining elements corresponding to the absent input sequence 
            # characters until the max_encoder_sequence_length is reached will 
            # be filled with the one-hot encoding for the space character.
            self.encoder_input_data[i,t+1:,self.input_token_index[" "]] = 1.0
            # Loop through each character of the target sequence.
            # Take into consideration that the target data decoder should be one
            # timestep ahead from the input data decoder disregarding the starting 
            # character of the target sequence.
            for t, char in enumerate(target_sentence):
                self.decoder_input_data[i,t,self.target_token_index[char]] = 1.0
                # Ensure that the very first target sequence character is not 
                # being processed.
                if t > 0:
                    self.decoder_target_data[i,t-1,self.target_token_index[char]] = 1.0
            # The remaining elements corresponding to the absent target sequence
            # characters until the max_decoder_sequence_length is reached will
            # be filled with the one-hot encoding for the space character.
            self.decoder_input_data[i,t+1:,self.target_token_index[" "]] = 1.0
            self.decoder_target_data[i,t:,self.target_token_index[" "]] = 1.0