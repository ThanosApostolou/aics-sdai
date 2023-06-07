# This class file provides the fundamental code functionality for the 
# implementation of the Positional Encoding mechanism which which is utilized 
# by the Transformer Model. This functional component is used to inject the
# position information of each token in the input sequence. Sine and Cosine 
# functions of different frequences are used in order to generate the final 
# positional encoding.

# The PositionalEncoding class initializes this structural module with the 
# input parameters d_model and max_seq_length. Its ultimate purpose is to 
# create a tensor which will be storing the positional encoding values. The 
# class computes sine and cosine values for even and odd indices of the input 
# instances, respectively, based on the scaling factor div_term.

# The forward method computes the positional encoding by adding the previously
# stored positional encoding values to the input tensor, allowing the model to
# capture the position information of the input sequence.

# The required input parameters for the class constructor are the following:
# [i]:  The dimensionality d_model of each instance of the input sequence.
# [ii]: The maximum input sequence length that will be feeded within the 
#       Transformer model.

# Import required libraries.
import torch
import torch.nn as nn
import math

# Class Definition.
class PositionalEncoding(nn.Module):
    
    # Class Constructor.
    def __init__(self, d_model, max_seq_length):
        # Call the super class constructor.
        super(PositionalEncoding, self).__init__()
        
        # Initialize the position encoding tensor.
        pe = torch.zeros(max_seq_length, d_model)
        
        # Initialize the position tensor.
        # Mind that the unsqueeze function generates a one-dimensional tensor 
        # at the dimension specified by the corresponding input argument.  
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Initialize the div_term tensor.
        div_term = torch.exp(torch.
                             arange(0, d_model, 2).
                             float() * (math.log(10000.0) / d_model))
        
        # Replace the even and odd positions of the position encoding.
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        # The position encoding tensor should be registered as a buffer that 
        # will not be considered as a model parameter. Buffers, however, are 
        # persistent by default and will be saved alongside parameters. This 
        # behavior may be altered by setting the persistent logical argument to
        # false. Thus, the buffer will not be part of the the module's 
        # state_dict.
        self.register_buffer('pe',pe.unsqueeze(0))
    
    # Forward Pass Function.
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]