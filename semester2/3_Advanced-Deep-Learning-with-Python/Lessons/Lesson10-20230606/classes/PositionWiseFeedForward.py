# This class file provides the fundamental code functionality for the 
# implementation of the Position Wise Feed Forward Neural Network component 
# which lies at the core of the Transformer Model. This functional component 
# extends the fundamental functionality provided by PyTorch in order to define
# a position-wise feed-forward neural network architecture. 

# The PositionWiseFeedForward class initializes with two linear transformation
# layers and a ReLU activation function. Assuming that the internal dimension
# parameter of this component is given by the parameter d_ff:
# [i]:  The first linear transformation layer provides a [d_model x d_ff] 
#       mapping
# [ii]: The second linear transformation layer provides a [d_ff x d_model] 
#       mapping.

# The forward method applies these transformations and activation function 
# sequentially in order to compute the output. This process enables the model
# to take into consideration the position of the input elements while making
# predictions.

# The required input parameters for the class constructor are the following:
# [i]:  The dimensionality d_model of each instance of the input sequence.
# [ii]: The intrinsic dimensionality d_ff of the combined linear transformation
#       layer which is realized by this component.

# Import required libraries.
import torch.nn as nn

# Class Definition.
class PositionWiseFeedForward(nn.Module):  
    
    # Class Constructor.
    def __init__(self, d_model, d_ff):
        # Call the super class constructor.
        super(PositionWiseFeedForward, self).__init__()
        
        # Define the first linear transformation layer.
        self.fc1 = nn.Linear(d_model, d_ff)
        # Defin the second linear transformation layer.
        self.fc2 = nn.Linear(d_ff, d_model)
        # Define the activation function layer.
        self.relu = nn.ReLU()
        
    # Forward Pass Function.
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))