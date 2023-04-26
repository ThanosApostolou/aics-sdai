# -----------------------------------------------------------------------------
# This class file provides fundamental computational functionality for the 
# implementation of the Recurrent Neural Network for a simple language model 
# based on the pytorch framework.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Import all required Python modules.
import torch
from torch import nn
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Class Definition.
class RecurrentNeuralNetworkModel(nn.Module): # [STEP # 10.0]
# -----------------------------------------------------------------------------
#   Execution Enviroment Initializer: # [STEP # 10.1]
# -----------------------------------------------------------------------------
    def initialize_execution_environment(self):
        # Check the existence of a GPU.
        is_cuda = torch.cuda.is_available()
        # If a GPU device is available it will be utilized during training.
        # Otherwise, a CPU device will be used instead.
        if is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available!")
            print(70*"=")
        else:
            self.device = torch.device("cpu")
            print("GPU is not available, CPU will be used instead!")
            print(70*"=")
# -----------------------------------------------------------------------------
#   Class Constructor: # [STEP # 10.2]
# -----------------------------------------------------------------------------
    def __init__(self,input_size,output_size,hidden_dim,n_layers,n_epochs,lr):
        # The key idea behind building a custom neural network model is to 
        # define a class that will be inheriting from PyTorch's base class 
        # nn.module which provides the archetypical behaviour for all neural
        # network models. This particular model will be implemented through the
        # utilization of a single RNN layer followed by a fully connected layer.
        # In fact, the fully connected layer will be responsible for converting
        # the RNN output to the desired shape.
                
        # Call the super class constructor.
        super(RecurrentNeuralNetworkModel,self).__init__()
        
        # Initilize the exexution environment.
        self.initialize_execution_environment()
        
        # Set the internal class parameters.
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        
        # Define the neural network architecture.
        
        # Define the RNN layer.
        
        # Keep in mind that when the input data have the following shape:
        # (seq_len,batch_size,features) then batch_first = True is not required.
        # However, when the input data have the following shape:
        # (batch_size,seq_len,features) as it is true for this particular 
        # implementation, then setting the batch_first parameter to true forces
        # the RNN module to provide an output of the following shape:
        # (batch_size,seq_len,hidden_size).
        self.rnn = nn.RNN(input_size,hidden_dim,n_layers,batch_first=True)
        
        # Define the fully connected layer.
        self.fc = nn.Linear(hidden_dim,output_size)
        
        # Set the execution enviroment of the model.
        self = self.to(self.device)
        
        # Initialize the training criterion.
        self.criterion = nn.CrossEntropyLoss()
        
        # Initiliaze the optimization method.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#   Hidden State Initializer Method: # [STEP # 10.3]
# -----------------------------------------------------------------------------    
    def initialize_hidden_state(self,batch_size):
        # This function generates the first hidden state of zeros that will be
        # utilized during the forward pass of the information through the 
        # neural network architecture. In fact, the tensor will be returned by 
        # storing its contents within the previously determined device.
        hidden = torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(self.device)
        return hidden        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#   Forward Pass Method: # [STEP # 10.4]
# -----------------------------------------------------------------------------
    def forward(self,x):
        # It is imperative to define the forward pass function as a class method.
        # Take into consideration that the forward function is executed 
        # sequentially. This, in turn, yields that the inputs as well as the 
        # zero-initialized hidden state has to pass through the RNN layer first,
        # before the RNN outputs are passed to the fully connected layer.
        
        # Get the batch size of the input tensor x.
        batch_size = x.size(0)
        
        # Initialize the hidden state for the first instance of the sequence
        hidden = self.initialize_hidden_state(batch_size)
        
        # Input and hidden state should be passed into the neural network model 
        # in order to obtain the corresponding outputs.
        out, hidden = self.rnn(x,hidden)
        
        # Outputs should be reshaped so that they can be fed into the fully 
        # connected layer. Take into consideration the fact that the RNN output
        # contains the last hidden states (last with respect to the number of 
        # layers) for all timesteps. Therefore, we need to reshape the respective
        # tensor into a two-dimensional one whos columns will be equal to the 
        # dimensionality of the hidden state.
        out = out.contiguous().view(-1,self.hidden_dim)
        out = self.fc(out)
        
        # Return output variables.
        return out, hidden
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#   Training Method: # [STEP # 10.5]
# -----------------------------------------------------------------------------
    def train_model(self,input_seq,target_seq):
        input_seq = input_seq.to(self.device)
        # Loop through the various training epochs.
        for epoch in range(1,self.n_epochs+1):
            # Clear existing gradient from previous training steps.
            self.optimizer.zero_grad()
            output, hidden = self(input_seq)
            output = output.to(self.device)
            target_seq = target_seq.to(self.device)
            # Compute current loss value by first converting the target sequence to 
            # a one-dimensional long tensor.
            loss = self.criterion(output,target_seq.view(-1).long())
            # Perform backpropagation and calculate gradients.
            loss.backward()
            # Update the weights accordingly.
            self.optimizer.step()
            # Provide terminal output every 10 training epochs featuring the current
            # training epoch and the associated value of the loss function.
            if epoch%10 == 0:
                print("Training Epoch {0}/{1}............".format(epoch,self.n_epochs),end=' ')
                print("Loss: {:.4f}".format(loss.item()))
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#   Model Parameters Method: # [STEP # 10.5]
# -----------------------------------------------------------------------------
    def report_model(self):
        # This function reports all the internal parameters of the model.
        # Loop through the internal parameters of the model.
        for name,param in self.named_parameters():
            print("{0}: {1}".format(name,param.shape))
            print("{0}: {1}".format(type(param),param.requires_grad))
            print(70*"=")
