# This script file provides fundamental computational functionality for 
# simulating the behaviour of the Transformer Model on a synthetic dataset.
# This script file is responsible for the initialization of the transformer 
# as well as for training the model.

# Import all required Python modules.
import torch
import torch.nn as nn
import torch.optim as optim
import os
from classes.Transformer import Transformer

# Parameters Initialization.
source_vocab_size = 5000
target_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
batch_size = 64
dropout = 0.10
epochs = 500

# Initialize the Transformer Model.
transformer = Transformer(source_vocab_size, target_vocab_size, d_model, 
                         num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random source and target data.
source_data = torch.randint(1, source_vocab_size, (batch_size, max_seq_length))
target_data = torch.randint(1, target_vocab_size, (batch_size, max_seq_length))

# Define the fundamental training parameters of the transformer model.
# The ignore_index below specifies a target value that is ignored and does not 
# contribute to the input gradient.
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, 
                             betas=(0.9, 0.98), eps=1e-9)

# Check the existence of a trained transformer model stored in file:
# "transformer_model.pth"
if not os.path.exists("transformer_model.pth"):
    print("Trained model file does not exist. Begining training:")
    # Train the Transformer Model.
    transformer.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = transformer(source_data, target_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, target_vocab_size),
                         target_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1} Loss:{loss.item()}")
    # Save the trained model.
    torch.save(transformer, 'transformer_model.pth')
else:
    print("Trained model file exists. Model will be loaded:")
    # Load the trained model.
    transformer = torch.load("transformer_model.pth")