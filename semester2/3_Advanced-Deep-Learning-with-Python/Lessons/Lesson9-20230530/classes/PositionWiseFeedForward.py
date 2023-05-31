import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """1-Linear Transformation Layer: W: [d_model x d_ff]
    1-Linear Transformation Layer: W: [d_ff x d_model]
    1-ReLU activation function
    """
    def __init__(self, d_model, d_ff) -> None:
        super(PositionWiseFeedForward, self).__init__()
        # define the transformation layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # define the activation function layer
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.fc2(self.relu(self.fc1(x)))


