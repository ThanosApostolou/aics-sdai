import torch
from torch import nn


class RecurrentNeuralNetworkModel(nn.Module):
    # class model that initializes the execution environment.
    def initialize_exectution_environment(self):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device('cuda')
            print('GPU is available')
        else:
            self.device = torch.device('cpu')
            print('CPGU is not available, CPU will be used instead!')
    

    # class constructor method:
    def __init__(self, input_size, output_size, hidden_dim, n_layers, n_epochs, lr):
        # Call Super class constructor
        super(RecurrentNeuralNetworkModel, self).__init__()
        self.initialize_exectution_environment()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr

        # Define the Neural Network Architecture
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        # define the execution devices
        self = self.to(self.device)

        # Define Fundamental Training Parameters of the model.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # X has size [batch_size, seq_len, dict_size]


    # Hidden State Initializer
    def initialize_hidden_state(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden
    

    # Define the Forward Information Pass:
    def forward(self, x):
        batch_size = x.size[0]
        hidden = self.initialize_hidden_state(batch_size)

        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1,self.hidden_dim)
        out = self.fc(out)
        
        # Return output variables.
        return out, hidden
    

    # Training Method
    def train_model(self, input_seq: torch.Tensor, target_seq: torch.Tensor):
        input_seq = input_seq.to(self.device)
        for epoch in range(1, self.n_epochs + 1):
            # Clear existing gradients
            self.optimizer.zero_grad()
            out, hidden = self(input_seq)

            output = self.output.to(self.device)
            target_seq = target_seq.to(self.device)

            loss = self.criterion(output,target_seq.view(-1).long())
            # Perform backpropagation and calculate gradients.
            loss.backward()

            self.optimizer

            self.optimizer.step()
            # Provide terminal output every 10 training epochs featuring the current
            # training epoch and the associated value of the loss function.
            if epoch % 10 == 0:
                print("Training Epoch {0}/{1}............".format(epoch,self.n_epochs),end=' ')
                print("Loss: {:.4f}".format(loss.item()))
