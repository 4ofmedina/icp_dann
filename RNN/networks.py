import torch
from torch import nn
import torch.nn.functional as F

class LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) implementation
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device):
        super(LSTM, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        # Defining the layers: LSTM + FC
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True) 
        #self.fc = nn.Linear(hidden_dim, output_size) #!!!!!

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        (h0, c0) = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.lstm(x, (h0, c0))

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # concatenates all hidden states and reshape
        out = out.contiguous().view(-1, self.hidden_dim)
        #out = self.fc(out)
        #out = self.fc2(out)
        return out, _

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return (h0, c0)
    
    
class GradReverse(torch.autograd.Function):
"""
Extension of grad reverse layer
"""
@staticmethod
def forward(ctx, x, constant):
    ctx.constant = constant
    return x.view_as(x)

@staticmethod
def backward(ctx, grad_output):
    grad_output = grad_output.neg() * ctx.constant
    return grad_output, None

def grad_reverse(x, constant):
    return GradReverse.apply(x, constant)


class Domain_classifier(nn.Module):

    def __init__(self, input_size):
        super(Domain_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 2)
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        # logits = F.relu(self.bn1(self.fc1(input)))
        # logits = F.log_softmax(self.fc2(logits), 1)
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)

        return logits
    
    
class Regressor(nn.Module): #Regression

    def __init__(self, input_size):
        super(Regressor, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 100)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(100, 1)
        

    def forward(self, input):
        #input = input.expand(input.data.shape[0], 3, 28, 28)
        #input = input.expand(input.data.shape[0], 1, 300)
        # x = F.relu(F.max_pool2d(self.bn1(self.conv1(input)), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        # x = x.view(-1, 50 * 4 * 4)
        x = F.relu(F.max_pool1d(self.conv1(input), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
   ##     x = x.view(-1, 48)

        return x