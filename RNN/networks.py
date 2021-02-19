import torch
from torch import nn
import torch.nn.functional as F


class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) implementation
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        # Defining the layers: RNN + FC
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        # here you could add dropout and a 2nd FC layer, or dropout before the current FC layer
        # dropout_rate = 0.1
        # self.fc1 = nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim/2)), nn.ReLU(), nn.Dropout(dropout_rate))
        # self.fc2 = nn.Linear(int(hidden_dim/2), output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        h0 = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.rnn(x, h0)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        # out = self.fc2(out)
        return out, _

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return h0


class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) implementation
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device):
        super(GRU, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        # Defining the layers: GRU + FC
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        h0 = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.gru(x, h0)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, _

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return h0


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
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        (h0, c0) = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.lstm(x, (h0, c0))

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # concatenates all hidden states and reshape
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.fc2(out)
        return out, _

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return (h0, c0)


class BiLSTM(nn.Module):
    """
    Bidirectional Long Short-Term Memory (LSTM) implementation
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device):
        super(BiLSTM, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        # Defining the layers: bidirectional LSTM + FC
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        (h0, c0) = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.lstm(x, (h0, c0))

        # Take only last hidden state
        out = out[:, -1, :]
        out = self.fc(out)
        return out, _

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        h0 = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to(self.device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return (h0, c0)


class FlowCNN(nn.Module):
    """
    Convolutional Network (ConvNet) implementation
    """
    def __init__(self):
        super(FlowCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layerfc1 = nn.Sequential(
            nn.Linear(2368, 1024),  # 2368 depends on the sequence size
            nn.ReLU(),
            nn.Dropout())

        self.fc2 = nn.Linear(1024, 1)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layerfc1(out)
        out = self.fc2(out)
        return out.view(-1)


class FlowCNNRecons(nn.Module):
    """
    Convolutional Network (ConvNet) implementation with input reconstruction
    """
    def __init__(self):
        super(FlowCNNRecons, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layerfc1 = nn.Sequential(
            nn.Linear(2368, 1024),  # 2368 depends on the sequence size
            nn.ReLU(),
            nn.Dropout())
        self.layerfc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout())
        self.fc3 = nn.Linear(512, 1)

        self.layerfc4 = nn.Sequential(
            nn.Linear(2368, 1024),
            nn.ReLU(),
            nn.Dropout())
        self.layerfc5 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout())
        self.fc6 = nn.Linear(512, 600)  # 600 is the sequence size

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        embedded = out.reshape(out.size(0), -1)

        out_reg = self.layerfc1(embedded)
        out_reg = self.layerfc2(out_reg)
        predictions = self.fc3(out_reg)

        out_rec = self.layerfc4(embedded)
        out_rec = self.layerfc5(out_rec)
        reconstructions = self.fc6(out_rec)

        return predictions.view(-1), reconstructions.view(-1)
    
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

    def __init__(self):
        super(Domain_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 2)
        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        # logits = F.relu(self.bn1(self.fc1(input)))
        # logits = F.log_softmax(self.fc2(logits), 1)
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)

        return logits
    
    
class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        ##self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.conv2 = nn.Conv1d(32, 48, 1)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size= 5)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 50, kernel_size= 5)
        # self.bn2 = nn.BatchNorm2d(50)
        self.conv2_drop = nn.Dropout()

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