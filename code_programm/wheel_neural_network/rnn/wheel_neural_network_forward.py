import torch
import torch.nn as nn


class DeepRNNNet(nn.Module):
    def __init__(self, input_size=12289, hidden_size_1=512, hidden_size_2=128,
                 output_size=2, num_layers=1):  # num_layers теперь равен 2
        super(DeepRNNNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size_1
        self.rnn = nn.RNN(input_size, hidden_size_1, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc2 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input_data):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(input_data.device)

        out, hn = self.rnn(input_data, h0)

        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out
