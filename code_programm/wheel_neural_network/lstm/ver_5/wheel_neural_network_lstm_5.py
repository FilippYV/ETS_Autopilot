import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=12289, hidden_size=512, hidden_size_1=512, output_size=2, num_layers=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.linear(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out.squeeze(0)
