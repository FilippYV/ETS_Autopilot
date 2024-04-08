import torch
import torch.nn as nn


class IS_LSTM_Net_v3(nn.Module):
    def __init__(self, input_size=12289, hidden_size_0=1024, hidden_size_1=512, hidden_size_2=128, num_layers=2):
        super(IS_LSTM_Net_v3, self).__init__()
        self.hidden_size_0 = hidden_size_0
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size_0,
                            num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size_0, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)

        self.relu = nn.ReLU()

    def forward(self, combined_input):
        lstm_out, _ = self.lstm(combined_input)

        lstm_out = lstm_out[:, -1]

        out = self.relu(self.fc1(lstm_out))

        out = self.relu(self.fc2(out))

        output = torch.tanh(self.fc3(out))

        return output
