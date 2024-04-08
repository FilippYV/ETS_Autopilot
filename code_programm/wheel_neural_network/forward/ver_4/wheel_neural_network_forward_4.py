import torch
import torch.nn as nn


class FeedforwardNet(nn.Module):
    def __init__(self, input_size=12289, hidden_size_1=512, hidden_size_2=256, hidden_size_3=128,
                 output_size=2):
        super(FeedforwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input_data):
        out = self.fc1(input_data)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.tanh(out)
        return out.squeeze(0).squeeze(0)

