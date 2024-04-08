import torch
import torch.nn as nn


class FeedforwardNet(nn.Module):
    def __init__(self, input_size=12288, hidden_size_1=256, hidden_size_2=256, hidden_size_3=128,
                 hidden_size_4=64, output_size=2):
        super(FeedforwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3 + 1, hidden_size_4)
        self.fc5 = nn.Linear(hidden_size_4, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, image, speed):
        out = self.fc1(image)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(torch.cat((out, speed), dim=0))
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.tanh(out)
        return out

