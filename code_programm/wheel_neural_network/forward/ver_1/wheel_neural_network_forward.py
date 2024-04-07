import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class FeedforwardNet(nn.Module):
    def __init__(self, input_size=12288, hidden_size_1=256, hidden_size_2=128, hidden_size_3=64, speed_size=1,
                 output_size=1):
        super(FeedforwardNet, self).__init__()
        self.device = 'cuda'
        self.fc1 = nn.Linear(input_size, hidden_size_1).to(self.device)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2).to(self.device)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3).to(self.device)
        self.fc4 = nn.Linear(hidden_size_3 + speed_size, output_size).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, image, speed):
        image = image.to(self.device)
        speed = speed.to(self.device)
        out = self.fc1(image).to(self.device)
        out = self.relu(out).to(self.device)
        out = self.fc2(out).to(self.device)
        out = self.relu(out).to(self.device)
        out = self.fc3(out).to(self.device)
        out = self.relu(out).to(self.device)

        out = torch.cat((out, speed), dim=0).to(self.device)

        out = self.fc4(out).to(self.device)
        output = self.tanh(out).to(self.device)
        return output.detach().cpu()
