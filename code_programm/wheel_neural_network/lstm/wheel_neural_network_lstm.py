import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, input_size=12288, hidden_size=128, num_layers=2, speed_size=1, output_size=1):
        super(LSTMNet, self).__init__()
        self.device = 'cuda'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size  # Add this line
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_size + speed_size, output_size).to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, image, speed):
        image = image.to(self.device)
        speed = speed.to(self.device)
        batch_size = 1  # Assuming a single sample
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Reshape the input tensor to match the expected input_size
        image = image.view(batch_size, self.input_size)

        out, _ = self.lstm(image.unsqueeze(1), (h0, c0))
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Unsqueeze the speed tensor to add a dimension
        speed = speed.unsqueeze(1)  # Shape: (batch_size, 1)

        out = torch.cat((out, speed), dim=1).to(self.device)

        out = self.fc(out).to(self.device)
        output = self.tanh(out).to(self.device)
        return output
