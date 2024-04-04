import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, input_size=12288, hidden_size=64, num_layers=1, speed_size=1, output_size=1):
        super(LSTMNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_size + speed_size, output_size).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, image, speed):
        image = image.to(self.device)
        speed = speed.to(self.device)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(image.unsqueeze(0), (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Concatenate with speed
        out = torch.cat((out, speed), dim=1)

        # Fully connected layer
        out = self.fc(out)

        # Activation function
        out = self.tanh(out)

        return out
