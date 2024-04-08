import torch
import torch.nn as nn


class EnhancedLSTM(nn.Module):
    def __init__(self, input_size=12289, hidden_size=512, num_layers=2, output_size=3, device='cuda'):
        super(EnhancedLSTM, self).__init__()
        self.device = device  # Добавляем параметр устройства
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc1 = nn.Linear(hidden_size, 256).to(device)
        self.relu = nn.ReLU().to(device)
        self.fc2 = nn.Linear(256, 128).to(device)
        self.fc3 = nn.Linear(128, output_size).to(device)
        self.tanh = nn.Tanh().to(device)  # Добавляем Tanh как функцию активации для выходного слоя

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = hn[-1]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  # Применяем Tanh к выходу последнего линейного слоя
        return x
