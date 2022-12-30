import torch
import torch.nn as nn
import torch.optim as optim


class SimpleFCN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleFCN, self).__init__()
        self.bounds = (0, 1)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
