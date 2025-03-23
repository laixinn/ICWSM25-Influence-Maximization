import torch.nn as nn
import torch


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits


class MLP(nn.Module):
    def __init__(self, dim, n_classes=2, drop=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.fc_1 = nn.Linear(dim, 80)
        self.fc_2 = nn.Linear(80, 10)
        self.fc_3 = nn.Linear(10, n_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop, inplace=False)

    def forward(self, g, x, *args):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)