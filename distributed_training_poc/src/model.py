import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_features):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.fc(x)