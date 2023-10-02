import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def test_model(model, test_data, test_target):
    test_dataset = TensorDataset(test_data, test_target)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()

    test_loss /= len(test_loader.dataset)
    return test_loss