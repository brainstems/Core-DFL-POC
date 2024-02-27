import torch
import torch.nn.functional as F

def train(model, data, target, optimizer, epochs=5):
    criterion = torch.nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return model
