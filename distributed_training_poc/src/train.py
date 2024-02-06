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


# model_peer1 = train(peer1, data_peer1, target_peer1, SimpleNN())
# model_peer2 = train(peer2, data_peer2, target_peer2, SimpleNN())
# model_peer3 = train(peer3, data_peer3, target_peer3, SimpleNN())
