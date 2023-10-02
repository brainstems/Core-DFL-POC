# main.py

# Import necessary libraries and modules
import torch
import torch.nn.functional as F
from src.model import SimpleNN
from src.train import train 
from src.test import test_model
from src.aggregate import average_weights

# Initialize PySyft and virtual workers (peers)
import syft as sy

def main():
    # All your code here...
    hook = sy.TorchHook(torch)
    peer1 = sy.VirtualWorker(hook, id="peer1")
    peer2 = sy.VirtualWorker(hook, id="peer2")
    peer3 = sy.VirtualWorker(hook, id="peer3")

    # Load and split the dataset (for simplicity, we'll use dummy data here)
    # In a real-world scenario, you'd load your dataset from the 'data/' directory
    data = torch.randn(300, 10)  # Dummy data with 300 samples, 10 features each
    target = torch.randn(300, 1)  # Dummy target values

    # Split the data for each peer
    data_splits = torch.chunk(data, 3)
    target_splits = torch.chunk(target, 3)

    # Train the model on each peer
    model_peer1 = train(peer1, data_splits[0], target_splits[0], SimpleNN())
    model_peer2 = train(peer2, data_splits[1], target_splits[1], SimpleNN())
    model_peer3 = train(peer3, data_splits[2], target_splits[2], SimpleNN())

    # Aggregate the models
    global_model = average_weights([model_peer1, model_peer2, model_peer3])

    # Test the unified model (using a dummy test set for simplicity)
    test_data = torch.randn(100, 10)
    test_target = torch.randn(100, 1)
    test_loss = test_model(global_model, test_data, test_target)

    print(f"Test loss of the unified model: {test_loss}")

    if __name__ == "__main__":
        # This ensures the script runs only when executed directly
        main()