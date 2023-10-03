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
# Launch a Syft server
node = sy.orchestra.launch(name="my-fl-test", port=8080, dev_mode=True, reset=True)

def main():

    print(f"Start:")

    # Connect to the launched server as a client
    domain_client = sy.login(port=8080, email="omarsaad@hotmail.com", password="123456")

    # All your code here... 
    peer1 = sy.VirtualMachine(name="peer1")
    peer2 = sy.VirtualMachine(name="peer2")
    peer3 = sy.VirtualMachine(name="peer3")
    print(f"Workers created:")
    # Load and split the dataset (for simplicity, we'll use dummy data here)
    # In a real-world scenario, you'd load your dataset from the 'data/' directory
    data = torch.randn(300, 10)  # Dummy data with 300 samples, 10 features each
    target = torch.randn(300, 1)  # Dummy target values
    print(f"Data generated:")
    # Split the data for each peer
    data_splits = torch.chunk(data, 3)
    target_splits = torch.chunk(target, 3)

    print(f"Training starts:")
    # Train the model on each peer
    model_peer1 = train(peer1, data_splits[0], target_splits[0], SimpleNN())
    model_peer2 = train(peer2, data_splits[1], target_splits[1], SimpleNN())
    model_peer3 = train(peer3, data_splits[2], target_splits[2], SimpleNN())
    print(f"Training finished, starting averaging:")
    # Aggregate the models
    global_model = average_weights([model_peer1, model_peer2, model_peer3])
    print(f"Averaging finished.")
    print(f"Testing....")
    # Test the unified model (using a dummy test set for simplicity)
    test_data = torch.randn(100, 10)
    test_target = torch.randn(100, 1)
    test_loss = test_model(global_model, test_data, test_target)

    print(f"Test loss of the unified model: {test_loss}")

if __name__ == "__main__":
    # This ensures the script runs only when executed directly
    print(f"Welcome:")
    main()