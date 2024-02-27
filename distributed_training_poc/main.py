# main.py

# Import necessary libraries and modules
import torch
from src.model import SimpleNN
from src.train import train 
from src. standardize import standardize 
from src.test import test_model
from src.encrpt_decrypt import *
from src.aggregate import *
import pandas as pd
from dotenv import load_dotenv
import os
import syft as sy
import tenseal as ts


# Load env variables
load_dotenv()

data_path = os.getenv("DATA_PATH")
peers = os.getenv("PEERS")
attributes_csv_path = os.getenv("ATTRIBUTES_CSV_PATH")
target_csv_path = os.getenv("TARGET_CSV_PATH")
test_attributes_csv_path = os.getenv("TEST_ATTRIBUTES_CSV_PATH")
test_target_csv_path = os.getenv("TEST_TARGET_CSV_PATH")
num_attributes = os.getenv("NUM_ATTRIBUTES")

# Function to create a TenSEAL context
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

# Main function
def main():

    # Connect to the launched server as a client
    sy.login(port=8080, email="inmind.desa@gmail.com", password="123456")

    # All your code here... 
    workers = [sy.Worker(name=f"peer{i+1}") for i in range(int(peers))]
    print(f"Workers created: {peers}")

    # Load the data from CSV files
    attributes_source = pd.read_csv(attributes_csv_path)
    target_source = pd.read_csv(target_csv_path)

    # Convert the DataFrame to PyTorch tensors
    data = torch.tensor(attributes_source.values, dtype=torch.float32)
    target = torch.tensor(target_source.values, dtype=torch.float32).view(-1, 1)  # Reshape target to match expected dimensions
    
    # Normalization/Standardization
    standardized_data = standardize(data)
    standardized_target = standardize(target)

    # Split the data for each peer
    data_splits = torch.chunk(standardized_data, int(peers))
    target_splits = torch.chunk(standardized_target, int(peers))

    # Create TenSEAL context
    context = create_context()

    print(f"Training starts:")
    # Train models on each peer
    encrypted_weights_list = []
    for i, worker in enumerate(workers):
        # Initialize your model
        model = SimpleNN(input_features=int(num_attributes))
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Simulate training on data for each worker
        model = train(model, data_splits[i], target_splits[i], optimizer, epochs=5)
        
        # Collecting Parameter Shapes Before Encryption
        # Before encrypting the weights, you can collect the shapes of the model's parameters to use later during decryption. 
        # Update the part of your code where you perform encryption to also store parameter shapes
        # Right before encrypting model weights
        param_shapes = {name: param.size() for name, param in model.named_parameters()}
        
        # Encrypt model weights after training
        encrypted_weights = encrypt_weights(model, context)
        encrypted_weights_list.append(encrypted_weights)
    
    # Aggregate encrypted weights
    avg_encrypted_weights = average_encrypted_weights(encrypted_weights_list)
    
    # Decrypt aggregated weights
    decrypted_avg_weights = decrypt_weights(avg_encrypted_weights, context, param_shapes)

    # Apply decrypted averaged weights to the global model
    
    global_model = SimpleNN(input_features=int(num_attributes))
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in decrypted_avg_weights:
                param.copy_(decrypted_avg_weights[name])
            if not torch.isfinite(param).all():
                print(f"Non-finite weights detected in {name}")
            

    print(f"Averaging finished.")
    print(f"Testing....")

    # Load the data from CSV files
    attributes_source = pd.read_csv(test_attributes_csv_path)
    target_source = pd.read_csv(test_target_csv_path)

    # Convert the DataFrame to PyTorch tensors
    test_data = torch.tensor(attributes_source.values, dtype=torch.float32)
    test_target = torch.tensor(target_source.values, dtype=torch.float32).view(-1, 1)  # Reshape target to match expected dimensions 
    
    # Normalization/Standardization
    standardized_test_data = standardize(test_data)
    standardized_test_target = standardize(test_target)

    test_loss = test_model(global_model, standardized_test_data, standardized_test_target)
    
    print("Test loss of the unified model: " + str(test_loss))

    
# Run the main function
if __name__ == "__main__":
    # This ensures the script runs only when executed directly
    print(f"Welcome:")
    # Launch a Syft server
    node = sy.orchestra.launch(name="my-jr-test", port=8080, dev_mode=True, reset=True)
    main()



