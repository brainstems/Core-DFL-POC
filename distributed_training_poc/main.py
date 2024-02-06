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
import tenseal as ts

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

# Function to encrypt model weights using TenSEAL
def encrypt_weights(model, context):
    encrypted_weights = {}
    for name, param in model.named_parameters():
        original_weight = param.data

        param_list = param.data.view(-1).tolist()
        encrypted_weights[name] = ts.ckks_vector(context, param_list)

        # print(f"Original: " ,  original_weight)
        # print(f"Encrypted: " , encrypted_weights[name].serialize()[:8])

    return encrypted_weights

# Function to decrypt model weights
def decrypt_weights(encrypted_weights, context, param_shapes):
    decrypted_weights = {}
    for name, ew in encrypted_weights.items():
        # Decrypt the encrypted parameter
        decrypted_list = ew.decrypt()
        # Reshape the decrypted list into its original tensor shape
        shape = param_shapes[name]  # Get the shape for the current parameter
        decrypted_weights[name] = torch.tensor(decrypted_list).view(shape)
    return decrypted_weights


def average_encrypted_weights(encrypted_weights_list):
    avg_encrypted_weights = {}
    num_peers = len(encrypted_weights_list)
    # Calculate the reciprocal of num_peers as a floating-point number
    reciprocal = 1.0 / num_peers
    
    for name in encrypted_weights_list[0].keys():
        # Initialize averaged weights with the first set of weights to start accumulation
        avg_encrypted_weights[name] = encrypted_weights_list[0][name]
        
        # Accumulate the rest of the weights
        for ew in encrypted_weights_list[1:]:
            avg_encrypted_weights[name] += ew[name]
        
        # Multiply by the reciprocal to average
        avg_encrypted_weights[name] *= reciprocal
    
    return avg_encrypted_weights



def main():

    print(f"Start:")
    
    # Connect to the launched server as a client
    domain_client = sy.login(port=8080, email="julque@gmail.com", password="123456")

    # All your code here... 
    workers = [sy.Worker(name=f"peer{i+1}") for i in range(3)]
    
    # Set a fixed seed for PyTorch's random number generator
    torch.manual_seed(42)

    # Example setup
    print(f"Workers created:")
    # Load and split the dataset (for simplicity, we'll use dummy data here)
    # In a real-world scenario, you'd load your dataset from the 'data/' directory
    data = torch.randn(300, 10)  # Dummy data with 300 samples, 10 features each
    target = torch.randn(300, 1)  # Dummy target values
    print(f"Data generated:")
    # Split the data for each peer
    data_splits = torch.chunk(data, 3)
    target_splits = torch.chunk(target, 3)

    # Create TenSEAL context
    context = create_context()

    print(f"Training starts:")
    # Train models on each peer
    encrypted_weights_list = []
    for i, worker in enumerate(workers):
        # Initialize your model
        model = SimpleNN()
        print("START TRAINING MODEL: ", i)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Simulate training on data for each worker
        model = train(model, data_splits[i], target_splits[i], optimizer, epochs=5)
        
        # Right before encrypting model weights
        param_shapes = {name: param.size() for name, param in model.named_parameters()}
        
        # Encrypt model weights after training
        encrypted_weights = encrypt_weights(model, context)

        encrypted_weights_list.append(encrypted_weights)
        print("END TRAINING MODEL: ", i)
    
    # All client Nodes will send the encrypted weigths to the Aggregator Node
    # From NOW ON, this is going to happen in the Aggregator Node
        
    # Aggregate encrypted weights
    avg_encrypted_weights = average_encrypted_weights(encrypted_weights_list)
    
    # Decrypt aggregated weights
    decrypted_avg_weights = decrypt_weights(avg_encrypted_weights, context, param_shapes)

    # Apply decrypted averaged weights to the global model
    
    global_model = SimpleNN()
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in decrypted_avg_weights:
                param.copy_(decrypted_avg_weights[name])
                print(f"Decrypted: " , param.data)
            

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
    # Launch a Syft server
    node = sy.orchestra.launch(name="my-jr-test", port=8080, dev_mode=True, reset=True)
    main()



