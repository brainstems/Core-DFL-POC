import tenseal as ts
import torch

# Function to encrypt model weights using TenSEAL
def encrypt_weights(model, context):
    encrypted_weights = {}
    for name, param in model.named_parameters():
        param_list = param.data.view(-1).tolist()
        encrypted_weights[name] = ts.ckks_vector(context, param_list)
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
