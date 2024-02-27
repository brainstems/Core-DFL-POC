def average_weights(models):
    """
    Averages the weights of multiple models.
    
    Args:
    - models (list): List of PyTorch models whose weights need to be averaged.
    
    Returns:
    - A PyTorch model with averaged weights.
    """
    
    # Get the weights of the first model as a reference
    avg_weights = models[0].state_dict()
    
    # Iterate over each model's weights and accumulate the sum
    for model in models[1:]:
        model_weights = model.state_dict()
        for key in avg_weights:
            avg_weights[key] += model_weights[key]
    
    # Divide by the number of models to get the average
    for key in avg_weights:
        avg_weights[key] /= len(models)
    
    # Create a new model with the averaged weights
    averaged_model = type(models[0])()  # This assumes all models are of the same type
    averaged_model.load_state_dict(avg_weights)
    
    return averaged_model

# Function to average encrypted weights
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