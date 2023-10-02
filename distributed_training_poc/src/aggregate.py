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
