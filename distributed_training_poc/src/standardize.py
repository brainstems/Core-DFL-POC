import torch

def standardize(data):
    
    # Standardization
    # It ensures each feature contributes equally to the model's input layer, avoiding potential issues with large value ranges.

    mean = data.mean(0, keepdim=True)
    std = data.std(0, keepdim=True)
    standardized_data = (data - mean) / std

    # Ensure 'std' is not zero to avoid division by zero errors; add a small epsilon if necessary
    std = std.clamp(min=1e-6)
    standardized_data = (data - mean) / std

    # Check for non-finite values
    if not torch.isfinite(standardized_data).all():
        print("Non-finite values found in test data")

    return standardized_data


# For Input Data (Features):
# Standard Practice: It's standard practice to normalize or standardize input features. This helps in speeding up the training process, achieving faster convergence, and avoiding issues related to numerical instability, such as exploding gradients.

# For Target Data (Labels):
# Regression Tasks: In regression tasks, where the target variable is a continuous value (like house prices), it can be beneficial to normalize or standardize the target data, especially if the range of target values is large or if it significantly differs from the input features in scale.
#   Normalization: Can be used if you know the specific range of your target variable and want to scale predictions into this range.
#   Standardization: Useful when you don't want to assume a specific range for your target variable but wish to keep the target values centered and scaled based on their distribution.
#   Reversing the Process: After predictions, you may need to invert the scaling (denormalize or destandardize) to interpret the model's outputs in their original scale.
# Classification Tasks: For categorical targets (classification tasks), normalization or standardization of the target is not applicable because the targets are often encoded as one-hot vectors or integers representing different classes.

# If you decide to scale the target values for a regression task, you can apply a similar approach as with the input features but remember to:
#   Use separate scalers for the input features and the target variable to avoid information leak and ensure correct scaling.
#   Invert the scaling of the predictions to interpret the model's output in the original scale of the target variable.



