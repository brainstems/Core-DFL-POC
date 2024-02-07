import numpy as np
import pandas as pd

def generate_data_attributes_csv(num_samples, num_attributes, csv_file_path):
    """
    The attributes are dynamically created based on the num_attributes parameter.
    
    Parameters:
    - num_samples: Number of rows (samples) to generate.
    - num_attributes: Number of attributes (columns) to generate.
    - csv_file_path: Path to save the generated CSV file.
    """
    np.random.seed(42) 
    data = {}

    for i in range(1, num_attributes + 1):
        attr_name = f"Attr{i}"
        
        if i % 5 == 0:
            data[attr_name] = np.random.uniform(0.0001, 1, num_samples).round(2)
        elif i % 4 == 0:
            data[attr_name] = np.random.randint(0, 999999, num_samples)
        elif i % 3 == 0:
            data[attr_name] = np.random.uniform(1, 50.0, num_samples).round(2)
        elif i % 2 == 0:
            data[attr_name] = np.random.randint(0, 10000, num_samples)
        else:
            data[attr_name] = np.random.randint(1, 11, num_samples)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    return csv_file_path  

def generate_target_value_csv(num_samples, csv_file_path):
    """
    Generate a CSV file with synthetic data for attributes based on specified ranges.
    
    Parameters:
    - num_samples: Number of rows (samples) to generate.
    - csv_file_path: Path to save the generated CSV file.
    """
    # Set a seed for numpy's random number generator for reproducibility
    np.random.seed(42)

    # Generate synthetic attributes data
    data = {
        "Target": np.random.randint(100, 1000, num_samples)  
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    return csv_file_path  # Return the path for confirmation

