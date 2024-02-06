# Re-importing necessary libraries after code execution state reset
import numpy as np
import pandas as pd

def generate_house_attributes_csv(num_samples, csv_file_path):
    """
    Generate a CSV file with synthetic data for house attributes based on specified ranges.
    
    Parameters:
    - num_samples: Number of rows (samples) to generate.
    - csv_file_path: Path to save the generated CSV file.
    """
    # Set a seed for numpy's random number generator for reproducibility
    np.random.seed(42)

    # Generate synthetic house attributes data
    data = {
        "Size": np.random.randint(600, 3500, num_samples),  # Size in square feet
        "Rooms": np.random.randint(1, 10, num_samples),  # Number of rooms
        "Age": np.random.randint(1, 100, num_samples),  # Age of the property in years
        "Distance": np.random.uniform(0.5, 50.0, num_samples).round(2),  # Distance in miles
        "Bathrooms": np.random.randint(1, 5, num_samples),  # Number of bathrooms
        "Garage": np.random.randint(0, 4, num_samples),  # Garage size in car capacity
        "LotSize": np.random.uniform(0.1, 10.0, num_samples).round(2),  # Lot size in acres
        "Condition": np.random.randint(1, 11, num_samples),  # Overall condition of the house
        "Basement": np.random.randint(0, 2, num_samples),  # Whether the house has a basement
        "SchoolRating": np.random.randint(1, 11, num_samples)  # Rating of nearby schools
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    return csv_file_path  # Return the path for confirmation

def generate_house_prices_csv(num_samples, csv_file_path):
    """
    Generate a CSV file with synthetic data for house attributes based on specified ranges.
    
    Parameters:
    - num_samples: Number of rows (samples) to generate.
    - csv_file_path: Path to save the generated CSV file.
    """
    # Set a seed for numpy's random number generator for reproducibility
    np.random.seed(42)

    # Generate synthetic house attributes data
    data = {
        "Price": np.random.randint(100, 1000, num_samples)  # Price
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    return csv_file_path  # Return the path for confirmation



