

# Federated Learning PoC

This project demonstrates a proof of concept for federated learning using PySyft and PyTorch. The goal is to train models on distributed nodes and then aggregate their knowledge into a single global model.

## Prerequisites

- Python 3.11
- Virtual environment (recommended)

## Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/fedML.git
   cd fedML/distributed_training_poc
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   ```bash
   python3 -m venv distributed_training_env
   source distributed_training_env/bin/activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

1. **Activate the Virtual Environment** (if you set it up):
   ```bash
   source distributed_training_env/bin/activate
   ```

2. **Run the Main Script**:
   ```bash
   python main.py
   ```

## Expected Output

Upon successful execution, the script will train models on three virtual peers, aggregate their weights, and then test the unified model. The test loss of the unified model will be printed to the console.

## Contributing

Feel free to fork this repository, make changes, and submit pull requests. Any contributions, whether it's improving the documentation, adding more features, or fixing bugs, are always welcome!
