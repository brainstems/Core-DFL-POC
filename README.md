

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

## Installing TenSEAL

Installing `tenseal` from source on a Mac with an M1 chip can be a bit more involved due to the ARM architecture. Here's a step-by-step guide to help you through the process:

### 1. Prepare Your Environment

Ensure your system is set up for development. This involves having Xcode Command Line Tools installed, which you can install by running:

```sh
xcode-select --install
```

### 2. Install Homebrew

If you haven't already, install Homebrew, which is a package manager for macOS. It will help you install other necessary tools easily. You can install Homebrew by running the following command in the terminal:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. Install Python

Although macOS comes with Python, it's recommended to use a version managed by Homebrew or pyenv for development, as it provides more control over the versioning and avoids permissions issues.

- **Using Homebrew:**

  ```sh
  brew install python
  ```

- **Using `pyenv`:**

  ```sh
  brew install pyenv
  pyenv install 3.9.1  # Replace 3.9.1 with the version you need
  pyenv global 3.9.1
  ```

### 4. Create a Virtual Environment

It's best practice to use a virtual environment for Python projects. This keeps your project's dependencies separate from the system Python. Create and activate a virtual environment by running:

```sh
python3 -m venv tenseal-env
source tenseal-env/bin/activate
```

### 5. Install Required Dependencies

Before building `tenseal` from source, you'll need to install its dependencies. This includes `cmake` and other libraries that might be required for building.

```sh
brew install cmake gmp mpfr
```

### 6. Clone `tenseal` Repository

Next, clone the `tenseal` repository from GitHub:

```sh
git clone https://github.com/OpenMined/TenSEAL.git
cd TenSEAL
```

### 7. Build from Source

Once inside the `TenSEAL` directory, you can build the library. Since you're on an M1 Mac, you might need to ensure that the build process uses the ARM architecture. Set the environment variables if necessary to target the ARM architecture.

```sh
python setup.py build_ext --inplace
```

### 8. Install the Package

After building, install the package:

```sh
pip install .
```

### Troubleshooting

- If you encounter errors related to architecture mismatches (e.g., trying to use x86_64 binaries on ARM), ensure that you're using ARM versions of all dependencies. Homebrew on M1 Macs installs ARM versions by default, but there can be exceptions.
- If you run into issues with missing dependencies or compilation errors, double-check the installed versions of `cmake`, `gmp`, and `mpfr`, and consult the `tenseal` GitHub issues for similar problems.

### Verify Installation

After installation, verify that `tenseal` was installed correctly by running:

```sh
python -c "import tenseal as ts; print(ts.__version__)"
```

This process should install `tenseal` on your M1 Mac. If you encounter specific errors during the installation, they may require more detailed troubleshooting based on the error messages you receive.


## Contributing

Feel free to fork this repository, make changes, and submit pull requests. Any contributions, whether it's improving the documentation, adding more features, or fixing bugs, are always welcome!