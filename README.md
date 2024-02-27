

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
   pip3 install -r requirements.txt
   ```

## Running the Code

1. **Activate the Virtual Environment** (if you set it up):
   ```bash
   source distributed_training_env/bin/activate
   ```

2. **Install Required Packages**:
   ```bash
   pip3 install -r requirements.txt
   ```


3. **Run the Main Script**:
   ```bash
   python3 main.py
   ```

## Expected Output

Upon successful execution, the script will train models on three virtual peers, aggregate their weights, and then test the unified model. The test loss of the unified model will be printed to the console.

## Data generation

To generate a new set of data:

```sh
cd data
python3 data_gen_dyn.py 
```

After this, you will need to modify the environment variables to set the new generated files.


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

## Run example

After having created and activated the Virtual Environment, the user can create a set of input data for the model training and also some test data.

To do this, the user must follow the example described on `Data generation` section. By just using the default parameters, the output will be:

- data_attributes_100.csv
   - This file is the source attributes file to train the model. It will contain 100 records of 10 attributes each.
- data_target_100.csv
   - This file is the target values file to train the model. It will contain 100 records of 1 attributes each.
- data_attributes_test_10.csv
   - This file is the source attributes file to test the model. It will contain 10 records of 10 attributes each.
- data_target_test_10.csv
   - This file is the target values file to test the model. It will contain 10 records of 1 attributes each.

If the user inspects the files, it will have generated a random set of values on each of the files, preserving the format. The amount of records and attributes can be changed, but both, source and test files, must have the same amount of attributes.

After doing this, the user must specify the file names on the `.env` file

```sh
ATTRIBUTES_CSV_PATH=./data/data_attributes_100.csv
TARGET_CSV_PATH=./data/data_target_100.csv
TEST_ATTRIBUTES_CSV_PATH=./data/data_attributes_test_10.csv
TEST_TARGET_CSV_PATH=./data/data_target_test_10.csv
```

Output for the files should look like this:

`data_attributes_100.csv`
```
Attr1,Attr2,Attr3,Attr4,Attr5,Attr6,Attr7,Attr8,Attr9,Attr10,Attr11,Attr12,Attr13,Attr14,Attr15,Attr16,Attr17,Attr18,Attr19,Attr20
7,3104,20.54,56958,0.69,5.49,8,912259,17.8,0.39,6,35954,2,271,0.57,243373,1,29.39,9,0.35
4,7215,41.01,82074,0.06,30.52,1,894647,24.22,0.53,8,27856,9,3511,0.27,347697,8,33.47,7,0.24
...
```

`data_target_100.csv`
```
Target
202
535
...
```

`data_attributes_test_10.csv`
```
Attr1,Attr2,Attr3,Attr4,Attr5,Attr6,Attr7,Attr8,Attr9,Attr10,Attr11,Attr12,Attr13,Attr14,Attr15,Attr16,Attr17,Attr18,Attr19,Attr20
7,8322,30.97,591723,0.45,41.83,2,348951,9.1,0.93,6,924414,3,9914,0.12,905533,8,15.74,10,0.63
4,1685,1.35,319030,0.01,9.49,10,274329,1.77,0.65,3,897421,7,3157,0.36,790180,9,14.96,6,0.63
...
```

`data_target_test_10.csv`
```
Target
202
535
...
```

With everything in place, now the user can run the training and test phases (please refer to `Running the Code` section).

The code standardizes the input values, either for training or testing.

After standardizing the target values, they have a mean of `0` and a standard deviation of `1`. This transformation alters the scale and distribution of the target values, making them unitless and centered around `0`.

From the input we have generated, the standard deviation could be a value proximate to `263`.

In the standardized scale, where most data points should fall within a few standard deviations from the mean (assuming a normal distribution), an MSE of `X` suggests that the model's predictions are, on average, about sqrt(X) (the square root of X) standard deviations away from the actual standardized values.

Being said that, we should expected an output value between `0` and `2`. This value represents the evaluation of the testing model using the Mean Squared Error (MSE) loss approach. The program will average the loss over all test samples.

For example:
An MSE of 1.5 suggests that the model's predictions are, on average, about 1.22 (the square root of 1.5) standard deviations away from the actual standardized values, which is about `320`.

The closer to 0, the better.

This would be the output example for the values we have defined before:

```sh
Welcome:
Staging Protocol Changes...

Starting my-bs-test server on 0.0.0.0:8080

WARNING: private key is based on node name: my-bs-test in dev_mode. Don't run this in production.
SQLite Store Path:
!open file:///var/folders/4f/_4qmrlvd6kz24t32jc2p6xzw0000gn/T/a2dfae66b7bb4b72a8d06740ed81ae06.sqlite

Creating default worker image with tag='local-dev'
Building default worker image with tag=local-dev
Setting up worker poolname=default-pool workers=0 image_uid=2c4ae1e48b244f848efccb217d39a1ea in_memory=True
Created default worker pool.
Data Migrated to latest version !!!
INFO:     Started server process [45362]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     127.0.0.1:54783 - "GET /api/v2/metadata HTTP/1.1" 200 OK
Waiting for server to start... Done.
INFO:     127.0.0.1:54785 - "GET /api/v2/metadata HTTP/1.1" 200 OK
INFO:     127.0.0.1:54785 - "GET /api/v2/metadata HTTP/1.1" 200 OK
INFO:     127.0.0.1:54785 - "POST /api/v2/login HTTP/1.1" 200 OK
Creating default worker image with tag='local-dev'
Building default worker image with tag=local-dev
Setting up worker poolname=default-pool workers=0 image_uid=03092643c7ee48caa5ff9e31dc3c9844 in_memory=True
Created default worker pool.
Creating default worker image with tag='local-dev'
Building default worker image with tag=local-dev
Setting up worker poolname=default-pool workers=0 image_uid=9bcbd3b564d949ed824dfa8a1069d684 in_memory=True
Created default worker pool.
Creating default worker image with tag='local-dev'
Building default worker image with tag=local-dev
Setting up worker poolname=default-pool workers=0 image_uid=9c3ef69fbe9244a9880de0251d31bb67 in_memory=True
Created default worker pool.
Workers created: 3
Training starts
Averaging finished.
Testing....
Test loss of the unified model: 1.2039868354797363
```

This means that we have a deviation on the prediction of `1.09 * (263) = 315` for values between 100 and 1000. This is a moderate error result. 