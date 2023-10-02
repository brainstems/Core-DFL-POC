#!/bin/bash

# Create main project directory
mkdir distributed_training_poc
cd distributed_training_poc

# Create virtual environment directory
mkdir env

# Create data directory
mkdir data
touch data/train_data.csv
touch data/test_data.csv

# Create models directory
mkdir models
touch models/model_peer1.pth
touch models/model_peer2.pth
touch models/model_peer3.pth

# Create source code directory
mkdir src
touch src/__init__.py
touch src/model.py
touch src/train.py
touch src/aggregate.py
touch src/test.py

# Create main script and requirements file
touch main.py
touch requirements.txt

echo "Directory structure and files created successfully!"
