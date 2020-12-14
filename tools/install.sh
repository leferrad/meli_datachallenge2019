#!/bin/bash -e

# Create virtualenv "venv"
virtualenv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the library
python setup.py install