#!/bin/bash -e

# Check Lint issues
echo "Running Flake8..."
flake8 --ignore=--statistics "melidatachall19"

# Check Docstring issues
echo "Running PyDocStyle..."
pydocstyle "${REPO_ROOT_DIR}/magscore"