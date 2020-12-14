#!/bin/bash -e

# Execute tests
if [[ "$1" == "unit" ]]
then
  echo "Executing unit tests"
  pytest -rsxX --cov=melidatachall19 --cov-report=term --cov-report=html:tests/coverage ./tests/unit --cache-clear
elif [[ "$1" == "system" ]]
then
  echo "Executing system tests"
  pytest -rsxX --cov=melidatachall19 --cov-report=term --cov-report=html:tests/coverage ./tests/system --cache-clear
elif [[ "$1" == "all" ]]
then
  echo "Executing unit + system tests"
  pytest -rsxX --cov=melidatachall19 --cov-report=term --cov-report=html:tests/coverage ./tests/ --cache-clear
else
  # Unit tests by default
  echo "Executing unit tests"
  pytest -rsxX --cov=melidatachall19 --cov-report=term --cov-report=html:tests/coverage ./tests/unit --cache-clear
fi
