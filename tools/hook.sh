#!/bin/bash -e

# Run Linter checks
echo "Running Linter..."
./tools/lint.sh || { echo "Code style issues found, fix them before the commit"; exit 1; }

# Run Tests
echo "Executing Tests..."
./tools/tests.sh unit || { echo "Test errors found, fix them before the commit"; exit 1; }