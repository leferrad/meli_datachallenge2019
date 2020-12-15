<img style="display: inline;" src="docs/img/logo.png" width="300"/>

*A Python solution for the MercadoLibre Data Challenge 2019*

## Description

This repository is intended to provide a Python library ready to fit and evaluate Machine Learning models to classify categories of items on data of text titles extracted from the MercadoLibre platform. 
In particular, the models are based on Keras neural networks, and the data used was extracted from the MercadoLibre Challenge 2019 (available [on Kaggle](https://www.kaggle.com/abugim/meli-data-challenge-2019)).

The goal of this development was to provide not only a Machine Learning solution with good enough results for this challenge, but also a productive solution that is ready to evolve along the time in production scenarios.

For more details about the work done, check the [document with notes](./docs/notes.md).

## Project structure
                            
      ├── README.md              <- The top-level README for developers using this project
      │      
      ├── melidatachall19        <- Core package of this project.
      │   ├── base               <- Module to provide base abstractions for the development                           
      │   ├── evaluate           <- Module to provide logic for the model evaluation
      │   ├── kb                 <- Module to place all knowledge base of the project, like constants and rules   
      │   ├── model              <- Module to fit the classifier model      
      │   ├── preprocess         <- Module to preprocess text input data      
      │   └── utils              <- Module to provide utils in general       
      │                
      ├── data                   <- Folder for datasets to be used locally  
      │
      ├── docs                   <- Project documentation and resources  
      │
      ├── notebooks              <- Place to store all Jupyter notebooks  
      │
      ├── scripts                <- Scripts to execute defined scenarios 
      │
      ├── tests                  <- Unit and System tests of the core library  
      │
      ├── tools                  <- Tools for the development of this project  
      │
      └─── requirements.txt      <- File that specifies the dependencies for this project


## Setup

You can install this library through the following commands:

```
# Install the library
$ ./tools/install.sh
# Activate the environment
$ source tools/environment.sh
```        

## Tests

Tests are developed using [pytest](https://docs.pytest.org/en/stable/>) and its plugins. To run all tests in ``tests/``, execute:
```
# Run unit tests
$ ./tools/tests.sh unit
# Run system tests
$ ./tools/tests.sh system
# Run all the tests
$ ./tools/tests.sh all
``` 

More details in the [tests's README.md](./tests/README.md)

## Tools

This is the list of tools used to develop in this project.

```
./environment.sh  Script to activate the environment
./hook.sh         Steps to run before a commit
./install.sh      Create the environment, install the dependencies
./lint.sh         Execute code styling checks on the library
./tests.sh        Execute all the tests
```       

## License

This repository is released under the [MIT License](LICENSE). 
