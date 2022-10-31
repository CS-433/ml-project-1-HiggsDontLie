Machine Learning Project: Higgs Boson?
==========

Overview
========
This machine learning project helps to predict whether a given signature, resulting from the crash 
of protons into each other, is a Higgs boson or not. <br/>
<br/>


Directory layout
================

    Directory                           # Main directory
    ├── cross_validation.py
    ├── helpers.py
    ├── implementations.py
    ├── plots
    ├── plots.py
    ├── project_1.ipynb
    ├── README.md
    ├── train.csv
    ├── run.py
    └── Higgs boson?.pdf



Description of contents
==============

Directories:
---------
Directory name                  | description
--------------------------------|------------------------------------------
plots           			  |Contains all the plots of the preliminary data visualization and of the tuning of hyper-parameters


Non Python files:
-----------

filename                        | description
--------------------------------|------------------------------------------
README.md                       | Text file (markdown format) describing all the files of the project
train.csv                       | CSV file, training data for the prediction models (found on AICrowd)

Python files:
---------

filename                        | description
--------------------------------|------------------------------------------
cross_validation.py             |Set of functions that perform the k_fold-cross validation of the different prediction models from implementations.py
helpers.py                   	  |Set of useful functions used throughout the project
implementations.py              |Data preprocessing function as well as our model prediction functions. These include Least Squares, Ridge Regression, Gradient Descend, Stochastic Gradient Descend, Polynomial Regression, Logistic Regression and Regularized Logistic Regression
plots.py                        |Functions used to plot some of our results
run.py                          |File that runs a our best model and creates the submission file for AICrowd
project_1.ipynb	              |Initial data visualisation, tuning of hyper-parameters and model comparison to find the best prediction model

Authors
=======
Emilie MARCOU, Lucille NIEDERHAUSER & Anabel SALAZAR DORADO
