Machine Learning Project: Higgs Boson?
==========

Overview
========
This machine learning project helps to predict whether a given signature, resulting from the crash 
of particules into each other, is a Higgs boson or not. <br/>
<br/>


Directory layout
================

    Directory                           # Main directory
    ├── cross_validation.py
    ├── helpers.py
    ├── implementations.py
    ├── model_selection.py
    ├── plots
    ├── plots.py
    ├── project_1.ipynb
    ├── README.md
    ├── run.py
    └── Report_Project1_HiggsBoson.pdf

To properly run our project, we expect the train.csv to be placed at this level, i.e., at the same level as e.g., 
helpers.py or run.py, in the main directory

Description of contents
==============

Directories:
---------
Directory name                  | description
--------------------------------|------------------------------------------
plots           			    |Contains all the plots of the preliminary data visualization and of the tuning of hyper-parameters. Every plot that is run in our scripts will be saved in this directory 


Non Python files:
-----------

filename                        | description
--------------------------------|------------------------------------------
README.md                       | Text file (markdown format) describing all the files of the project
Project1_Higgs_boson.pdf        | Project report containing our results and analyse on the challenge
train.csv                       | Data used to train our models


Python files:
---------

filename                        | description
--------------------------------|------------------------------------------
cross_validation.py             |Set of functions that perform the k_fold-cross validation of the different prediction models from implementations.py
helpers.py                      |Set of useful functions used throughout the project
implementations.py              |Two data preprocessing function as well as our model prediction functions. These include Least Squares, Ridge Regression, Gradient Descend, Stochastic Gradient Descend, Polynomial Regression, Logistic Regression, Break Logistic Regression, Regularized Logistic Regressio and Break Regularized Logistic Regresssion
model_selection.py              |File containing the resulting MSE computed by different hyperparameters.
plots.py                        |Functions used to plot some of our results
run.py                          |File that runs a our best model and creates the submission file for AICrowd
project_1.ipynb	                |Initial data visualisation, tuning of hyper-parameters and model comparison to find the best prediction model

Authors
=======
Emilie MARCOU, Lucille NIEDERHAUSER & Anabel SALAZAR DORADO
