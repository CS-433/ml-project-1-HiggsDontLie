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
    ├── model_selection.py
    ├── plots.py
    ├── README.md
    ├── train.csv
    └── run.py


Description of files
==============

Non Python files:
-----------

filename                        | description
--------------------------------|------------------------------------------
README.md                       | Text file (markdown format) describing all the files of the project
train.csv                        | CSV file, training data for the prediction models (found on AICrowd)

Python files:
---------

filename                        | description
--------------------------------|------------------------------------------
cross_validation.py                   |Set of functions that perform the k_fold-cross validation of the different prediction models from implementations.py
helpers.py                   |Set of useful functions used throughout the project
implementations.py                  | Data preprocessing function as well as our model rediction functions. These include Least Squares, Ridge Regression, Gradient Descend, Stochastic Gradient Descend and Polynomial Regression
model_selection.py                   |File where we compared the different errors of our cross-validation models to pick the one with the smallest error, i.e. the best prediction model
plots.py                   |Function used to plot the resulting mses and visualize the shape of the mses according to a particular parameter e.g., the stepsize gamma 
run.py                   |File that runs a selected model and creates the submission file for AICrowd

Authors
=======
Emilie MARCOU, Lucille NIEDERHAUSER & Anabel SALAZAR DORADO
