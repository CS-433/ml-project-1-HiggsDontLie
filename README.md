Machine Learning Project: Will it rain tomorrow?
==========

Overview
========
This machine learning project helps to predict whether it will rain in Pully tomorrow given previously measured data from the Pully weather station. This code is used for a Kaggle competition and uses linear and non-linear machine learning algorithms as prediction models. <br/>
<br/>
Run program
===========
Files
-----
The user will have to run the main Julia file, ML_project.jl, to be able to obtain the results of the prediction models. This file takes the user step by step in the making process of each of the tested prediction models: Logistic Classifier, K-Nearest Neighbors Classifier, Random Tree Classifier, Neural Network Classifier, and XGBoost Classifier. To obtain a correct outcome, the user need to download the “trainingdata.csv” and the “testdata.csv” provide on Kaggle. Then, the outcome will be two submission files of CSV format. Each of these files correspond to the predictions of one linear model and one non-linear model. In order to run this Julia file, the user must follow these next steps: <br/>
<br/>

Steps
------

1) Open the Julia program command prompt. <br/>

2) In the terminal, activate the packages with the command line: <br/>

        using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

3) Then to open the document, write the command prompt: <br/>

        using Pluto; Pluto.run()

4) In Pluto, open the document by inserting the document path in the appropriate spot. For example: <br/>

        C:\Users\Name\Desktop\ML_project.jl

5) When the document opens in Pluto, it will run all of the required cells to obtain the results of the linear and non-linear models. The submissionModelName.csv files will be created in the same repository as the Julia file. This will enable the user to get the predictions of the linear and non-linear models. <br/>

Directory layout
================

    Directory                           # Main directory
    ├── ML_project.jl
    ├── Project_Will_it_rain_in_Pully.pdf
    ├── trainingdata.csv
    ├── testdata.csv
    ├── submissionModelName.csv
    ├── README.md
    └── ...

Description of files
==============

Non Julia files:
-----------

filename                        | description
--------------------------------|------------------------------------------
README.md                       | Text file (markdown format) how to run the project
Project_Will_it_rain_in_Pully.pdf  | Text file (PDF format) deeper description of the project
trainingdata.csv                        | CSV file, training data for prediction models (Provided in Kaggle)
testdata.csv                    | CSV file, test data to verify accuracy of prediction models (Provided in Kaggle)
submissionModelName.csv                 | CSV file, submission file to insert in Kaggle (One per model)

Julia file:
---------

filename                        | description
--------------------------------|------------------------------------------
ML_project.jl                   |Code for project

Author
=======
Emilie MARCOU & Léa PISTORIUS
