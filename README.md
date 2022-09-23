### Overview

This project started with the regression lesson of [ML-zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp). 

In this project I built a framework to train and validate a regression model based on normal form.
I built an abstract class with an abstract method `extract_features` which receives a dataframe and extract the matrix of features. Then everything else: training and prediction is implemented in abstract class because this is common. In this way I avoid the duplication of the code along the notebook.

All the compared models defferenciate from each in other in the way they build the matrix of features.