import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset, and splitting it into the features matrix and the dependent variable
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training data with Simple Linear Regression Model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting using the test set
test_value = regressor.predict(x_test)
print(y_test)
print(test_value)
