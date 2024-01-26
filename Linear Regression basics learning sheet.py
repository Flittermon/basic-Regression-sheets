# LINEAR REGRESSION BASICS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Intro to linear regression
"""
from sklearn.datasets import load_diabetes

#What is a linear regression model?

#We try to predict values (features) on the basis of data
#so lets say for example we have a graph of the height of men
#and we want to predict the height of their children
#we could do that, because there is a direct correlation between the fathers height and the sons height

#the load_diabetes dataset uses a lot of different valuables to predict the chance of someone getting diabetes. For example:
# - weight
# - bloodpressure
# - age

#most basic prediciton model

#Load our dataset
diabetes_data = load_diabetes()

#get a pandas dataframe
diabetes_df = pd.DataFrame(diabetes_data.data, columns= diabetes_data.feature_names)

#my_plot = diabetes_df.plot(kind="bar")
#plt.show()

#predict the target

#add the target
diabetes_df["target"] = diabetes_data.target

#show the first rows
print(diabetes_df.head())

#the linear regression model puts an average line between a scatter plot.
#The prediction is then made for y -> x * linearregression, for x = feature and y = target
"""

#train test split and Model creation
"""

from sklearn.datasets import load_diabetes

diabetes_data = load_diabetes()
diabetes_df = pd.DataFrame(diabetes_data.data, columns= diabetes_data.feature_names)
diabetes_df["target"] = diabetes_data.target

#split into X and y

X = diabetes_data.data
y = diabetes_data.target

print(X.shape)          # there are 442 data entries and 10 different data points / columns
print(y.shape)          # there are 442 data entries and 1 target column which makes sense because we want to predict one outcome (chance for diabetes)

# split the data into training and testing data

# -> train the model on the train data
# -> test how good the model is on the test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)  # 20 percent of the data is test data

# print out the shape of the training and testing sets

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#the data has been split into 4 variables

#now we create a linear regression model

from sklearn.linear_model import LinearRegression

#create the model:

my_model = LinearRegression()

#train the model

my_model.fit(X_train, y_train)

#predict on the testing set
#y_pred are the predicted target based on the X_test features

y_pred = my_model.predict(X_test)       # --> test the output of the model (y) on the remaining test data (X_test)

#after that we can compare the y_pred values to the actual y values (y_test)
#the closer these two align, the better our model is

print(y_pred.shape)     # 89 values 

#things needed for evaluation:
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#evaluate

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
intercept = my_model.intercept_

print(r2)
print(mse)
print(mae)
print(intercept)

#the r2 is a value between 0 and 1, the closer the value is to 1 means how much there is a correlation between the features and the target
# --> how well the model fits the data
#the mse is the average swaured distance between the predicted and actual values
# --> lower is better (the data is closer to the line / more accurate)
#the mae is the same thing but not squared, means it can have negative values
# --> lower is better (the data is closer to the line / more accurate)
#the intercept is the starting point of the regression line on the y-axis
# --> value of target when the features are 0, if positive: target increases when features increase, if negative the opposite
#   --> if age goes up, diabetes goes up, if weight goes up diabetes goes up --> positive
#   --> if speed goes up, controllabity goes down, if weight goes up speed goes down --> negative

#visualize the data

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha= 0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.show()

#plot the residuals
#the residuals are plotted against the predicted values of the target variable
#If the linear regression model is a good fit for the data,
#the residual plot should show a random scatter of the points around zero
#With no discernable trend

plt.scatter(y_pred, y_test - y_pred, alpha = 0.5)
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.show()
"""

#another example
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

cancer_data = fetch_california_housing()

X = cancer_data.data
y = cancer_data.target

print(X.shape, y.shape)

#split the data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.25)

#create the model

my_model = LinearRegression()
my_model.fit(X_train, y_train)

y_pred = my_model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
intercept = my_model.intercept_

print(r2)
print(mae)
print(mse)
print(intercept)

#visualize the data

plt.scatter(y_test, y_pred, alpha = 0.5)
plt.plot([y.min(), y.max()],[y.min(), y.max()], color="red")
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.show()
"""

