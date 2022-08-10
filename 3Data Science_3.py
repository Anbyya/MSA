import pandas as pd

# Just one of the models capable of modelling the dataset.
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#Select training model
X, y = make_regression(n_samples=10000, n_features=10, noise=100, random_state=42)

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
#Fit tranining dataset
model.fit(trainX, trainY)
#Test dataset score
model.score(testX, testY)
#Printout the model predictions
print(model.predict([[1,1,1,2,3,2,1,4,5,3]]))

#The data modelling process involves select,fit,train data models, and use the model to make predictions.
