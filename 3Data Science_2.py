import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("example.csv")
dataset.drop(columns=["s_pos", "c_pos", "x"], inplace=True) #Filter useless values
#Use the current dataset to genereate value
dataset["tan_x"] = dataset.apply(lambda x: x["sin_x"] / x["cos_x"] if x["cos_x"] != 0 else "undefined", axis=1)
#Create train model using the dataset
X_train, X_test, y_train, y_test = train_test_split(dataset[["sin_x", "cos_x"]], dataset["tan_x"], test_size=0.2)
#Get result of train models
X_train.info()
X_test.info()
y_train.info()
y_test.info()

print(dataset.head())