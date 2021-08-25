import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#ploting the target by patel width and patal length
#using the pandas(?)

from sklearn.datasets import load_iris
iris = load_iris()

from matplotlib import pyplot as plt

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()
print("-"*60)

#------------------------------------------------------------------------------

#reading the csv file
iris =pd.read_csv('iris.csv')

#------------------------------------------------------------------------------

print(iris.describe())
print("-"*60)
print(iris.info())
print("-"*60)

#no null entry 
#------------------------------------------------------------------------------
x = iris.drop('variety',axis=1)
y = iris['variety']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

'''from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
print(scaler.fit(data))'''






