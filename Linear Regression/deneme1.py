import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
# import pickle
from matplotlib import style

data = pd.read_csv ("kilo_boy.csv", sep=",")

data = data [["Boy", "Kilo"]]

predict = "Kilo"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.03)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)
print(acc)

print(linear.coef_)
print(linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print('tahmin edilen kilo:', predictions[x], 'boy:', x_test[x], 'kilo:', y_test[x])


p = 'Boy'
style.use("ggplot")
pyplot.scatter(data[p], data["Kilo"])
pyplot.xlabel(p)
pyplot.ylabel("Kilo")
pyplot.show()
