from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data_magdeburg = pd.read_csv('ma.csv', sep=',')

data_magdeburg = data_magdeburg.drop(['date'], axis=1)

X = data_magdeburg.drop(['activity'], axis=1).copy()
Y = data_magdeburg['activity'].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
