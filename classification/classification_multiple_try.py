import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


data_magdeburg = pd.read_csv('ma_nearest_neighbor_transform_all.csv', sep=',')

data_magdeburg = data_magdeburg.drop(['date'], axis=1)

data_temp = data_magdeburg['temperature']
data_tavg = data_magdeburg['tavg']
data_tmax = data_magdeburg['tmax']
data_tmin = data_magdeburg['tmin']
data_prcp = data_magdeburg['prcp']
data_tsun = data_magdeburg['tsun']
data_humidity = data_magdeburg['humidity']
data_weight = data_magdeburg['weight']

data_all_temp = data_tavg.copy()
data_all_temp['tmax'] = data_tmax.copy()
data_all_temp['temperature'] = data_temp.copy()
data_all_temp['tmin'] = data_tmin.copy()

traindata = [
    data_temp,
    data_tavg,
    data_tmax,
    data_tmin,
    data_prcp,
    data_tsun,
    data_humidity,
    data_weight,
    data_all_temp
]

traindata_names = [
    "temp", "tavg", "tmax", "tmin",
    "prcp", "tsun", "humidity", "weight",
    "all_temp"
]

X = data_magdeburg.drop(['activity'], axis=1).copy()
Y = data_magdeburg['activity'].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"
         ]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    print(name, clf.score(X_test, Y_test))


# print(clf.feature_importances_)
# prediction = clf.predict(X_test)
# print(clf.predict_proba(X_test))

