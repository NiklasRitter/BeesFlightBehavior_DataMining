from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

data_magdeburg = pd.read_csv('ma.csv', sep=',')

data_magdeburg = data_magdeburg.drop(['date'], axis=1)


data_tavg = data_magdeburg['tavg']
data_tmax = data_magdeburg['tmax']
data_temp = data_magdeburg['temperature']

data_pca = data_tavg.copy()
data_pca['tmax'] = data_tmax.copy()
data_pca['temperature'] = data_temp.copy()

#X = data_pca.copy()
X = data_magdeburg.drop(['activity'], axis=1).copy()
Y = data_magdeburg['activity'].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

tree.plot_tree(clf)

print(clf.feature_importances_)
prediction = clf.predict(X_test)
print(clf.score(X_test, Y_test))


