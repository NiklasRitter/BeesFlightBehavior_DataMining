import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def starterictransform(neighbors):
    # einstellbare Parameter

    # daten einlesen (nur daten mit einer column verwenden)
    filepath = '../classification/fr_and_ma_without_nan.csv'

    data_to_preprocess = pd.read_csv(filepath, sep=',')

    cols = data_to_preprocess.columns.tolist()

    def clear_nan(fr):
        for i in range(len(fr)):
            if pd.isna(fr[type][i]):
                fr[type][i] = fr[type][i - 1]

        return fr

    def transform(data_prep, type):

        resul = data_prep.copy()

        data_prep = clear_nan(data_prep)

        for i in range(0, len(data_prep)):
            div = 0
            list = []
            if i < neighbors:
                before = i

                for j in range(before):
                    list.append(data_prep[type][j])
                    div += 1

                for k in range(before, neighbors * 2):
                    list.append(data_prep[type][k + 1])
                    div += 1

                list.sort()
                resul[type][i] = resul[type][i] - list[div // 2]

            elif i > len(data_prep) - neighbors:
                after = len(data_prep) - i

                dif = neighbors * 2 - after

                for k in range(dif):
                    list.append(data_prep[type][i - k - 1])
                    div += 1

                for j in range(1, after):
                    list.append(data_prep[type][i + j])
                    div += 1

                list.sort()
                resul[type][i] = resul[type][i] - list[div // 2]
            else:

                for k in range(neighbors * 2):
                    if k < neighbors:
                        list.append(data_prep[type][i - k])
                        div += 1
                    else:
                        if (i + 1 + k - neighbors) < len(data_prep):
                            list.append(data_prep[type][i + 1 + k - neighbors]);
                            div += 1

                list.sort()
                resul[type][i] = resul[type][i] - list[div // 2]

        return resul

    data_prep = data_to_preprocess.copy()
    res = data_to_preprocess.copy()

    for j in cols:
        for i in cols:
            if i != j:
                data_to_preprocess = data_to_preprocess.drop([i], axis=1)

        type = j
        if type != "date" and type != "activity":
            result = transform(data_to_preprocess, type)
            # result.to_csv("./nearest_neighbor_transform" + j + str(neighbors) + ".csv", index=False)
            res[j] = result[j]

        data_to_preprocess = data_prep

    res.to_csv("ma_nearest_neighbor_transform_all_median_" + str(neighbors) + ".csv", index=False)


# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-download-auto-examples-classification-plot-classifier-comparison-py

def pca(filenumber):
    maDf = pd.read_csv('ma_nearest_neighbor_transform_all_median_' + str(filenumber) + '.csv', sep=',')
    maDf = maDf.drop(['date', 'activity'], axis=1)

    # Create features and target datasets
    features = ['temperature', 'tavg', 'tmax', 'tmin', 'prcp', 'tsun', 'humidity', 'weight']
    X = maDf[features].values

    # Standardize the features
    X = StandardScaler().fit_transform(X)

    # Preview X
    pd.DataFrame(data=X, columns=features).head()

    # Instantiate PCA
    pca = PCA()

    # Fit PCA to features
    principalComponents = pca.fit_transform(X)

    # write pca data into file
    pd.DataFrame(principalComponents).to_csv("pca_data.csv", index=False)

    # look at the PC´s
 #   for a in pca.components_:
  #      print(list(map(lambda x: round(x, 3), list(a))))

   # print(pca.explained_variance_ratio_)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
   # plt.show()

    # Calculate the variance explained by principle components
    plt.scatter(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('component')
    plt.ylabel('explained variance ratio');
    #plt.show()


def classification():
    names = ["Nearest Neighbors", "Linear SVM", "Poly SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(5),
        SVC(kernel="linear", C=0.025),
        SVC(kernel='poly', gamma='auto', C=1),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    # get data
    maDf = pd.read_csv("pca_data.csv", sep=',')
    X = maDf

    # get target data
    maDfAll = pd.read_csv("fr_and_ma_without_nan.csv", sep=',')
    y = maDfAll['activity']
    # HIGH LOW -> 1,0
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    iterations = 4

    # initialize dictionary
    res = {}
    for name in names:
        res[name] = []

    # initialize dictionary - cross val
    cross_v_dict = {}
    for name in names:
        cross_v_dict[name] = []

    # make classification
    for i in range(iterations):

        # split data
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2, random_state=30)

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            tmplist = res.get(name)
            tmplist.append(score)
            res[name] = tmplist
            # print(name + " : " + str(round(score, 3)))
            cross_val = cross_val_score(clf, X_train, y_train, cv=4)
            tmplist = cross_v_dict.get(name)
            tmplist.append(cross_val)
            cross_v_dict[name] = tmplist

    # print classifier scores
    for name in names:
        print(name + " : " + str(list(map(lambda x: round(x, 3), res.get(name)))) + "     Max Value: " + str(
            max(list(map(lambda x: round(x, 3), res.get(name))))) + "         Cross Val: " + str(cross_v_dict.get(name)))


if __name__ == '__main__':

    for i in range(3, 10):
        starterictransform(i)
        pca(i)
        print("------- i = " + str(i) + " ---------")
        classification()
        print("\n\n\n")
