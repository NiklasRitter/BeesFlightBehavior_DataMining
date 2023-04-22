import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
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
                            list.append(data_prep[type][i + 1 + k - neighbors])
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
    plt.ylabel('cumulative explained variance')
    # plt.show()

    # Calculate the variance explained by principle components
    plt.scatter(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('component')
    plt.ylabel('explained variance ratio')
    # plt.show()

def withoutpca(filenumber):
    maDf = pd.read_csv('ma_nearest_neighbor_transform_all_median_' + str(filenumber) + '.csv', sep=',')
    maDf = maDf.drop(['date', 'activity'], axis=1)

    # Create features and target datasets
    features = ['temperature', 'tavg', 'tmax', 'tmin', 'prcp', 'tsun', 'humidity', 'weight']
    X = maDf[features].values

    # Standardize the features
    X = StandardScaler().fit_transform(X)

    # write pca data into file
    pd.DataFrame(X).to_csv("pca_data.csv", index=False)


######################################################################################################
######################################################################################################
######################################################################################################

def gridNearestNeighbors(X, y):
    clf = KNeighborsClassifier()

    parameters = {
        "n_neighbors": [2, 5, 10, 15, 20, 30, 50, 60, 100, 300, 350, 400, 450, 500, 550, 600, 650, 700],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p": [1, 2],
        "n_jobs": [-1],
    }
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,
    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print("Nearest Neighbors")

    #for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
    #                             grid.cv_results_["std_test_score"]):
        #if mean > 0.72:
          #  print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)


######################################################################################################

def gridLinearSVM(X, y):
    clf = SVC()

    parameters = {
        "kernel": ["linear"],
        "C": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]
    }

    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print("Linear SVM")

    #for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
           #                      grid.cv_results_["std_test_score"]):
        #if mean > 0.72:
        #    print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)

######################################################################################################

def gridPolySVM(X, y):
    clf = SVC()

    parameters = {
        "gamma": ["auto", "scale"],
        "kernel": ["poly"],
        "C": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2],
        "degree": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print("Poly SVM")

    #for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
            #                     grid.cv_results_["std_test_score"]):
        #if mean > 0.72:
            #print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)

######################################################################################################

def gridRBFSVM(X, y):
    clf = SVC(gamma=2, C=1)

    parameters = {
        "gamma": ["auto", "scale"],
        "kernel": ["rbf"],
        "C": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 5, 10],
    }

    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print("RBF SVM")

    #for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
     #                            grid.cv_results_["std_test_score"]):
      #  if mean > 0.72:
       #     print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)

######################################################################################################

def gridGaussianProcess(X, y):
    clf = GaussianProcessClassifier(1.0 * RBF(1.0))

    parameters = {
        "max_iter_predict": [500,1000],
        "warm_start":[True, False],
    }

    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True),
        refit=True,
        n_jobs=-1
    )

    grid.fit(X, y)


    print("Gaussian Process")

    #for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
     #                            grid.cv_results_["std_test_score"]):
     #   print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)

######################################################################################################

def gridDecisionTree(X, y):
    clf = DecisionTreeClassifier()

    parameters = {
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
    )

    grid.fit(X, y)

    print("Decision Tree")

   # for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
      #                           grid.cv_results_["std_test_score"]):
     #   print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)

######################################################################################################

def gridRandomForest(X, y):
    clf = RandomForestClassifier()

    parameters = {
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "n_estimators": [5, 10, 20, 30, 40, 50],
    }

    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
    )

    grid.fit(X, y)

    print("Random Forest")

   # for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
    #                             grid.cv_results_["std_test_score"]):
     #   print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)

######################################################################################################

def gridNeuralNet(X, y):
    clf = MLPClassifier()

    parameters = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
        "max_iter": [1000000],
        "solver": ["lbfgs", "sgd", "adam"],
        "warm_start": [True, False],
    }

    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print("Neural Net")

   # for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
      #                           grid.cv_results_["std_test_score"]):
      #  if mean > 0.72:
      #      print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)

######################################################################################################

def gridAdaBoost(X, y):
    clf = AdaBoostClassifier()

    parameters = {
        "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 4, 5, 10]
    }

    grid = GridSearchCV(
        clf,
        parameters,
        cv=KFold(n_splits=5, shuffle=True),
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print("AdaBoost")

   # for params, mean, std in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"],
       #                          grid.cv_results_["std_test_score"]):
       # if mean > 0.71:
       #     print(params, mean, std)
    print(grid.best_params_)
    print(grid.best_score_)

######################################################################################################

def gridNaiveBayes(X, y):
    clf = GaussianNB()

    cross_val = cross_val_score(clf,
                                X,
                                y,
                                cv=KFold(n_splits=5, shuffle=True))

    print("Naive Bayes")
    print(cross_val)


######################################################################################################

def gridQDA(X, y):
    clf = QuadraticDiscriminantAnalysis()

    cross_val = cross_val_score(clf,
                                X,
                                y,
                                cv=KFold(n_splits=5, shuffle=True))

    print("QDA")
    print(cross_val)


######################################################################################################


def classification():
    # get data
    maDf = pd.read_csv("pca_data.csv", sep=',')
    X = maDf

    # get target data
    maDfAll = pd.read_csv("fr_and_ma_without_nan.csv", sep=',')
    y = maDfAll['activity']
    # HIGH LOW -> 1,0
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    gridNearestNeighbors(X, y)
    gridLinearSVM(X, y)
    gridPolySVM(X, y)
    gridRBFSVM(X, y)
    gridGaussianProcess(X, y)
    gridDecisionTree(X, y)
    gridRandomForest(X, y)
    gridNeuralNet(X, y)
    gridAdaBoost(X, y)
    gridNaiveBayes(X, y)
    gridQDA(X, y)


if __name__ == '__main__':

    for i in range(4, 8):
        starterictransform(i)
        withoutpca(i)
        # pca(i)
        print("------- i = " + str(i) + " ---------")
        classification()
        print("\n\n\n")
