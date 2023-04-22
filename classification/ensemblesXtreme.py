# compare ensemble to each baseline classifier
import pandas as pd
from matplotlib import pyplot
from numpy import mean
from numpy import std
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge, Lasso, LinearRegression, ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# einstellbare Parameter

# daten einlesen (nur daten mit einer column verwenden)



def starterictransform(neighbors, filepath):
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
    #############################################################
    # Instantiate PCA
    pca = PCA(n_components=4)
    #############################################################
    # Fit PCA to features
    principalComponents = pca.fit_transform(X)

    # write pca data into file
    pd.DataFrame(principalComponents).to_csv("pca_data.csv", index=False)

    # look at the PC´s


#   for a in pca.components_:
#      print(list(map(lambda x: round(x, 3), list(a))))

# print(pca.explained_variance_ratio_)


def withoutpca(filenumber):
    maDf = pd.read_csv('ma_nearest_neighbor_transform_all_median_' + str(filenumber) + '.csv', sep=',')
    maDf = maDf.drop(['date', 'activity'], axis=1)

    # Create features and target datasets
    features = ['temperature', 'tavg', 'tmax', 'tmin', 'prcp', 'tsun', 'humidity', 'weight']
    X = maDf[features].values

    # Standardize the features
    X_neu = StandardScaler().fit_transform(X)

    # write pca data into file
    pd.DataFrame(X_neu).to_csv("pca_data.csv", index=False)



# get the dataset
def get_dataset(i, filepath):

    starterictransform(i, filepath)
    #pca(i)
    withoutpca(i)
    maDf = pd.read_csv("pca_data.csv", sep=',')
    X = maDf

    # get target data
    maDfAll = pd.read_csv(filepath, sep=',')
    y = maDfAll['activity']
    # HIGH LOW -> 1,0
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    return X, y


# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression(C=0.01, max_iter=100, solver='newton-cg')))
    #level0.append(('knn', KNeighborsClassifier(algorithm='auto', n_jobs=-1, n_neighbors=350, p=2, weights='uniform')))
    #level0.append(('cart', DecisionTreeClassifier(max_depth=2)))
    #level0.append(('svm rbf', SVC(gamma='scale', C=0.2)))
    level0.append(('svm linear', SVC(kernel="linear", C=0.01)))
    level0.append(('svm poly', SVC(kernel='poly', degree=1, gamma='scale', C=0.05)))
    #level0.append(('gaussian process', GaussianProcessClassifier(max_iter_predict=500)))
    level0.append(('adaboost', AdaBoostClassifier(learning_rate=0.2, n_estimators=60)))
    #level0.append(('bayes', GaussianNB()))
    #level0.append(('qda', QuadraticDiscriminantAnalysis()))
    level0.append(('neuralnet', MLPClassifier(alpha=1, learning_rate_init=0.001, max_iter=10000)))
    level0.append(('randomforest', RandomForestClassifier(max_depth=3, n_estimators=40)))

    # define meta learner model
    level1 = LogisticRegression(max_iter=100000000)
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


# get a list of models to evaluate
def get_models():
    models = dict()
    models['lr'] = LogisticRegression(C=0.01, max_iter=100, solver='newton-cg')
    models['knn'] = KNeighborsClassifier(algorithm='auto', n_jobs=-1, n_neighbors=350, p=2, weights='uniform')
    models['cart'] = DecisionTreeClassifier(max_depth=2)
    models['svm rbf'] = SVC(gamma='scale', C=0.2)
    models['svm linear'] = SVC(kernel="linear", C=0.01)
    models['svm poly'] = SVC(kernel='poly', degree=1, gamma='scale', C=0.05)
    models['gaussian process'] = GaussianProcessClassifier(max_iter_predict=500)
    models['adaboost'] = AdaBoostClassifier(learning_rate=0.2, n_estimators=60)
    models['bayes'] = GaussianNB()
    models['qda'] = QuadraticDiscriminantAnalysis()
    models['neuralnet'] = MLPClassifier(alpha=1, learning_rate_init=0.001, max_iter=10000)
    models['randomforest'] = RandomForestClassifier(max_depth=8, n_estimators=50)
    models['stacking'] = get_stacking()
    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

if __name__ == '__main__':

    # filepath = '../classification/ma.csv'
    # # define dataset
    # X, y = get_dataset(5, filepath)
    #
    # model = LinearRegression()
    # # fit the model
    # model.fit(X, y)
    # # get importance
    # importance = model.coef_
    # # summarize feature importance
    # for i, v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    # # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()






    for x in [3, 4, 5, 6, 7]:
        print("-------------------------" + str(x) + "------------------------------")
        filepath = '../classification/fr_and_ma_without_nan.csv'
        # define dataset
        X, y = get_dataset(x, filepath)
        # get the models to evaluate
        models = get_models()
        # evaluate the models and store results
        results, names = list(), list()
        for name, model in models.items():
            scores = evaluate_model(model, X, y)
            results.append(scores)
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        # plot model performance for comparison
        pyplot.figure(figsize=(25, 7), dpi=300)
        pyplot.boxplot(results, labels=names, showmeans=True)
    #    # pyplot.show()


