# compare ensemble to each baseline classifier
import itertools

import pandas as pd
from matplotlib import pyplot
from numpy import mean
from numpy import std
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# einstellbare Parameter

filepath = 'fr.csv'

magdeburg = False
frankfurt = True

showAll = False
featureWeight = False
classi = False
bruteforce = True

with_pca = False
N_COMPONENTS = 4

#to_delete_features_fra = ['prcp', 'dif_temperature', 'dif_tavg', 'dif_tmax']
#to_delete_features_magde = ['tmax', 'dif_weight', 'dif_tavg', 'dif_tsun']

#to_delete_features.append('temperature')
#to_delete_features.append('tavg')
#to_delete_features.append('tmax')
#to_delete_features.append('tmin')
#to_delete_features.append('prcp')
#to_delete_features.append('tsun')
#to_delete_features.append('humidity')
#to_delete_features.append('weight')
#to_delete_features.append('dif_temperature')
#to_delete_features.append('dif_weight')
#to_delete_features.append('dif_tavg')
#to_delete_features.append('dif_tmin')
#to_delete_features.append('dif_tmax')
#to_delete_features.append('dif_prcp')
#to_delete_features.append('dif_tsun')
#to_delete_features.append('dif_humidity')


def starterictransform(neighbors, filepath):

    data_to_preprocess = pd.read_csv(filepath + '.csv', sep=',')

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

    res.to_csv(filepath + "/savings/" + "median_" + str(neighbors) + ".csv", index=False)


def transform_data(neighbours, filepath):

    n = neighbours

    fr = pd.read_csv(filepath, sep=',')

    filepath_without_csv = filepath[0:len(filepath) - 4]

    starterictransform(n, filepath_without_csv)

    fr_transformed = pd.read_csv(filepath_without_csv + "/savings/" + "median_" + str(n) + ".csv", sep=',')

    fr['dif_temperature'] = fr_transformed['temperature']
    fr['dif_weight'] = fr_transformed['weight']
    fr['dif_tavg'] = fr_transformed['tavg']
    fr['dif_tmin'] = fr_transformed['tmin']
    fr['dif_tmax'] = fr_transformed['tmax']
    fr['dif_prcp'] = fr_transformed['prcp']
    fr['dif_tsun'] = fr_transformed['tsun']
    fr['dif_humidity'] = fr_transformed['humidity']

    fr.to_csv(filepath_without_csv + '_' + str(neighbours) + '_and_dif.csv', index=False)


def pca(neighbours):
    if magdeburg:
        read_df = pd.read_csv('ma_' + str(neighbours) + '_and_dif.csv', sep=',')

    if frankfurt:
        read_df = pd.read_csv('fr_' + str(neighbours) + '_and_dif.csv', sep=',')

    read_df = read_df.drop(['date', 'activity'], axis=1)

    features = read_df.columns.tolist()
    to_standardize = read_df[features].values

    # Standardize the features
    standardized = StandardScaler().fit_transform(to_standardize)

    #############################################################
    # Instantiate PCA
    pca_instance = PCA(n_components=N_COMPONENTS)
    #############################################################

    # Fit PCA to features
    principal_components_data = pca_instance.fit_transform(standardized)

    to_save_df = pd.DataFrame(principal_components_data)

    to_save_df.columns = features

    # write pca data into file
    to_save_df.to_csv("tmp_data_for_classification.csv", index=False)

    if showAll:
        for a in pca_instance.components_:
            print(list(map(lambda x1: round(x1, 3), list(a))))
        print(pca_instance.explained_variance_ratio_)


def without_pca(neighbours):
    if magdeburg:
        read_df = pd.read_csv('ma_' + str(neighbours) + '_and_dif.csv', sep=',')

    if frankfurt:
        read_df = pd.read_csv('fr_' + str(neighbours) + '_and_dif.csv', sep=',')

    read_df = read_df.drop(['date', 'activity'], axis=1)

    features = read_df.columns.tolist()
    to_standardize = read_df[features].values

    # Standardize the features
    standardized = StandardScaler().fit_transform(to_standardize)

    to_save_df = pd.DataFrame(standardized)

    to_save_df.columns = features

    # write pca data into file
    to_save_df.to_csv("tmp_data_for_classification.csv", index=False)


# get the dataset
def get_dataset(neighbours, filepath):

    transform_data(neighbours, filepath)

    if with_pca:
        pca(neighbours)
    else:
        without_pca(neighbours)

    X = pd.read_csv("tmp_data_for_classification.csv", sep=',')

    # get target data
    if magdeburg:
        read_df = pd.read_csv('ma_' + str(neighbours) + '_and_dif.csv', sep=',')

    if frankfurt:
        read_df = pd.read_csv('fr_' + str(neighbours) + '_and_dif.csv', sep=',')

    y_tmp = read_df['activity']
    # HIGH LOW -> 1,0
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y_tmp)

    return X, y


# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression(C=0.01, max_iter=100, solver='newton-cg')))
    level0.append(('knn', KNeighborsClassifier(algorithm='auto', n_jobs=-1, n_neighbors=350, p=2, weights='uniform')))
    level0.append(('cart', DecisionTreeClassifier(max_depth=2)))
    level0.append(('svm rbf', SVC(gamma='scale', C=0.2)))
    level0.append(('svm linear', SVC(kernel="linear", C=0.01)))
    level0.append(('svm poly', SVC(kernel='poly', degree=1, gamma='scale', C=0.05)))
    level0.append(('gaussian process', GaussianProcessClassifier(max_iter_predict=500)))
    level0.append(('adaboost', AdaBoostClassifier(learning_rate=0.2, n_estimators=60)))
    level0.append(('bayes', GaussianNB()))
    level0.append(('qda', QuadraticDiscriminantAnalysis()))
    #level0.append(('neuralnet', MLPClassifier(alpha=1, learning_rate_init=0.001, max_iter=10000)))
    #level0.append(('randomforest', RandomForestClassifier(max_depth=3, n_estimators=40)))

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
    #models['stacking'] = get_stacking()
    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


def combinations123(objects, k):
    print("k")
    object = list(objects)
    print(object)
    if objects == [] or len(object) < k or k == 0:
        return []
    elif len(object) == k:
        return object
    else:
        for combination in combinations123(object[1:], k-1):
            var = [object[0]] + combination
            print(var)
        for combination in combinations123(object[1:], k):
            yield combination


if __name__ == '__main__':
    print("ahllo")
    combinations123(
        {"temperature", "tavg", "tmax", "tmin", "prcp", "tsun", "humidity", "weight", "dif_temperature", "dif_weight",
         "dif_tavg", "dif_tmin", "dif_tmax", "dif_prcp", "dif_tsun", "dif_humidity"}, 4)
    print("hllo")



    if featureWeight:
        for neighbours in [3, 4, 5, 6, 7]:
            filepath = 'ma.csv'
            # define dataset
            X, y = get_dataset(neighbours, filepath)

            model = Ridge()
            # fit the model
            model.fit(X, y)
            # get importance
            importance = model.coef_
            # summarize feature importance
            for i, v in enumerate(importance):
                print('Feature: %0d, Score: %.5f' % (i, v))
            features = X.columns.tolist()
            pyplot.figure(figsize=(25, 7), dpi=300)
            # plot feature importance
            pyplot.bar(features, importance)
            pyplot.show()

    if classi:
        for neighbours in [5]:

            print("-------------------------" + str(neighbours) + "------------------------------")

            # get dataset
            X, y = get_dataset(neighbours, filepath)

            to_delete_features_fra = ['prcp', 'dif_temperature', 'dif_tavg', 'dif_tmax']
            #['tsun', 'humidity', 'weight', 'dif_tmin', 'dif_tsun', 'dif_prcp', 'tmax', 'tavg', 'temperature']tmint difweight dif hum

            to_delete_features_magde = ['tmax', 'dif_weight', 'dif_tavg', 'dif_tsun']
            #to_delete_features
            X = X.drop(to_delete_features_fra, axis=1)

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
            # pyplot.show()

    if bruteforce:
        neighbours = 4
        b = 0
        X, y = get_dataset(neighbours, filepath)
        features = X.columns.tolist()
        for x in ['temperature', 'tavg', 'weight', 'tmin', 'tmax', 'dif_humidity', 'humidity', 'dif_tsun']:
            features.remove(x)

        for L in range(5, 8):
            conclude = [[], []]
            for subset in itertools.combinations(features, L):
                if subset == ():
                    continue

                # get dataset
                #X, y = get_dataset(neighbours, filepath)
                X1 = X[list(subset)]

                # get the models to evaluate
                models = get_models()

                # evaluate the models and store results
                results, names = list(), list()
                tmp_list = []
                conclude[0].append(X1.columns.tolist())
                for name, model in models.items():
                    scores = evaluate_model(model, X1, y)
                    results.append(scores)
                    names.append(name)
                    tmp_list.append(mean(scores))
                    #print(X.columns.tolist())
                    #print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

                conclude[1].append(max(tmp_list))


            huh = pd.DataFrame(columns=['features', 'values'])
            huh['features'] = conclude[0]
            huh['values'] = conclude[1]
            huh.to_csv("fr333_bruteforcedata_" + str(L) + "_.csv", index=False)

#temperature,tavg,tmax,tmin,prcp,tsun,humidity,weight,dif_temperature,dif_weight,dif_tavg,dif_tmin,dif_tmax,dif_prcp,dif_tsun,dif_humidity

#8
#"['prcp', 'tsun', 'dif_temperature', 'dif_weight', 'dif_tavg', 'dif_tmin', 'dif_prcp', 'dif_tsun']"
#"['prcp', 'tsun', 'humidity', 'dif_weight', 'dif_tavg', 'dif_tmin', 'dif_tmax', 'dif_prcp']"


#11
#"['prcp', 'tsun', 'humidity', 'dif_temperature', 'dif_weight', 'dif_tavg', 'dif_tmin', 'dif_tmax', 'dif_prcp', 'dif_tsun', 'dif_humidity']"
#"['tmin', 'prcp', 'tsun', 'humidity', 'dif_temperature', 'dif_weight', 'dif_tavg', 'dif_tmin', 'dif_tmax', 'dif_prcp', 'dif_tsun']"

#14
#"['tavg', 'tmax', 'tmin', 'prcp', 'tsun', 'humidity', 'weight', 'dif_temperature', 'dif_weight', 'dif_tmin', 'dif_tmax', 'dif_prcp', 'dif_tsun', 'dif_humidity']"
#"['temperature', 'tmax', 'tmin', 'prcp', 'tsun', 'humidity', 'dif_temperature', 'dif_weight', 'dif_tavg', 'dif_tmin', 'dif_tmax', 'dif_prcp', 'dif_tsun', 'dif_humidity']"


