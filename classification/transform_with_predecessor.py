import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# einstellbare Parameter

# daten einlesen (nur daten mit einer column verwenden)
filepath = '../classification/ma.csv'

neighbors = 2

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
        mean = 0
        div = 0

        if i < neighbors:
            if i > 0:
                before = i

                for j in range(before):
                    mean += data_prep[type][j]
                    div += 1

                resul[type][i] = resul[type][i] - (mean / div)

        else:
            for j in range(1, neighbors):
                mean += data_prep[type][i - j]
                div += 1

            resul[type][i] = resul[type][i] - (mean / div)

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

res.to_csv("ma_predecessor_transform_all.csv", index=False)
