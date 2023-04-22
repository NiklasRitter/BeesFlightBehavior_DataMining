import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# einstellbare Parameter

# daten einlesen (nur daten mit einer column verwenden)
filepath = '../classification/ma.csv'

neighbors = 7

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
        if i < neighbors:
            mean = 0
            before = i


            for j in range(before):
                mean += data_prep[type][j]
                div += 1

            for k in range(before, neighbors * 2):
                mean += data_prep[type][k + 1]
                div += 1

            resul[type][i] = resul[type][i] - (mean/div)

        elif i > len(data_prep) - neighbors:
            mean = 0
            after = len(data_prep) - i

            dif = neighbors * 2 - after

            for k in range(dif):
                mean += data_prep[type][i - k - 1]
                div += 1

            for j in range(1, after):
                mean += data_prep[type][i + j]
                div += 1

            resul[type][i] = resul[type][i] - (mean/div)

        else:
            mean = 0
            for k in range(neighbors * 2):
                if k < neighbors:
                    mean += data_prep[type][i - k]
                    div += 1
                else:
                    if (i + 1 + k - neighbors) < len(data_prep):
                        mean += data_prep[type][i + 1 + k - neighbors]
                        div += 1

            resul[type][i] = resul[type][i] - (mean/div)

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
        result.to_csv("./nearest_neighbor_transform_" + j + ".csv", index=False)
        res[j] = result[j]

    data_to_preprocess = data_prep

res.to_csv("ma_nearest_neighbor_transform_all.csv", index=False)




