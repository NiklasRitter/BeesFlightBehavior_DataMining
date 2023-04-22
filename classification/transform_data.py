import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# einstellbare Parameter

# daten einlesen (nur daten mit einer column verwenden)
filepath = '../classification/ma.csv'

number_of_sections = 52

data_to_preprocess = pd.read_csv(filepath, sep=',')
copy = pd.read_csv(filepath, sep=',')

section_length = len(data_to_preprocess) // number_of_sections

cols = data_to_preprocess.columns.tolist()


def clear_nan(fr):
    for i in range(len(fr)):
        if pd.isna(fr[type][i]):
            fr[type][i] = fr[type][i - 1]

    return fr


def transform(data_prep, type):
    first_sample = 0

    sections = []
    data_prep = clear_nan(data_prep)

    for i in range(0, number_of_sections):

        if i == 0:
            section = data_prep[0:section_length]

        elif i == number_of_sections - 1 and (len(data_prep) - (section_length * i) > 0):
            section = data_prep[(section_length * i): len(data_prep)]

        else:
            section = data_prep[(section_length * i): section_length * (i + 1)]

        section_mean = section.mean()
        #print(section_mean)

        for j in range(len(section)):
            section[type][first_sample + j] = section[type][first_sample + j] - section_mean

        first_sample += len(section)
        sections.append(section)

    result = pd.concat(sections)
    return result

data_prep = data_to_preprocess.copy()
res = data_to_preprocess.copy()

for j in cols:
    for i in cols:
        if i != j:
            data_to_preprocess = data_to_preprocess.drop([i], axis=1)

    type = j
    if type != "date" and type != "activity":
        result = transform(data_to_preprocess, type)
        result.to_csv("./mean_transform_" + j + ".csv", index=False)
        res[j] = result[j]

    data_to_preprocess = data_prep

res.to_csv("ma_transform_all.csv", index=False)




