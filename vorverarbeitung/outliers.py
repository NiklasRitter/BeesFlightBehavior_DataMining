
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


# einstellbare Parameter

# daten einlesen (nur daten mit einer column verwenden)
filepath = '../data/2019/test_weather_frankfurt.csv'

neighbors = 20
number_of_sections = 14
type = 'tmin'

# daten in file schreiben
write_to_file = True
add_to_exisiting_file = True
pred_filepath = '../final_classification/fr_2019.csv'

data_to_preprocess = pd.read_csv(filepath, sep=',')
copy = pd.read_csv(filepath, sep=',')

cols = data_to_preprocess.columns.tolist()
for i in cols:
    if i != type:
        data_to_preprocess = data_to_preprocess.drop([i], axis=1)

section_length = len(data_to_preprocess) // number_of_sections


def identify_outliers(section, neighbors):
    clf = LocalOutlierFactor(n_neighbors=neighbors)
    outliers = clf.fit_predict(section)
    return outliers


def clear_outliers(section, outliers, number):
    for i in range(len(section)):
        if outliers[i] == -1:
            if i < len(section) - 2 and i > 1:
                section[type][number + i] = (section[type][number + i - 1] + section[type][number + i + 1] + section[type][number + i - 2] + section[type][number + i + 2]) / 4
            elif i <= 1:
                section[type][number + i] = (section[type][number + i + 1] + section[type][number + i + 2]) / 2
            else:
                section[type][number + i] = (section[type][number + i - 1] + section[type][number + i - 2]) / 2

    return section



def clear_nan(fr):
    for i in range(len(fr)):
        if pd.isna(fr[type][i]):
            fr[type][i] = fr[type][i - 1]

    return fr

first_sample = 0


sections = []
data_to_preprocess = clear_nan(data_to_preprocess)

for i in range(0, number_of_sections):
    n_neighbors = 0

    if i == 0:
        section = data_to_preprocess[0:section_length]
        n_neighbors = neighbors
    elif i == number_of_sections - 1 and (len(data_to_preprocess) - (section_length * i) > 0):
        section = data_to_preprocess[(section_length * i): len(data_to_preprocess)]
        n_neighbors = len(section)
    else:
        section = data_to_preprocess[(section_length * i): section_length * (i + 1)]
        n_neighbors = neighbors

    while True:

        outliers = identify_outliers(section, n_neighbors)

        if -1 not in outliers:
            break

        cleared_section = clear_outliers(section, outliers, first_sample)

        section = cleared_section[:]

    first_sample += len(section)
    sections.append(section)


result = pd.concat(sections)

list = []
for i in range(len(result)):
    list.append(i)

if write_to_file:
    if add_to_exisiting_file:
        ex_file = pd.read_csv(pred_filepath, sep=',')
        ex_file[type] = result[type]
        ex_file.to_csv(pred_filepath, index=False)
    else:
        result.to_csv(pred_filepath, index=False)

plt.figure(figsize=(30,7), dpi=300)
plt.plot(list, result[type])
plt.plot(list, copy[type])
plt.title("blue = new / orange = old")
plt.show()
