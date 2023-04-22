import pandas as pd
from matplotlib import pyplot as plt

csv_input_bee_f = pd.read_csv('../final_classification/ma_2019.csv', sep=',')
#csv_input_bee_f = pd.read_csv('../data/bee_magdeburg_date.csv', sep=',').drop(['humidity', 'temperature'], axis=1)


csv_input_bee_f['weight'][0] = csv_input_bee_f['weight'][1]

for i in range(11):
    csv_input_bee_f['weight'][i] = csv_input_bee_f['weight'][i] + 54.50213583333335


csv_input_bee_f.to_csv('../final_classification/ma_2019.csv', index=False)