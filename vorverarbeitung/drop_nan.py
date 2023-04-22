import pandas as pd

filepath = '../data/all_ma.csv'

data_to_preprocess = pd.read_csv(filepath, sep=',').drop(['temperature'], axis=1)
data_to_preprocess = data_to_preprocess.dropna()

data_to_preprocess.to_csv('../data/all_ma_without_nan_and_temp.csv', index=False)