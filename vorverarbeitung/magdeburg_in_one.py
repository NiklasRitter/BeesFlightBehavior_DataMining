import pandas as pd

magdeburg_bee = pd.read_csv('../final_classification/fr_2019.csv', sep=',')
magdeburg_weather = pd.read_csv('../data/2019/test_weather_frankfurt.csv', sep=',')

magdeburg_bee['prcp'] = magdeburg_weather['prcp']
magdeburg_bee['tsun'] = magdeburg_weather['tsun']


magdeburg_bee.to_csv('../final_classification/fr_2019.csv', index=False)
