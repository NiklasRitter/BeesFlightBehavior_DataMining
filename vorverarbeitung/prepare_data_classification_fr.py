import pandas as pd

a = pd.read_csv("../preproccessed_data/ma_tavg.csv", sep=',')
b = pd.read_csv("../preproccessed_data/ma_tmin.csv", sep=',')
c = pd.read_csv("../preproccessed_data/ma_tmax.csv", sep=',')
d = pd.read_csv("../data/weather_magdeburg.csv", sep=',')

e = pd.read_csv("../data/target_magdeburg.csv", sep=',')
f = pd.read_csv("../preproccessed_data/ma_weight.csv", sep=',')
g = pd.read_csv("../regression/regression_9.csv", sep=',')
h = pd.read_csv("../data/fr", sep=',')

g['tavg'] = a['tavg']
g['tmax'] = c['tmax']
g['tmin'] = b['tmin']
g['prcp'] = d['prcp']
g['tsun'] = d['tsun']
g['humidity'] = h['humidity']
g['weight'] = f['weight']
g['activity'] = e['activity']

g.to_csv('../classification/ma.csv', index=False)

########
#dropna

drop = pd.read_csv("../classification/ma.csv", sep=',')
drop = drop.dropna()
drop.to_csv('../classification/ma.csv', index=False)