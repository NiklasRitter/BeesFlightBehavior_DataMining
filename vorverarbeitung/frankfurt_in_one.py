import pandas as pd

frankfurt_humidity = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_humidity.csv', sep=',')
frankfurt_temperature = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_temperature.csv', sep=',')
frankfurt_weight = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_weight.csv', sep=',')

frankfurt_tavg = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_tavg.csv', sep=',')
frankfurt_tmin = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_tmin.csv', sep=',')
frankfurt_tmax = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_tmax.csv', sep=',')
frankfurt_prcp = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_prcp.csv', sep=',')
frankfurt_tsun = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_tsun.csv', sep=',')

frankfurt_humidity['temperature'] = frankfurt_temperature['temperature']
frankfurt_humidity['weight'] = frankfurt_weight['weight']

frankfurt_humidity['tavg'] = frankfurt_tavg['tavg']
frankfurt_humidity['tmin'] = frankfurt_tmin['tmin']
frankfurt_humidity['tmax'] = frankfurt_tmax['tmax']
frankfurt_humidity['prcp'] = frankfurt_prcp['prcp']
frankfurt_humidity['tsun'] = frankfurt_tsun['tsun']

frankfurt_humidity.to_csv('../data_bearbeitet/frankfurt/data_old.csv', index=False)
