import numpy as np
import pandas as pd
import sklearn.neighbors
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot

# Import Data
bee_f = pd.read_csv('data/RAW_bee_frankfurt.csv', sep=",")
weather_f = pd.read_csv('data/RAW_weather_frankfurt.csv', sep=",")


bee_m = pd.read_csv('data/magde_all.csv', sep=",")
weather_m = pd.read_csv('data/RAW_weather_magdeburg.csv', sep=",")



#---------------------------------------------------------------------------------------------------------------
# Frankfurt

bee_f['tavg'] = weather_f['tavg']
bee_f['tmin'] = weather_f['tmin']
bee_f['tmax'] = weather_f['tmax']
bee_f['prcp'] = weather_f['prcp']
bee_f['tsun'] = weather_f['tsun']

df_bee_f_cleaned = bee_f.dropna()


list_temperatureF = df_bee_f_cleaned['temperature'].array

list_humidityF = df_bee_f_cleaned['humidity'].array
list_weightF = df_bee_f_cleaned['weight'].array
list_tavgF = df_bee_f_cleaned['tavg'].array
list_tminF = df_bee_f_cleaned['tmin'].array
list_tmaxF = df_bee_f_cleaned['tmax'].array
list_prcpF = df_bee_f_cleaned['prcp'].array
list_tsunF = df_bee_f_cleaned['tsun'].array


corr_temp_tavgF = np.corrcoef(list_temperatureF, list_tavgF)[0][1]
corr_temp_tminF = np.corrcoef(list_temperatureF, list_tminF)[0][1]
corr_temp_tmaxF = np.corrcoef(list_temperatureF, list_tmaxF)[0][1]
corr_temp_prcpF = np.corrcoef(list_temperatureF, list_prcpF)[0][1]
corr_temp_tsunF = np.corrcoef(list_temperatureF, list_tsunF)[0][1]

plt.scatter(list_temperatureF, list_tavgF)
plt.title("Frankfurt temp <-> tavg | Correl: " + str(corr_temp_tavgF))
plt.show()

plt.scatter(list_temperatureF, list_tminF)
plt.title("Frankfurt temp <-> tmin | Correl: " + str(corr_temp_tminF))
plt.show()

plt.scatter(list_temperatureF, list_tmaxF)
plt.title("Frankfurt temp <-> tmax | Correl: " + str(corr_temp_tmaxF))
plt.show()

plt.scatter(list_temperatureF, list_prcpF)
plt.title("Frankfurt temp <-> prcp | Correl: " + str(corr_temp_prcpF))
plt.show()

plt.scatter(list_temperatureF, list_tsunF)
plt.title("Frankfurt temp <-> tsun | Correl: " + str(corr_temp_tsunF))
plt.show()

#---------------------------------------------------------------------------------------------------------------
# Magdeburg

bee_m['tavg'] = weather_m['tavg']
bee_m['tmin'] = weather_m['tmin']
bee_m['tmax'] = weather_m['tmax']
bee_m['prcp'] = weather_m['prcp']
bee_m['tsun'] = weather_m['tsun']

df_bee_m_cleaned = bee_m.dropna()


list_temperatureM = df_bee_m_cleaned['temperature'].array

list_humidityM = df_bee_m_cleaned['humidity'].array
list_weightM = df_bee_m_cleaned['weight'].array
list_tavgM = df_bee_m_cleaned['tavg'].array
list_tminM = df_bee_m_cleaned['tmin'].array
list_tmaxM = df_bee_m_cleaned['tmax'].array
list_prcpM = df_bee_m_cleaned['prcp'].array
list_tsunM = df_bee_m_cleaned['tsun'].array


corr_temp_tavgM = np.corrcoef(list_temperatureM, list_tavgM)[0][1]
corr_temp_tminM = np.corrcoef(list_temperatureM, list_tminM)[0][1]
corr_temp_tmaxM = np.corrcoef(list_temperatureM, list_tmaxM)[0][1]
corr_temp_prcpM = np.corrcoef(list_temperatureM, list_prcpM)[0][1]
corr_temp_tsunM = np.corrcoef(list_temperatureM, list_tsunM)[0][1]

plt.scatter(list_temperatureM, list_tavgM)
plt.title("Magdeburg temp <-> tavg | Correl: " + str(corr_temp_tavgM))
plt.show()

plt.scatter(list_temperatureM, list_tminM)
plt.title("Magdeburg temp <-> tmin | Correl: " + str(corr_temp_tminM))
plt.show()

plt.scatter(list_temperatureM, list_tmaxM)
plt.title("Magdeburg temp <-> tmax | Correl: " + str(corr_temp_tmaxM))
plt.show()

plt.scatter(list_temperatureM, list_prcpM)
plt.title("Magdeburg temp <-> prcp | Correl: " + str(corr_temp_prcpM))
plt.show()

plt.scatter(list_temperatureM, list_tsunM)
plt.title("Magdeburg temp <-> tsun | Correl: " + str(corr_temp_tsunM))
plt.show()




