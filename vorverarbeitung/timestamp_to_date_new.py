import pandas as pd

data_m_timestamps = pd.read_csv('../data/2019/test_bee_magdeburg.csv', sep=',')
data_m_date = pd.read_csv('../data/dummy.csv', sep=',')
#data_m_date = data_m_date.drop(['date'], axis=1)

for i in range(139):
    newValue= 0
    for j in range(24):
        newValue += data_m_timestamps['humidity'][(i * 24) + j]
    newValue = newValue / 24
    data_m_date['humidity'][i] = newValue

value = 0
for i in range(14):
    value += data_m_timestamps['humidity'][i + 3336]
value = value / 14
print(value)
data_m_date['humidity'][139] = value



for i in range(139):
    newValue = 0
    for j in range(24):
        newValue += data_m_timestamps['temperature'][(i * 24) + j]
    newValue = newValue / 24
    data_m_date['temperature'][i] = newValue


value = 0
for i in range(14):
    value += data_m_timestamps['temperature'][i + 3336]
value = value / 14
data_m_date['temperature'][139] = value


for i in range(139):
    newValue = 0
    for j in range(24):
        newValue += data_m_timestamps['weight'][(i * 24) + j]
    newValue = newValue / 24
    data_m_date['weight'][i] = newValue


value = 0
for i in range(14):
    value += data_m_timestamps['weight'][i + 3336]
value = value / 14

data_m_date['weight'][139] = value

data_m_date.to_csv('../data/bee_magdeburg_date_ne.csv', index=False)

print(data_m_date)

tach= 74.63416666666667+77.45083333333334+80.27583333333334+81.23333333333336+82.0933333333333+82.81666666666662+84.60166666666667+89.36666666666666+85.9+84.64283333333333+78.22966666666667+74.33683333333333+72.22533333333331+67.60449999999999+61.96383333333333+59.56083333333331+59.29416666666667+58.97066666666666+59.98250000000001+58.99583333333334+63.70250000000002+66.0595+70.223+69.93933333333335+72.34433333333331

test = 72.34433333333331 + 74.86716666666669+76.61883333333338+78.53683333333333+79.21133333333334+80.40483333333331+83.54166666666669+85.92833333333333+85.625+84.89333333333335+82.92666666666666+82.7016666666667+84.01500000000001+83.30425531914894
print(test/14)

print(tach/24)