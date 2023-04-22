import pandas as pd
from matplotlib import pyplot as plt

csv_input_bee_f_old = pd.read_csv('../data/bee_frankfurt.csv', sep=',')
csv_input_bee_f_old1 = pd.read_csv('../old_stuff/data_bearbeitet/frankfurt/frankfurt_weight.csv', sep=',')
csv_input_bee_f = pd.read_csv('../data/bee_frankfurt.csv', sep=',')

csv_input_bee_f['weight'] = csv_input_bee_f['weight'].fillna(87000.0)

for i in range(730):
    if csv_input_bee_f['weight'][i] <= 40000:
        csv_input_bee_f['weight'][i] = csv_input_bee_f['weight'][i] + 65700

csv_input_bee_f['weight'][482] = csv_input_bee_f['weight'][481]
csv_input_bee_f['weight'][547] = csv_input_bee_f['weight'][546]

for i in range(730):
    csv_input_bee_f['weight'][i] = (csv_input_bee_f['weight'][i])/1000

list1 = []
for i in range(len(csv_input_bee_f)):
    list1.append(i)


# plt.figure(figsize=(30,7), dpi=300)
# plt.plot(list1, csv_input_bee_f_old1['weight'])
# plt.plot(list1, csv_input_bee_f['weight'])
# plt.show()

csv_input_bee_f.to_csv('../preproccessed_data/fr_weight.csv', index=False)