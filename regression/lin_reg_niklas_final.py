import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

# learn - without nan (dropped with temp)

# pred - ma_t (730)
# in without temp drop all for combination (579)

plot_all = pd.read_csv('../data/all_ma.csv', sep=',')
plot_temp = plot_all['temperature']

tavg_learn = pd.read_csv('../preproccessed_data/ma_without_nan_tavg.csv', sep=',')
tmax_learn = pd.read_csv('../preproccessed_data/ma_without_nan_tmax.csv', sep=',')
tmin_learn = pd.read_csv('../preproccessed_data/ma_without_nan_tmin.csv', sep=',')

t_data_learn = tavg_learn.copy()
t_data_learn['tmax'] = tmax_learn
t_data_learn['tmin'] = tmin_learn

t_avg_max_learn = tavg_learn.copy()
t_avg_max_learn['tmax'] = tmax_learn

temp_learn = pd.read_csv('../preproccessed_data/ma_without_nan_temp.csv', sep=',')

tavg_pred = pd.read_csv('../preproccessed_data/ma_tavg.csv', sep=',')
tmax_pred = pd.read_csv('../preproccessed_data/ma_tmax.csv', sep=',')
tmin_pred = pd.read_csv('../preproccessed_data/ma_tmin.csv', sep=',')

t_data_pred = tavg_pred.copy()
t_data_pred['tmax'] = tmax_pred
t_data_pred['tmin'] = tmin_pred

t_avg_max_pred = tavg_pred.copy()
t_avg_max_pred['tmax'] = tmax_pred

ma_t_learn = np.array(t_data_learn).reshape((-1, 3))
ma_avg_max_learn = np.array(t_avg_max_learn).reshape((-1, 2))

ma_tmin_learn = np.array(tmin_learn).reshape((-1, 1))
ma_tmax_learn = np.array(tmax_learn).reshape((-1, 1))
ma_tavg_learn = np.array(tavg_learn).reshape((-1, 1))
ma_temp_learn = np.array(temp_learn)

# reg tavg
reg = LinearRegression().fit(ma_tavg_learn, ma_temp_learn)
r_sq = reg.score(ma_tavg_learn, ma_temp_learn)
print('r_sq of tavg', r_sq)
pred_tavg = reg.predict(tavg_pred)
cross_val = cross_val_score(reg, ma_tavg_learn, ma_temp_learn, cv=5)
print('cross val tavg', cross_val)

# reg tmax
reg = LinearRegression().fit(ma_tmax_learn, ma_temp_learn)
r_sq = reg.score(ma_tmax_learn, ma_temp_learn)
print('r_sq of tmax', r_sq)
pred_tmax = reg.predict(tmax_pred)
cross_val = cross_val_score(reg, ma_tmax_learn, ma_temp_learn, cv=4)
print('cross val', cross_val)

# reg tmin
reg = LinearRegression().fit(ma_tmin_learn, ma_temp_learn)
r_sq = reg.score(ma_tmin_learn, ma_temp_learn)
print('r_sq of tmin', r_sq)
pred_tmin = reg.predict(tmin_pred)
cross_val = cross_val_score(reg, ma_tmin_learn, ma_temp_learn, cv=4)
print('cross val', cross_val)

# reg t data
reg = LinearRegression().fit(ma_t_learn, ma_temp_learn)
r_sq = reg.score(ma_t_learn, ma_temp_learn)
print('r_sq of t data', r_sq)
pred_tdata = reg.predict(t_data_pred)
cross_val = cross_val_score(reg, ma_t_learn, ma_temp_learn, cv=5)
print('cross val', cross_val)
coef = reg.coef_
print('coefs: ', coef)

# reg avg,max data
reg = LinearRegression().fit(ma_avg_max_learn, ma_temp_learn)
r_sq = reg.score(ma_avg_max_learn, ma_temp_learn)
print('r_sq of avg,max data', r_sq)
pred_avg_max_data = reg.predict(t_avg_max_pred)
cross_val = cross_val_score(reg, ma_avg_max_learn, ma_temp_learn, cv=4)
print('cross val', cross_val)

#######################################################################################################################

#######################################################################################################################
# test with fr data

# all_tavg_learn = pd.read_csv('preproccessed_data/all_tavg.csv', sep=',')
# all_tmax_learn = pd.read_csv('preproccessed_data/all_tmax.csv', sep=',')
# all_tmin_learn = pd.read_csv('preproccessed_data/all_tmin.csv', sep=',')
#
# all_temp_learn = pd.read_csv('preproccessed_data/all_temp.csv', sep=',')
#
# all_d = all_tavg_learn.copy()
# all_d['tmax'] = all_tmax_learn.copy()
# all_d['tmin'] = all_tmin_learn.copy()
# all_d['temperature'] = all_temp_learn.copy()
# all_d = all_d.dropna()
#
# all_d_temp_learn = all_d['temperature'].copy()
# all_d_temp_learn = all_d_temp_learn.dropna()
#
# all_with_temp = all_d.drop(['temperature'], axis=1)
#
# all_array = np.array(all_with_temp).reshape((-1, 3))
#
#
# # reg mit frankfurt
# reg = LinearRegression().fit(all_array, all_d_temp_learn)
# r_sq = reg.score(all_array, all_d_temp_learn)
# print('r_sq of fr data', r_sq)
# pred_with_fr = reg.predict(t_data_pred)
#
# list5 = []
# for i in range(730):
#     list5.append(i)
#
# # tavg,max plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list5, plot_temp)
# plt.plot(list5, pred_with_fr)
# plt.title('fr')
# plt.show()

#######################################################################################################################

list1 = []
for i in range(730):
    list1.append(i)
#
# # tavg plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_temp)
# plt.plot(list1, pred_tavg)
# plt.title('tavg')
# plt.show()
#
# # tmax plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_temp)
# plt.plot(list1, pred_tmax)
# plt.title('tmax')
# plt.show()
#
# tmin plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_temp)
# plt.plot(list1, pred_tmin)
# plt.title('tmin')
# plt.show()

# # t data plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_temp)
# plt.plot(list1, pred_tdata_lasso)
# plt.title('tdata')
# plt.show()
#
# # tavg,max plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_temp)
# plt.plot(list1, pred_avg_max_data)
# plt.title('tavg,max')
# plt.show()



plot_reg = plot_temp.copy()

# for i in range(244, 300):
#     plot_reg[i] = pred_tavg[i]
#
# # t data plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_reg)
# plt.plot(list1, plot_temp)
# plt.plot(list1, tavg_pred)
# plt.plot(list1, tmax_pred)
# plt.title('tavg')
# plt.show()

# for i in range(244, 300):
#     plot_reg[i] = pred_tdata[i]
#
# # t data plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_reg)
# plt.plot(list1, plot_temp)
# plt.title('t data')
# plt.show()
#
# for i in range(244, 300):
#     plot_reg[i] = pred_avg_max_data[i]
#
# # t data plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_reg)
# plt.plot(list1, plot_temp)
# plt.title('avg + max')
# plt.show()

# for i in range(244, 300):
#     plot_reg[i] = pred_tmax[i]
#
# # t data plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_reg)
# plt.plot(list1, plot_temp)
# plt.title('tmax')
# plt.show()

for i in range(244, 300):
    plot_reg[i] = pred_avg_max_data[i]

# # avg,max data plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, plot_reg)
# plt.plot(list1, plot_temp)
# plt.plot(list1, pred_tavg)
# plt.plot(list1, pred_tmax)
# plt.title('avg, max')
# plt.show()
#
# #################################################################
# # decision made for t data
# #################################################################

# # avg,max data plot
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, pred_tavg)
# plt.plot(list1, pred_tmax)
# plt.plot(list1, pred_tmin)
# plt.title('tavg')
# plt.show()
#
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, pred_tavg)
# plt.plot(list1, pred_tmin)
# plt.title('here')
# plt.show()
#
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, pred_tmax)
# plt.title('tmax')
# plt.show()
#
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, pred_tmin)
# plt.title('tmin')
# plt.show()
#
# plt.plot(list1, tmin_pred)
# plt.title('tmin')
# plt.show()
#
# plt.figure(figsize=(30, 7), dpi=300)
# plt.plot(list1, tmax_pred)
# plt.title('tmax')
# plt.show()
