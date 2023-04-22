import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


def linear_regression(x, y, test, name):
    reg = LinearRegression().fit(x, y)
    r_sq = reg.score(x, y)
    print('r_sq of', name, r_sq)
    pred = reg.predict(test)
    cross_val = cross_val_score(reg, x, y, cv=5)
    print('cross val', name, cross_val)
    coef = reg.coef_
    print('coefs', name, coef)

    return pred

def lasso_reg(x, y, test, name):
    reg = linear_model.Lasso(alpha=0.1).fit(x, y)
    r_sq = reg.score(x, y)
    print('r_sq of', name, r_sq)
    pred = reg.predict(test)
    cross_val = cross_val_score(reg, x, y, cv=5)
    print('cross val', name, cross_val)
    coef = reg.coef_
    print('coefs', name, coef)

    return pred

def lasso_reg1(x, y, test, name):
    reg = linear_model.Lasso(alpha=0.8).fit(x, y)
    r_sq = reg.score(x, y)
    print('r_sq of', name, r_sq)
    pred = reg.predict(test)
    cross_val = cross_val_score(reg, x, y, cv=5)
    print('cross val', name, cross_val)
    coef = reg.coef_
    print('coefs', name, coef)

    return pred

def random_forest_regression(x, y, test, name):
    reg = RandomForestRegressor(n_estimators=10, random_state=0).fit(x, y.ravel())

    pred = reg.predict(test)
    pred_rsq = reg.predict(x)
    r_sq = metrics.r2_score(y, pred_rsq)
    print('r_sq of', name, r_sq)

    cross_val = cross_val_score(reg, x, y.ravel(), cv=5)
    print('cross val', name, cross_val)

    return pred


def ridge_reg(x, y, test, name):
    reg = Ridge(alpha=.5)
    reg.fit(x, y)
    r_sq = reg.score(x, y)
    print('r_sq of', name, r_sq)
    pred = reg.predict(test)
    cross_val = cross_val_score(reg, x, y, cv=5)
    print('cross val', name, cross_val)
    coef = reg.coef_
    print('coefs', name, coef)

    return pred

def elastic_net_reg(x, y, test, name):
    reg = ElasticNet().fit(x, y)
    r_sq = reg.score(x, y)
    print('r_sq of', name, r_sq)
    pred = reg.predict(test)
    cross_val = cross_val_score(reg, x, y, cv=5)
    print('cross val', name, cross_val)
    coef = reg.coef_
    print('coefs', name, coef)

    return pred


plot_all = pd.read_csv('../data/all_ma.csv', sep=',')
plot_temp = plot_all['temperature']

tavg_learn = pd.read_csv('../preproccessed_data/ma_without_nan_tavg.csv', sep=',')
tmax_learn = pd.read_csv('../preproccessed_data/ma_without_nan_tmax.csv', sep=',')
tmin_learn = pd.read_csv('../preproccessed_data/ma_without_nan_tmin.csv', sep=',')
humidity_learn = pd.read_csv('../preproccessed_data/ma_without_nan_humidity.csv', sep=',')
weight_learn = pd.read_csv('../preproccessed_data/ma_without_nan_weight.csv', sep=',')

lasso_learn = tavg_learn.copy()
lasso_learn['tmax'] = tmax_learn
lasso_learn['tmin'] = tmin_learn
lasso_learn['humidity'] = humidity_learn
lasso_learn['weight'] = weight_learn


t_data_learn = tavg_learn.copy()
t_data_learn['tmax'] = tmax_learn
t_data_learn['tmin'] = tmin_learn

t_avg_max_learn = tavg_learn.copy()
t_avg_max_learn['tmax'] = tmax_learn

temp_learn = pd.read_csv('../preproccessed_data/ma_without_nan_temp.csv', sep=',')

tavg_pred = pd.read_csv('../preproccessed_data/ma_tavg.csv', sep=',')
tmax_pred = pd.read_csv('../preproccessed_data/ma_tmax.csv', sep=',')
tmin_pred = pd.read_csv('../preproccessed_data/ma_tmin.csv', sep=',')

#####
tavg_pred_lasso = pd.read_csv('../preproccessed_data/without_temp_drop_nan/ma_tavg.csv', sep=',')
tmax_pred_lasso = pd.read_csv('../preproccessed_data/without_temp_drop_nan/ma_tmax.csv', sep=',')
tmin_pred_lasso = pd.read_csv('../preproccessed_data/without_temp_drop_nan/ma_tmin.csv', sep=',')
hum_pred_lasso = pd.read_csv('../preproccessed_data/without_temp_drop_nan/ma_humidity.csv', sep=',')
weight_pred_lasso = pd.read_csv('../preproccessed_data/without_temp_drop_nan/ma_weight.csv', sep=',')
#####

lasso_pred_all = tavg_pred_lasso.copy()
lasso_pred_all['tmax'] = tmax_pred_lasso
lasso_pred_all['tmin'] = tmin_pred_lasso
lasso_pred_all['humidity'] = hum_pred_lasso
lasso_pred_all['weight'] = weight_pred_lasso

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

##################################################################################################
# Linear Regression
##################################################################################################

# tavg
pred_tavg = linear_regression(ma_tavg_learn, ma_temp_learn, tavg_pred, 'tavg')
# tmax
pred_tmax = linear_regression(ma_tmax_learn, ma_temp_learn, tmax_pred, 'tmax')
# tmin
pred_tmin = linear_regression(ma_tmin_learn, ma_temp_learn, tmin_pred, 'tmin')
# tdata
pred_tdata = linear_regression(ma_t_learn, ma_temp_learn, t_data_pred, 'tdata')
# avg, max
pred_avg_max_data = linear_regression(ma_avg_max_learn, ma_temp_learn, t_avg_max_pred, 'avg, max')

##################################################################################################
# Lasso Regression
##################################################################################################
print("\n")

# tavg
lasso_tavg = lasso_reg(ma_tavg_learn, ma_temp_learn, tavg_pred, 'lasso_tavg')
# tmax
lasso_tmax = lasso_reg(ma_tmax_learn, ma_temp_learn, tmax_pred, 'lasso_tmax')
# tmin
lasso_tmin = lasso_reg(ma_tmin_learn, ma_temp_learn, tmin_pred, 'lasso_tmin')
# tdata
lasso_tdata = lasso_reg(ma_t_learn, ma_temp_learn, t_data_pred, 'lasso_tdata')
# avg, max
lasso_avg_max_data = lasso_reg(ma_avg_max_learn, ma_temp_learn, t_avg_max_pred, 'lasso_avg, max')
#all
lasso_all = lasso_reg(lasso_learn, ma_temp_learn, lasso_pred_all, 'lasso_all')

##################################################################################################
# Random Forest Regression
##################################################################################################
print("\n")

rf_tavg = random_forest_regression(ma_tavg_learn, ma_temp_learn, tavg_pred, 'rf_tavg')

rf_tmax = random_forest_regression(ma_tmax_learn, ma_temp_learn, tmax_pred, 'rf_tmax')

rf_tmin = random_forest_regression(ma_tmin_learn, ma_temp_learn, tmin_pred, 'rf_tmin')

rf_tdata = random_forest_regression(ma_t_learn, ma_temp_learn, t_data_pred, 'rf_tdata')

rf_avg_max_data = random_forest_regression(ma_avg_max_learn, ma_temp_learn, t_avg_max_pred, 'rf_t_avg_max')

##################################################################################################
# Ridge Regression
##################################################################################################
print("\n")

ridge_tavg = ridge_reg(ma_tavg_learn, ma_temp_learn, tavg_pred, 'ridge_tavg')

ridge_tmax = ridge_reg(ma_tmax_learn, ma_temp_learn, tmax_pred, 'ridge_tmax')

ridge_tmin = ridge_reg(ma_tmin_learn, ma_temp_learn, tmin_pred, 'ridge_tmin')

ridge_tdata = ridge_reg(ma_t_learn, ma_temp_learn, t_data_pred, 'ridge_tdata')

ridge_avg_max_data = ridge_reg(ma_avg_max_learn, ma_temp_learn, t_avg_max_pred, 'ridge_t_avg_max')
#all
net_all = ridge_reg(lasso_learn, ma_temp_learn, lasso_pred_all, 'ridge_all')
##################################################################################################
# Elastic Net
##################################################################################################
print("\n")

# tavg
net_tavg = elastic_net_reg(ma_tavg_learn, ma_temp_learn, tavg_pred, 'net_tavg')
# tmax
net_tmax = elastic_net_reg(ma_tmax_learn, ma_temp_learn, tmax_pred, 'net_tmax')
# tmin
net_tmin = elastic_net_reg(ma_tmin_learn, ma_temp_learn, tmin_pred, 'net_tmin')
# tdata
net_tdata = elastic_net_reg(ma_t_learn, ma_temp_learn, t_data_pred, 'net_tdata')
# avg, max
net_avg_max_data = elastic_net_reg(ma_avg_max_learn, ma_temp_learn, t_avg_max_pred, 'net_avg, max')
#all
net_all = elastic_net_reg(lasso_learn, ma_temp_learn, lasso_pred_all, 'net_all')

plot_all = pd.read_csv('../data/all_ma.csv', sep=',')
date1 = pd.read_csv('../data/weather_magdeburg.csv', sep=',')
plot_temp = plot_all['temperature']

date1 = date1.drop(['tavg', 'tmax', 'tmin', 'tsun', 'prcp'], axis=1)

plot_reg = plot_temp.copy()
for i in range(243, 300):
    plot_reg[i] = lasso_tdata[i]

date1['temperature'] = plot_reg

date1.to_csv("regression_9.csv", index=False)

list1 = []
for i in range(730):
    list1.append(i)

plt.figure(figsize=(30, 7), dpi=300)
plt.plot(list1, lasso_tdata)
plt.plot(list1, plot_temp)
plt.title('tdata')
plt.show()


plot_all = pd.read_csv('regression_9.csv', sep=',')

plt.figure(figsize=(30, 7), dpi=300)
plt.plot(list1, plot_all['temperature'])
plt.title('tdata')
plt.show()