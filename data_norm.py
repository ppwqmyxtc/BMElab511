import numpy as np
import pandas as pd
import pickle
from scipy import interpolate

timestep_size = 12
vars_size = 33


def normalize_minmax(x, set_mins=0, set_maxs=0):  # x:nparrary, shape=[none * timestep_size, vars_size]
    if type(set_mins) == int:
        xmin, xmax = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
    else:
        xmin, xmax = set_mins, set_maxs
    xmins=np.tile(xmin, (x.shape[0], 1))
    xmaxs = np.tile(xmax, (x.shape[0], 1))
    norm_x=(x-xmins)/(xmaxs-xmins)
    return [norm_x, [xmin, xmax]]


def normalize_meanstd(x, set_means=0, set_stds=0):  # x:nparrary, shape=[none * timestep_size, vars_size]
    if type(set_means) == int:
        xmean, xstd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    else:
        xmean, xstd = set_means, set_stds
    xmeans = np.tile(xmean, (x.shape[0], 1))
    xstds = np.tile(xstd, (x.shape[0], 1))
    norm_x = (x-xmeans)/xstds
    return [norm_x, [xmean, xstd]]


def norm(s1_x, s2_x, norm_method):
    reshaped_s1_x = s1_x.reshape([-1, timestep_size, vars_size])
    norm_s1_x = reshaped_s1_x.reshape([len(reshaped_s1_x) * timestep_size, vars_size])
    if norm_method == 'minmax':
        [norm_s1_x, [mins, maxs]] = normalize_minmax(norm_s1_x)
    elif norm_method == 'meanstd':
        [norm_s1_x, [means, stds]] = normalize_meanstd(norm_s1_x)
    norm_s1_x = norm_s1_x.reshape([-1, timestep_size, vars_size])
    norm_s1_x = norm_s1_x.reshape([-1, timestep_size * vars_size])

    reshaped_s2_x = s2_x.reshape([-1, timestep_size, vars_size])
    norm_s2_x = reshaped_s2_x.reshape([len(reshaped_s2_x) * timestep_size, vars_size])
    if norm_method == 'minmax':
        [norm_s2_x, _] = normalize_minmax(norm_s2_x, mins, maxs)
    elif norm_method == 'meanstd':
        [norm_s2_x, _] = normalize_meanstd(norm_s2_x, means, stds)
    norm_s2_x = norm_s2_x.reshape((-1, timestep_size, vars_size))
    norm_s2_x = norm_s2_x.reshape((-1, timestep_size * vars_size))
    # 特殊处理
    norm_s1_x[np.isnan(norm_s1_x)] = 0##?补缺之后的数值还有nan?
    norm_s2_x[np.isnan(norm_s2_x)] = 0

    return norm_s1_x, norm_s2_x


print('load data...')

nodes_data_test = pd.read_excel('data/data_for_exmaple2.xlsx')
print('load finish...')
nodes_data_test = np.array(nodes_data_test)[:, 2:]
print(nodes_data_test.shape)

print('normal...')
imputed_normalize_train_x, imputed_normalize_test_x = norm(nodes_data_test, nodes_data_test, norm_method='meanstd')
print(imputed_normalize_train_x.shape, imputed_normalize_test_x.shape)
print('save...')
writer = pd.ExcelWriter('data/data_norm_for_example2.xlsx')
pd.DataFrame(imputed_normalize_train_x).to_excel(writer, sheet_name='data_norm_for_example2')
writer._save()
#
# file = open('data/xgb_test_12.pickle', 'wb')
# pickle.dump(imputed_normalize_test_x, file)
# file.close()