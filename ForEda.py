import pandas as pd
import numpy as np
from readfolders import *
from sklearn.model_selection import train_test_split, GridSearchCV

chem, lbls, vdpresume = chosefiles()
labl = getlabel(lbls)
X = getx(chem, vdpresume, lbls['Colata(IBA)'])
var0 = [s for s in X.columns if X[s].var() == 0]
X.drop(var0, axis=1, inplace=True)
nanx = X[pd.isnull(X).any(1)].index
X.drop(nanx, inplace=True)
labl.drop(nanx, inplace=True)
corr = get_feature_correlation(X)
corr.reset_index(drop=True, inplace=True)
print('Done...')
X_with_index = pd.concat([X.index.to_series(), X], axis=1).reset_index(drop=True)
Y = {labl[i].name: labl[i] for i in labl.columns}
Y_with_index = {k: pd.concat([v.index.to_series(), v], axis=1).reset_index(drop=True) for k, v in Y.items()}
og_X_tr, og_Y_tr, og_X_te, og_Y_te = dict(), dict(), dict(), dict()
low_std = [s for s in X.columns if X[s].mean()/X[s].std() > 10]
low_var = [s for s in X.columns if X[s].mean()/X[s].var() > 10]
for k, v in Y.items():
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, v, test_size=0.2, random_state=0, stratify=v)
    og_X_te[k] = X_te
    og_X_tr[k] = X_tr
    og_Y_te[k] = Y_te
    og_Y_tr[k] = Y_tr
    # df_X_nidx_tr, df_X_nidx_te, df_Y_nidx_tr, df_Y_nidx_te = train_test_split(X_with_index, Y_with_index[k], test_size=0.2, random_state=0)
    # np_X_idx_tr, np_X_idx_te, np_Y_idx_tr, np_Y_idx_te = train_test_split(X.values, v.values, test_size=0.2, random_state=0)
    # np_X_nidx_tr, np_X_nidx_te, np_Y_nidx_tr, np_Y_nidx_te = train_test_split(X_with_index.values, Y_with_index[k].values, test_size=0.2, random_state=0)
    print(k)
print('Done...')
