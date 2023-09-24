from readfolders import *
import pandas as pd
from os.path import join
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------for-slitting-and-scaling-------------------------------------------------------#
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from imblearn.over_sampling import SMOTE, ADASYN


chem, lbls, vdpresume = chosefiles()
labl = getlabel(lbls)
X = getx(chem, vdpresume, lbls['Colata(IBA)'])
var0 = [s for s in X.columns if X[s].var() == 0]
X.drop(var0, axis=1, inplace=True)
nanx = X[pd.isnull(X).any(1)].index
X.drop(nanx, inplace=True)
labl.drop(nanx, inplace=True)

# parameters = [{'C': [1, 10, 100, 1000], 'kernel':['linear']},
#               {'C': [1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 1, 10]},
#               {'C': [1, 10, 100, 1000], 'kernel':['poly'], 'degree': [3, 5, 10], 'gamma':[0.1, 1, 10]}]

# parameters = [{'C': [1, 10, 100, 1000], 'kernel':['linear']}]


def _test_on_list(list_to_test: [dict], parameters: [dict]):
    for l in list_to_test:
        print(f'With {l.__class__.__name__} oversampler:')


def _test_oversampler_on_hyperparamter(hyperparameters: [dict] or str = 'auto'):
    oversamplobj = [SMOTE(), ADASYN()]
    if hyperparameters == 'auto':
        hyperparameters = [{'gamma': [10, 5, 3.5, 1, 0.5, 0.05, 0.01],  'kernel': ['rbf'], 'C': [1, 10, 100, 1000]}]


# Create Log file
fold = sys.path[0]
log_file = join(fold, 'Log_SVM.txt')
print(log_file)
orig_stderr = sys.stderr
orig_stdout = sys.stdout
f = open(log_file, 'w')
sys.stdout = Tee(f, sys.stdout)
sys.stderr = sys.stdout


svc = SVC()

few_values = {i: len(X[i].value_counts()) for i in X.columns if len(X[i].value_counts()) < 10}

best_parm = {'SMOTE': {'CRICCA': [{'gamma': [10], 'C': [10]}, {'gamma': [0.1], 'C': [1000]}],
                       'DEFAULT': [{'gamma': [0.1], 'C': [1000]}]},
             'ADASYN': {'DEPRESS': [{'gamma': [0.1], 'C': [1]}, {'gamma': [0.1], 'C': [100]}, {'gamma': [0.1], 'C': [1000]}],
                        'DEFAULT': [{'gamma': [0.1], 'C': [1]}, {'gamma': [0.1], 'C': [100]}, {'gamma': [0.1], 'C': [1000]}]}
             }
oversamplobj = [SMOTE()]
for k in oversamplobj:

    for i in labl.columns:
        print(i + ': ')
        Y = labl[i].sort_index()
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        # Per prendere solo le prime 10 000 righe
        X_resampled, Y_resampled = [i for i in k.fit_resample(X_tr, Y_tr)]
        to_prove = best_parm[k.__class__.__name__].get(i) or best_parm[k.__class__.__name__]['DEFAULT']
        to_prove += parameters
        grid = GridSearchCV(estimator=svc,
                            param_grid=to_prove,
                            scoring='f1',
                            cv=4,
                            n_jobs=9,
                            verbose=10)
        grid.fit(X_resampled, Y_resampled)  # gridsearchcv k-fold is stratified by default
        gridsearch_cv_out(X_te, Y_te, grid, parameters, i)
        print('\n')

sys.stdout = orig_stdout
sys.stderr = orig_stderr
f.close()
