from readfolders import log_file_dec, generate_clean_dataset, output_classification, gridsearch_cv_out
from readfolders import test_combined_samplers, test_under_samplers, test_over_samplers
import pandas as pd
from os.path import join
from sys import path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
# ---------------------for-splitting-and-scaling-------------------------------------------------------#
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

# Create Log path string
fold = path[0]
log_file = join(fold, 'Log_SVM.txt')
svc = SVC()
gamma_list = [100, 10, 1, 0.5, 0.1, 0.05, 0.1, 'auto', 'scale']
c_list = [0.01, 0.1, 1, 10, 100, 1_000, 5_000]


@log_file_dec(log_file, True)
def svm_one_class(x: pd.DataFrame, y: pd.DataFrame):
    for i in y.columns:
        print(f'One Class SVM for defect {y.columns}')
        nu = y[i].sum() / len(y[i])
        x_train, x_test, y_train, y_test = train_test_split(x, y[i], test_size=0.2, random_state=0)
        means = x_train.mean()
        stds = x_train.std()
        x_train = (x_train - means) / stds
        x_test = (x_test - means) / stds
        for g in ['scale', 'auto', 10, 1, 0.5, 0.1, 0.05, 0.01]:
            print(f"Trying gaussian kernel with gamma = {g} ")
            one = OneClassSVM(nu=nu, kernel='rbf', gamma=g)
            one.fit(x_train)
            prediction = one.predict(x_test)  # Change the anomalies' values to make it consistent with the true values
            prediction = [1 if i == -1 else 0 for i in prediction]
            print(output_classification(y_test, prediction))
            score = one.score_samples(x_test)
            print('Customizing the threshold:')
            for t in [50, 10, 8, 5, 2, 1]:
                score_threshold = np.percentile(score, t)
                print(f'The customized score threshold for {t}% of outliers is {score_threshold:.3f}')
                customized_prediction = [1 if i < score_threshold else 0 for i in score]  # Check prediction performance
                print(output_classification(y_test, customized_prediction))


@log_file_dec(log_file, True)
def main_test(grid_search: GridSearchCV, parameters, x_train, y_train, x_test, y_test, defect: str = ""):
    if defect != "":
        print(f"Beginning tests for defect {defect}:")
    else:
        print("Beginning tests for anomaly detection")

    samplers_generator_to_test = list()
    samplers_generator_to_test.append(test_over_samplers(x_train, y_train))
    samplers_generator_to_test.append(test_under_samplers(x_train, y_train))
    samplers_generator_to_test.append(test_combined_samplers(x_train, y_train))

    for generator in samplers_generator_to_test:
        for x_resampled, y_resampled in generator:
            grid_search.fit(x_resampled, y_resampled)
            gridsearch_cv_out(x_te=x_test, y_te=y_test, trained_grid=grid_search, parameters=parameters, defect=defect)


if __name__ == '__main__':
    feats, label = generate_clean_dataset()
    # svm_one_class(x=feats, y=label)
    hyperparameters = [{'kernel': ['rbf'], 'gamma': gamma_list, 'C': c_list, 'cache_size': [1_700]},
                       {'kernel': ['linear'], 'C': c_list,  'cache_size': [1_700]},
                       {'kernel': ['poly'], 'C': c_list, 'gamma': ['auto', 'scale'], 'degree': [10, 8, 5, 3],
                        'cache_size': [1_700]}]
    for i in label.columns:
        grid = GridSearchCV(svc, hyperparameters, n_jobs=-1, verbose=2)
        f_train, f_test, l_train, l_test = train_test_split(feats, label[i], test_size=0.2, random_state=0,
                                                            stratify=label[i])
        f_means = f_train.mean()
        f_stds = f_train.std()
        f_train = (f_train - f_means) / f_stds
        f_test = (f_test - f_means) / f_stds
        main_test(grid_search=grid, parameters=hyperparameters, x_train=f_train, y_train=l_train, x_test=f_test,
                  y_test=l_test, defect=i)
