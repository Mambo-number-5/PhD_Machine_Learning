import os
import pandas as pd
from os.path import join, isdir
import logging as log
import re
import sys
from pprint import pprint
import time
from functools import wraps
from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import roc_curve, accuracy_score, f1_score, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours
from imblearn.combine import SMOTEENN
# from sklearn.cluster import KMeans
import torch
from torchvision import transforms

fold = join('..', 'Dataset_Parquets')

if not os.path.isdir(fold):
    print('The following folder with data files:\n"%s"\nhas not been found' % fold)
    fold = input('Please insert the folder with the data files manually or enter nothing:\n')
    if fold == "":
        sys.exit(0)

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
listsubdir = [i for i in os.listdir(fold) if isdir(join(fold, i))]


class KeyErrorMessage(str):
    def __repr__(self): return str(self)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        log.info(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


@timeit
def readdataset(path_str: str, whole: bool or None = None):
    chemistry = pd.read_parquet(join(path_str, 'chemistry.parquet.brotli'))
    labls = pd.read_parquet(join(path_str, 'defects.parquet.brotli'))
    if whole is None:
        vdpresume = pd.read_parquet(join(path_str, 'vdp-resume.parquet.brotli'))
        return chemistry, labls, vdpresume
    elif whole:
        vdptot = pd.read_parquet(join(path_str, 'vdps.parquet.brotli'))
        return chemistry, labls, vdptot
    else:
        vdptot = pd.read_parquet(join(path_str, 'vdps.parquet.brotli'))
        cols = vdptot.columns.str.rsplit('-', n=1)
        segns = {sublist[-1] for sublist in cols}
        vdps = [vdptot.filter(like=s) for s in segns]
        return chemistry, labls, vdps


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def log_file_dec(log_path: str, print_stderr: bool = False):

    def redirectedstdsys(func):
        fld = os.path.dirname(log_path)
        if not os.path.isdir(fld):
            raise OSError("Folder path: '{}' not found".format(fld))
        del fld

        @wraps(func)
        def wrapper(*args, **kwargs):
            f = open(log_path, 'a')
            t = time.localtime()
            now = time.strftime("%d/%m/%y----%H:%M:%S", t)
            f.write(f'Local (clock of the computer) time: {now}')
            del now, t
            f.write(f'Redirecting output of funcion {func.__name__}:\n')
            orig_stdout = sys.stdout
            sys.stdout = Tee(f, sys.stdout)
            if print_stderr:
                orig_stderr = sys.stderr
                sys.stderr = sys.stdout
                result_og_func = func(*args, **kwargs)
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
            else:
                result_og_func = func(*args, **kwargs)
                sys.stdout = orig_stdout
            f.close()
            return result_og_func
        return wrapper
    return redirectedstdsys


def _modify_data_objects(iterable_of_samplers,  x_to_resample, y_to_resample, msg: str = ""):
    for i in iterable_of_samplers:
        print(f'Resampling with {i.__class__.__name__} ' + msg + ':')
        x_res, y_res = i.fit_resample(x_to_resample, y_to_resample)
        yield x_res, y_res


def test_over_samplers(x_to_res, y_to_res):
    oversamplers_to_test = [SMOTE(), ADASYN()]
    return _modify_data_objects(oversamplers_to_test, x_to_res, y_to_res, msg='over-sampler')


def test_under_samplers(x_to_res, y_to_res):
    # kmm = KMeans(n_clusters=8)
    # cl_est = ClusterCentroids(kmm)
    undersamplers_to_test = [ClusterCentroids(random_state=0), EditedNearestNeighbours()]
    return _modify_data_objects(undersamplers_to_test, x_to_res, y_to_res, 'under-sampler')


def test_combined_samplers(x_to_res, y_to_res):
    combined_to_test = [SMOTEENN(random_state=0)]
    return _modify_data_objects(combined_to_test, x_to_res, y_to_res, 'sampler')


def _date_filter_function(dat_filter: str):
    if dat_filter == "from":
        def return_filter(a_t: int, m_t: int, d_t: int, a: int, m: int, d: int, file_path: str, target_list: list):
            if a > a_t or (a == a_t and m > m_t) or (a == a_t and m == m_t and d >= d_t):
                target_list.append(file_path)
    elif dat_filter == "to":
        def return_filter(a_t: int, m_t: int, d_t: int, a: int, m: int, d: int, file_path: str, target_list: list):
            if a < a_t or (a == a_t and m < m_t) or (a == a_t and m == m_t and d <= d_t):
                target_list.append(file_path)
    elif dat_filter == "between":
        def return_filter(a_b: int, m_b: int, d_b: int, a_e: int, m_e: int, d_e: int, a: int, m: int, d: int, file_path: str, target_list: list):
            if a_b < a < a_e or ((a <= a_e and m < m_e) and (a >= a_b and m > m_b)) or ((a == a_e and m == m_e and d <= d_e) and (a == a_b and m == m_b and d >= d_b)):
                target_list.append(file_path)
    else:
        raise ValueError('Error in the defining of the filter for the date function...')
    return return_filter


@timeit
def chosefiles(det: str = "", date_filter: str = "from", date_between_end: str = "", timeseries: bool = None) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if det == "":
        target = list()
        for i in listsubdir:
            try:
                a, m, d = i.split('-')
                d = d[:2]
                _, _, _ = [int(i) for i in (d, m, a)]
                target.append(i)
            except ValueError:
                log.error(f'The subfolder {i} has not the name of a date, ignoring this directory')
    else:
        filt_func = _date_filter_function(date_filter)
        try:
            d, m, a = det.split('-')
            d_t, m_t, a_t = [int(i) for i in (d, m, a)]
        except ValueError:
            log.critical('Critical error while parsing the target date(s)')
            raise KeyboardInterrupt('Manually interruption...')
        target = list()
        if date_filter != "between":
            for i in listsubdir:
                try:
                    a, m, d = i.split('-')
                    d = d[:2]
                    d, m, a = [int(f) for f in (d, m, a)]
                except ValueError:
                    log.error(f'The subfolder {i} has not the name of a date, ignoring this directory')
                filt_func(a_t, m_t, d_t, a, m, d, i, target)
        else:
            try:
                d, m, a = date_between_end.split('-')
                d_e, m_e, a_e = [int(i) for i in (d, m, a)]
            except ValueError:
                log.critical('Critical error while parsing the target date(s)')
                raise KeyboardInterrupt('Manually interruption...')
            for i in listsubdir:
                try:
                    a, m, d = i.split('-')
                    d = d[:2]
                    d, m, a = [int(f) for f in (d, m, a)]
                except ValueError:
                    log.error(f'The subfolder {i} has not the name of a date, ignoring this directory')
                filt_func(a_t, m_t, d_t, a_e, m_e, d_e, a, m, d, i, target)
    chemlis = list()
    lablslis = list()
    vdpresumelis = list()
    for i in target:
        chemdt, lablsdt, vdpresumedt = readdataset(join(fold, i), whole=timeseries)
        chemlis.append(chemdt)
        lablslis.append(lablsdt)
        vdpresumelis.append(vdpresumedt)
    chemistry = pd.concat(chemlis, verify_integrity=True)
    labels = pd.concat(lablslis, verify_integrity=True)
    vdpresume = pd.concat(vdpresumelis, verify_integrity=True)
    return chemistry, labels, vdpresume


def getlabel(slc: pd.DataFrame, build_interactive_filt: bool = False, filt_defs: dict or None = None):
    if build_interactive_filt:
        yes = '\n Yes (or y/ye) to begin creating filters or another key to refuse)'
        answer = input('Do you want to filter the defects?' + yes)
        if answer.lower() != 'y' and answer.lower() != 'ye' and answer.lower() != 'yes':
            return _getlabel(slc=slc)
        print('Beginning creating a filter:')
        answer = input('Do you want the filter automatically applied to all the defects?' + yes)
        cols_class = [i for i in slc.columns if i.endswith('_Class')]
        defs = {j for i in cols_class for j in slc[i].unique() if j is not None and str(j) != 'nan'}
        if answer.lower() == 'y' and answer.lower() == 'ye' and answer.lower() == 'yes':
            filtdefs = _getfiltprop()
            filtall = dict()
            for i in defs:
                filtall[i] = filtdefs
            return _getlabel(slc, filtall)
        else:
            filtdefs = _getdefsprop(defs)
            return _getlabel(slc, filtdefs)

    elif filt_defs is not None:
        cols_class = [i for i in slc.columns if i.endswith('_Class')]
        defs = {j for i in cols_class for j in slc[i].unique() if j is not None and str(j) != 'nan'}
        if 'all' in filt_defs.keys():
            filtall = dict()
            for i in defs:
                filtall[i] = filt_defs.get('all')
            return _getlabel(slc, filtall)
        else:
            return _getlabel(slc, filt_defs)

    else:
        return _getlabel(slc=slc)


def _getlabel(slc: pd.DataFrame, filt: dict or None = None) -> pd.DataFrame:
    cols_class = [i for i in slc.columns if i.endswith('_Class')]
    defs = tuple({j for i in cols_class for j in slc[i].unique() if j is not None and str(j) != 'nan'})
    df = pd.DataFrame(columns=defs)
    for i in defs:
        tmp = pd.Series([False] * len(slc), index=slc.index)
        if filt is None:
            for j in cols_class:
                tmp |= (slc[j] == i)
            df[i] = tmp
        else:
            restrain = filt.get(i)
            if bool(restrain):
                for j in cols_class:
                    colfilt = dict()
                    for k, v in restrain.items():
                        for col in slc.columns:
                            if col.startswith(j[:4]) and col.endswith(k):
                                colfilt[col] = v  # To select the right column to which apply the filter
                                break
                    tmp2 = (slc[j] == i)
                    for k, v in colfilt.items():
                        tmp2 &= (slc[k] > v)  # Verify condition for that specific defect
                    tmp |= tmp2  # Contain true if in that slice there is at least one defect that verify the conditions
                df[i] = tmp
            else:
                for j in cols_class:
                    tmp |= (slc[j] == i)
                df[i] = tmp
    df.sort_index(inplace=True)
    df['ANOMALY'] = df.any(axis=1)
    return df.astype('uint8')


def _getfiltprop(only: bool = True,  defect: str or None = None) -> dict:
    menu = """1: Add/Substitute filter for gravity
    2: Add/Substitute filter for depth
    3: Add/Substitute filter for percentage of the defect present in the slice
    4: Add/Substitute filter for percentage of the slice length covered by the defect
    5: Remove a filter for the defect
    0: Exit
    """
    if only:
        defect = 'all'
    choice = 1000
    result = dict()
    while choice != 0:
        print(menu)
        choice = input('Select propreties of' + defect + 'defect/s to filter by typing the numbers above:\n')
        try:
            choice = int(choice.strip())
        except ValueError:
            print('Not recognized key, please try again\n')
            continue
        else:
            if choice == 1:
                answer = input('What is the minimum gravity for the defect? (Must be an integer between 0 and 5)')
                answer = int(answer.strip())
                if answer < 1 or answer > 5:
                    print('Gravity of the defect outside of the avalaible range.')
                    print('Please try again...')
                    continue
                result['_GradoDifetto'] = answer

            elif choice == 2:
                answer = input('What is the minimum depth for the defect? ')
                answer = float(answer.strip())
                result['Profondita[mm]'] = answer

            elif choice == 3:
                answer = input('What is the minimum percentage of the slice length that the defect should cover? ')
                answer = float(answer.strip())
                result['slice cov. by defect'] = answer

            elif choice == 4:
                answer = input('What is the minimum percentage of the defect length that the should be in the slice? ')
                answer = float(answer.strip())
                result['defect in slice'] = answer
            elif choice == 5:
                j = menu.find('5')
                print(menu[:j - 1])
                answer = input('What of the above features you want to be removed?')
                answer = int(answer.strip())
                if answer == 1:
                    key = '_GradoDifetto'
                elif answer == 2:
                    key = 'Profondita[mm]'
                elif answer == 3:
                    key = 'slice cov. by defect'
                elif answer == 4:
                    key = 'defect in slice'
                else:
                    print('Key not found, please try again')
                    continue
                result.pop(key, None)

            elif choice == 0:
                print('Filter for' + defect + ' defect/s being created...')

            else:
                print(' Choice not valid, please try again')

        if only:
            pprint(result)

    return result


def _getdefsprop(defects: set) -> dict:
    print('Creating defects filter')
    dictdefs = {i: el for i, el in enumerate(defects, 2)}
    menu = '''0: Exit
    1: Remove defect from filter
    '''
    for i in range(2, len(dictdefs) + 2):
        menu += str(i) + ': Add/Substitute defect' + dictdefs[i] + '\n'

    idx2 = menu.find('2:')
    menu2 = menu[idx2]
    menu2 = re.sub(r'(\d):', _repl, menu2)
    del idx2

    filtdefs = dict()
    choice = 1000
    choice_str = "The defects to filter in dataset can be added (or removed) to filter by typing the above options..."
    while choice != 0:
        choice = input(choice_str)
        try:
            choice = int(choice.strip())
            if choice > len(dictdefs) + 2:
                raise ValueError
        except ValueError:
            print('Not recognized key, please try again\n')
            continue
        else:
            if choice > 1:
                filtdefs[dictdefs[choice]] = _getfiltprop(only=False)
            elif choice == 1:
                print(menu2)
                choice = input('Select what defect to remove from the filter, or enter an invalid key')
                try:
                    choice = int(choice.strip())
                    choice += 2
                    if choice > len(dictdefs) + 2:
                        raise ValueError
                except ValueError:
                    print('Not recognized key, please try again\n')
                    continue
                else:
                    filtdefs.pop(dictdefs[choice], None)
            else:
                print('Filter created, here it is:')
                pprint(filtdefs)
    return filtdefs


def _repl(matchobj):
    m = str(int(matchobj.group(1)) - 1)
    return m + ': '


def getx(chemistr: pd.DataFrame, vdpres: pd.DataFrame, labs: pd.Series, on_iba: bool = False) -> pd.DataFrame:
    selchem = chemistr.loc[:, 'AA':'ZR']
    if on_iba:
        info = pd.merge(selchem, labs, left_index=True, right_on=labs.name, how='right')
        info = info.drop(labs.name, axis=1)
        the_x = pd.concat([info, vdpres], axis=1)
        new_index_name = {the_x.index.names[0]: the_x.index.names[0].replace('Par', 'IBA')}
        the_x.rename_axis(index=new_index_name, inplace=True)
    else:
        the_x = selchem.join(vdpres, how='right')
    the_x = the_x.sort_index()
    return the_x


def get_feature_correlation(df: pd.DataFrame, top_n: int or None = None, corr_method: str = 'spearman',
                            remove_duplicates: bool = True, remove_self_correlations: bool = True,
                            unsigned: bool = True):
    """
    Compute the feature correlation and sort feature pairs based on their correlation

    :param df: The dataframe with the predictor variables
    :type df: pandas.core.frame.DataFrame
    :param top_n: Top N feature pairs to be reported (if None, all the pairs will be returned)
    :param corr_method: Correlation computation method
    :type corr_method: str
    :param remove_duplicates: Indicates whether duplicate features must be removed
    :type remove_duplicates: bool
    :param remove_self_correlations: Indicates whether self correlations will be removed
    :type remove_self_correlations: bool
    :param unsigned: Indicates the use the absolute value of the correlation
    :type remove_self_correlations: bool
    :return: pandas.core.frame.DataFrame
    """
    if unsigned:
        corr_matrix = df.corr(method=corr_method).abs()
        corr_str = 'Correlation (abs)'
    else:
        corr_matrix = df.corr(method=corr_method)
        corr_str = 'Correlation '

    corr_matrix_us = corr_matrix.unstack()
    sorted_correlated_features = corr_matrix_us.sort_values(kind="quicksort", ascending=False).reset_index()

    # Remove comparisons of the same feature
    if remove_self_correlations:
        sorted_correlated_features = sorted_correlated_features[
            (sorted_correlated_features.level_0 != sorted_correlated_features.level_1)
        ]

    # Remove duplicates
    if remove_duplicates:
        sorted_correlated_features = sorted_correlated_features.iloc[:-2:2]

    # Create meaningful names for the columns
    sorted_correlated_features.columns = ['Feature 1', 'Feature 2', corr_str]

    if top_n:
        return sorted_correlated_features[:top_n]

    return sorted_correlated_features


def generate_clean_dataset(from_date: str = "", timeseries: bool or None = None, build_interactive_filt: bool = False, filter_defs: dict or None = None, merge_chem_on_iba: bool = False, clean: bool = True) -> [pd.DataFrame, pd.DataFrame]:
    chem, lbls, vdpresume = chosefiles(from_date, timeseries=timeseries)
    labl = getlabel(lbls, build_interactive_filt=build_interactive_filt, filt_defs=filter_defs)
    x = getx(chem, vdpresume, lbls['Colata(IBA)'], on_iba=merge_chem_on_iba)
    if clean:
        var0 = [s for s in x.columns if x[s].var() is not pd.NA and x[s].var() == 0]
        x.drop(var0, axis=1, inplace=True)
        if timeseries:
            i = 0
            for i, j in enumerate(x.columns):
                if j[0] == '[':
                    break
            del j
            for col in x.columns[:i]:
                nanx = x[x[col].isna()].index
                x.drop(nanx, inplace=True)
                labl.drop(nanx, inplace=True)
        else:
            nanx = x[pd.isnull(x).any(1)].index
            x.drop(nanx, inplace=True)
            labl.drop(nanx, inplace=True)
    return x, labl


def conf_matrix_test(reals, predicted) -> str:
    tn, fp, fn, tp = confusion_matrix(reals, predicted).ravel()
    s1 = 'CONFUSION MATRIX'.center(60, '-') + '|'
    s2 = 'PREDICTED'.center(20, '-') + '|' + 'ACTUAL'.center(39, '-') + '|'
    s3 = ' ' * 20 + '|' + 'DEFECTED'.center(19, '-') + '|' + 'NOT DEFECTED'.center(19, '-') + '|'
    s4 = 'DEFECTED'.center(20) + '|' + '{:^19}|{:^19}|'.format(tp, fp)
    s5 = 'NOT DEFECTED'.center(20) + '|' + '{:^19}|{:^19}|'.format(fn, tn)
    s6 = '-' * 60 + '|'
    res = '\n'.join((s1, s2, s3, s4, s5, s6)) + '\n'
    return res


def print_parameters(trained_grid: GridSearchCV) -> str:
    grid_cols_to_print = ('mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time',  'mean_score_time',
                          'rank_test_score')
    to_result = [pd.Series(trained_grid.cv_results_[col], name=col) for col in grid_cols_to_print]
    resultcv = pd.concat(to_result, axis=1)
    for i in ('mean_fit_time', 'std_fit_time', 'mean_score_time'):
        if resultcv[i].mean() > 300:
            resultcv[i] /= 60
            n = i + ' [min]'
        else:
            n = i + ' [s]'
        resultcv.rename(columns={i: n}, inplace=True)
    params = pd.DataFrame.from_dict(trained_grid.cv_results_['params'])
    to_print = pd.concat([params, resultcv], axis=1)
    return to_print.to_markdown(index=False, tablefmt='grid')


def report_gridsearch_cv_hyperparameters(trained_grid: GridSearchCV, parameters, defect: str):
    msg = '\n\n'
    msg += print_parameters(trained_grid)
    msg += '\n\n'
    msg += f'The {trained_grid.estimator.__class__.__name__} best estimator between parameters:\n'
    msg += f'{parameters}\n for the \'{defect}\' defect is the following:\n'
    msg += str(trained_grid.best_params_)
    return msg


def output_classification(y_te, predictions):
    msg_out = 'Here is the confusion matrix:\n'
    msg_out += conf_matrix_test(y_te, predictions)
    msg_out += '\nHere is the classification report:\n'
    msg_out += classification_report(y_te, predictions, target_names=('NOT DEFECTED', 'DEFECTED'))
    return msg_out


def gridsearch_cv_out(x_te, y_te, trained_grid: GridSearchCV, parameters, defect: str):
    msg_before_test = report_gridsearch_cv_hyperparameters(trained_grid, parameters, defect)
    print(msg_before_test)
    grid_predictions = trained_grid.predict(x_te)
    msg_out = output_classification(y_te, grid_predictions)
    print(msg_out)


# For deep learning
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame or None = None):
        self.data = data

        if isinstance(labels, pd.DataFrame):
            self.y = labels.sort_index(axis=1)
        elif labels is not None:
            self.y = pd.DataFrame(labels)
        else:
            self.y = None

    def __getitem__(self, index):
        row_np = self.data.iloc[index].values
        if self.y is None:
            return row_np
        elif self.y.shape[1] > 1:
            lbl_df = self.y.iloc[index].values
            res_lis = [(row_np, i) for i in lbl_df]
            return res_lis
        else:
            lbl_df = self.y.iloc[index].values
            return row_np, lbl_df

    def __len__(self):
        return len(self.data)


def generate_custom_datasets(from_date: str = "", time_series: bool = False, build_interactive_filt: bool = False,
                             filter_defs: dict or None = None, merge_chem_on_iba: bool = False, clean: bool = True
                             ) -> [CustomDataset, CustomDataset]:
    x, y = generate_clean_dataset(from_date=from_date, timeseries=time_series,
                                  build_interactive_filt=build_interactive_filt, filter_defs=filter_defs,
                                  merge_chem_on_iba=merge_chem_on_iba, clean=clean)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

    cust_train = CustomDataset(data=x_train, labels=y_train)
    cust_test = CustomDataset(data=x_test, labels=y_test)

    return cust_train, cust_test


if __name__ == '__main__':
    print('In main...')
