import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rex_score = re.compile('score=([\d.]+)')
rex_c = re.compile('C=([\d.]+)')
rex_time = re.compile('time=\s*([\d.]+(s|min))')
rex_gamma = re.compile('gamma=([\d.]+)')
rex_degree = re.compile('degree=(\d+)')
rex_kernel = re.compile('kernel=(\w+);')
rex_cv = re.compile('\[CV (\d+)/\d+\]')
auto_path = 'E:\\Nuovi dati\\Script_nuovi_dati_ML\\Log_SVM5E4r.txt'


def visualize_log(file: str):
    with open(file) as f:
        times = pd.DataFrame()
        for i in f:
            i_dict = dict()
            t = rex_time.search(i)
            c = rex_c.search(i)
            gamma = rex_gamma.search(i)
            kernel = rex_kernel.search(i)
            cv = rex_cv.search(i)
            degree = rex_degree.search(i)
            score = rex_score.search(i)
            if score is not None:
                if t.groups()[1].endswith('s'):
                    i_dict['time'] = float(t.groups()[0][:-1])
                elif t.groups()[1].endswith('min'):
                    i_dict['time'] = float(t.groups()[0][:-3]) * 60
                else:
                    raise ValueError('Problem in the string or in the measure of the time')
                i_dict['score'] = float(score.groups()[0])
                i_dict['cv'] = int(cv.groups()[0])
                i_dict['C'] = float(c.groups()[0])
                i_dict['degree'] = int(degree.groups()[0]) if degree is not None else np.nan
                i_dict['kernel'] = kernel.groups()[0]
                i_dict['gamma'] = float(gamma.groups()[0]) if gamma is not None else np.nan
                times = pd.concat([times, pd.DataFrame(i_dict, index=[0])], ignore_index=True)
    return times


def group_notcv(times: pd.DataFrame):
    cols = times.columns
    cols = cols.drop(['cv', 'time', 'score']).to_list()
    for i in ['Mean Score', 'Mean time']:
        cols.insert(0, i)
    del i  # Così per sfizio
    notcv = pd.DataFrame(columns=cols)
    for knl, dt_temp in times.groupby('kernel'):
        cols = dt_temp.columns[dt_temp.notna().all()]
        cols = list(cols.drop(['cv', 'time', 'score', 'kernel']))
        for lbls, cvs in dt_temp.groupby(cols):
            row = cvs.iloc[0].drop(['cv', 'time', 'score'])
            row.loc['Mean time'] = np.float32(cvs['time'].mean())
            row.loc['Mean Score'] = np.float32(cvs['score'].mean())
            notcv = pd.concat([notcv, row.to_frame().T], ignore_index=True)
    return notcv

# cspec = g.groupby(['kernel','C','gamma','degree']).size().reset_index()
# for i in [1, 10, 100, 1000]:
#      for j in [3, 5, 10]:
#          for k in [0.1, 1.0, 10.0]:
#              s5 = c5[(c5['C'] == i) & (c5['degree'] == j) & (c5['gamma'] == k)]['count'].values
#              s50 = c50[(c50['C'] == i) & (c50['degree'] == j) & (c50['gamma'] == k)]['count'].values
#              if len(s5):
#                  s5 = s5[0]
#              else:
#                  s5 = 0
#              if len(s50):
#                  s50 = s50[0]
#              else:
#                  s50 = 0
#              temp = {'5R3': s5, '5R4': s50}
#              poly[(i, j, k)] = temp


if __name__ == '__main__':
    print(f'Insert the path of file log otherwise the path: \n\'{auto_path}\'\n will be used:\t')
    file_path = input() or auto_path
    dt = visualize_log(file_path)
    g = group_notcv(dt)
    print('DOne..')
