#!/usr/bin/env python
# coding: utf-8

import collections
import gc
import itertools
import math
import numbers
import os
import pathlib
import random
import re
import zipfile
from collections import Counter, OrderedDict
from datetime import datetime
from functools import reduce
from pprint import pprint
import datetime as dt

import lightgbm as gbm
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sklearn.metrics
import statsmodels.distributions.empirical_distribution as edf
from joblib import Parallel, delayed
import joblib
from matplotlib.pyplot import cm
from numpy import array, histogram, histogram_bin_edges
from scipy.interpolate import interp1d
from scipy.stats import (bartlett, chi2_contingency, f_oneway, fisher_exact,
                         kruskal, kstest, levene, mannwhitneyu, mode,
                         normaltest, pearsonr, sem, spearmanr, t, ttest_1samp,
                         ttest_ind, ttest_rel, wilcoxon)
from seaborn import heatmap
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, log_loss, recall_score, roc_auc_score
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder,
                                   PowerTransformer, label_binarize)
from sklearn.utils import (assert_all_finite, check_consistent_length,
                           column_or_1d, resample)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.multiclass import type_of_target
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportions_ztest
from tqdm.notebook import tqdm
import plotly.graph_objs as go
import plotly.offline as po
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# Выводит дату создания (git pull) и последнего локального изменения библиотеки
fl = os.path.join(os.path.dirname(__file__), os.path.basename(__file__))
created = dt.datetime.fromtimestamp(os.path.getctime(fl))
modified = dt.datetime.fromtimestamp(os.path.getmtime(fl))
fmt = "%d.%m.%Y %H:%M"
print(f'{os.path.basename(__file__)} version: Created: {created.strftime(fmt)}, Modified: {modified.strftime(fmt)}.')


def stratified_split(data=None,
                     target=None,
                     list_of_vars_for_strat=None,
                     sort_by_var=None,
                     size_of_test=None,
                     drop_technical=False,
                     random_state=42):

    """
    Стратифицированно бьет данные на трейн и тест.
    Подаваемые параметры:
    
    data - передаваемые данные
    target - название таргетной переменной
    list_of_vars_for_strat - список переменных, по которым производится стратификация
    sort_by_var - переменная, по которой производится группировка (id клиентов/транзакций и пр). В этой переменной 
    не должно быть пропусков, но если пропуски заполнить одним значением, метод "решит", что это один и тот же клиент. 
    Поэтому если есть пропуски, то их следует заполнять уникальными, не встреченными ранее в данных значениями
    size_of_test - размер тестовой выборки
    drop_technical - удалить ли "технические" переменные (лист переменных, по которым делается стратификация, 
    группирующая переменная)
    
    """
    
    max_target = data.groupby(sort_by_var).aggregate({target: 'max'})
    max_target = max_target.reset_index()
    
    data = pd.merge(data, max_target, on = sort_by_var, suffixes = ["", "_max"])
    
    target1 = target+"_max"
    list_of_vars_for_strat1 = list_of_vars_for_strat.copy()
    
    if len(list_of_vars_for_strat1) == 0:
        list_of_vars_for_strat1 = [target1]
    if target in list_of_vars_for_strat1:
        list_of_vars_for_strat1.remove(target)
        list_of_vars_for_strat1.append(target1)
    else:
        list_of_vars_for_strat1.append(target1)
        
    for i in list_of_vars_for_strat1:
        if i == list_of_vars_for_strat1[0]:
            data['For_stratify'] = data[i].astype('str')
        else:
            data['For_stratify'] += data[i].astype('str')
        
    data_nodup = data[[sort_by_var, 'For_stratify', target1]].drop_duplicates(subset = sort_by_var)
    
    
    train, test, target_train, target_test = train_test_split(data_nodup, data_nodup[target1], 
                                                test_size = size_of_test, 
                                                    stratify = data_nodup['For_stratify'], random_state = random_state)

    X_train = data[data[sort_by_var].isin(train[sort_by_var])].copy()
    train_index = X_train.index
    y_train = data.iloc[train_index][target].copy()
    X_test = data[data[sort_by_var].isin(test[sort_by_var])].copy()
    test_index = X_test.index
    y_test = data.iloc[test_index][target].copy()
   

    if drop_technical == True:
        
        X_train.drop(list_of_vars_for_strat1, axis = 1, inplace = True)
        X_train.drop(sort_by_var, axis = 1, inplace = True)
        
        X_test.drop(list_of_vars_for_strat1, axis = 1, inplace = True)
        X_test.drop(sort_by_var, axis = 1, inplace = True)
        
    else:
        X_train.drop(target1, axis = 1, inplace = True)
        X_test.drop(target1, axis = 1, inplace = True)
    
    X_train.drop(target, axis = 1, inplace = True)
    X_train.drop('For_stratify', axis = 1, inplace = True)
    
    X_test.drop(target, axis = 1, inplace = True)
    X_test.drop('For_stratify', axis = 1, inplace = True)   
    
    return X_train, X_test, y_train, y_test


def attributes_list(data, columns):
    
    """
    Рассчитывает статистики по выборке. Подаются данные и список колонок
    Статистики:
    
    Тип переменной - type_val
    Количество значений - count_val
    Количество уникальных значений - count_dist
    Количество пропусков - count_miss
    Минимальное значение - min_val
    Максимальное значение - max_val
    Медиана - val_mediana
    Мода - moda_val
    Количество наблюдений, равных моде - count_value_moda
    Стандартное отклонение - stand_d_val
    1й перцентиль - percentile_1
    2й перцентиль - percentile_2
    5й перцентиль - percentile_5
    95й перцентиль - percentile_95
    98й перцентиль - percentile_98
    99й перцентиль - percentile_99
    """
    
    attribute_list =pd.DataFrame()
    attribute_list['attribute'] = columns
    count_dist = []
    count_val = []
    count_miss = []
    type_val = []
    min_val = []
    max_val = []
    val_mediana = []
    moda_val = []
    count_value_moda = [] #кол-во элементов со значениме моды
    stand_d_val = []
    perc_1 = []
    perc_2 = []
    perc_5 = []
    perc_95 = []
    perc_98 = []
    perc_99 = []
    k = 0

    for i in attribute_list['attribute']:
        type_val.append(data[i].dtype)
        count_val.append(data[i].count())
        count_miss.append(data[i].isna().sum())
        count_dist.append(data[i].nunique())
        if data[i].dtype != 'O':
            k2 = np.nanmax(data[i])
            k3 = np.nanmin(data[i])
            k4 = data[i].median(skipna=True)
            k5 = np.nanpercentile(data[i], 1)
            k6 = np.nanpercentile(data[i], 2)
            k7 = np.nanpercentile(data[i], 5)
            k8 = np.nanpercentile(data[i], 95)
            k9 = np.nanpercentile(data[i], 98)
            k10 = np.nanpercentile(data[i], 99)
            if data[i].isna().sum()==data[i].shape[0]:
                k11 = np.nan
                k12 = np.nan
            else:
                k11 = list(data[i].mode())[0]
                k12 = sum(np.where(data[i] == list(data[i].mode())[0] , 1, 0))
            k13 = np.nanstd(data[i])
        else:
            k2 = -1000
            k3 = -1000
            k4 = -1000
            k5 = -1000
            k6 = -1000
            k8 = -1000
            k9 = -1000
            k10= -1000
            k11 = -1000
            k12 = -1000
            k13 = -1000
            
        min_val.append(k3) 
        max_val.append(k2) 
        val_mediana.append(k4)
        moda_val.append(k11)
        count_value_moda.append(k12) 
        stand_d_val.append(k13)
        perc_1.append(k5)
        perc_2.append(k6)
        perc_5.append(k7)
        perc_95.append(k8)
        perc_98.append(k9)
        perc_99.append(k10)
        
        k = k+1
        if k % 100 ==0:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Number of finished repetitions:', k , '| time: ' , tm)
            
    attribute_list['type_val'] = type_val
    attribute_list['count_val'] = count_val
    attribute_list['count_dist'] = count_dist
    attribute_list['count_miss'] = count_miss
    attribute_list['min_val'] = min_val
    attribute_list['max_val'] = max_val
    attribute_list['val_mediana'] = val_mediana  
    attribute_list['moda_val'] = moda_val
    attribute_list['count_value_moda'] = count_value_moda
    attribute_list['stand_d_val'] = stand_d_val
    attribute_list['percentile_1'] = perc_1 
    attribute_list['percentile_2'] = perc_2 
    attribute_list['percentile_5'] = perc_5 
    attribute_list['percentile_95'] = perc_95 
    attribute_list['percentile_98'] = perc_98 
    attribute_list['percentile_99'] = perc_99 
    
    gc.collect()
    
    return attribute_list


def attributes_list_new(data, 
                        columns, 
                        percentiles_list = None,
                        show_progress = True):
    
    """
    Рассчитывает статистики по выборке. Подаются данные и список колонок
    
    Статистики:
    
    Тип переменной - type_val
    Количество значений - count_val
    Количество уникальных значений - count_dist
    Количество пропусков - count_miss
    Минимальное значение - min_val
    Максимальное значение - max_val
    Медиана - val_mediana
    Мода - moda_val
    Количество наблюдений, равных моде - count_value_moda
    Стандартное отклонение - stand_d_val
    1й перцентиль - percentile_1
    2й перцентиль - percentile_2
    5й перцентиль - percentile_5
    95й перцентиль - percentile_95
    98й перцентиль - percentile_98
    99й перцентиль - percentile_99
    """

    if percentiles_list is None:
        percentiles = 0.01 * np.array([1, 2, 5, 95, 98, 99])
    else:
        percentiles = 0.01 * np.array(percentiles_list)

    data_lenght = len(data)
    data_width = data.shape[1]
    D = data[columns].describe(include = 'all', percentiles=percentiles).T
    D.rename(columns={'50%':'val_mediana', 
                      'min':'min_val', 
                      'max':'max_val', 
                      'count':'count_val', 
                      'std':'stand_d_val'}, inplace = True)

    attribute_list = pd.DataFrame()
    attribute_list['attribute'] = columns
    cm = data[columns].isna().sum(axis=0)
    
    moda_val = []
    count_dist = []
    count_value_moda = [] #кол-во элементов со значением моды

    if show_progress == True:
        iterable = tqdm(data[columns].iteritems(), desc='Attribute_list', total=data_width)
    else:
        iterable = data[columns].iteritems()
    
    for feature_name, feature_column in iterable:
        vc = feature_column.value_counts()
        count_dist.append(len(vc))
        if feature_column.dtype != 'O':
            if cm[feature_name] == data_lenght:
                k11 = np.nan
                k12 = np.nan
            else:
                k11 = vc.index[0]
                k12 = vc.iloc[0]
        else:
            k11 = -1000
            k12 = -1000

        moda_val.append(k11)
        count_value_moda.append(k12) 
    attribute_list['moda_val'] = moda_val
    attribute_list['count_miss'] = cm.values
    attribute_list['type_val'] = data[columns].dtypes.values
    attribute_list['count_dist'] = count_dist
    attribute_list['count_value_moda'] = count_value_moda
   
    return pd.merge(attribute_list, D, left_on = 'attribute', right_index = True)


def get_s_stat(y, z, N=None, category=False, encode = False, groupped_s = 'auto'):
    
    """
    Оценить S_stat 
    y, z - векторы для оценки
    N - количество наблюдений в бине
    category - категориальные переменные или нет? категориальные переменные не бьются на куски
    encode - делать LabelEncoder для категориальных переменных или нет? 
    groupped_s - как бинить нумерические переменные. Варианты: 
    'No bin' - отсутствие бининга
    'N number bins' - N означает количество бинов
    'N number obs' - N означает количество наблюдений в одном бине (количество наблюдений делится на N - получается количество бинов)
    'N share unique' - N означает долю от количества уникальных значений переменной. Количество бинов рассчитывается как round(len(np.unique(everything))*N)
    ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] - методы подбора в функции numpy.histogram_bin_edges¶
    Ссылка: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    
    """
    if y.dropna().shape[0] == 0 or z.dropna().shape[0] == 0:
        return -10
    else:
       
        if category == True and encode == True:

            everything = pd.concat([y, z])
            encoder = LabelEncoder().fit(everything.fillna('MISSING'))

            if len(encoder.classes_) == 0:
                return -10

            else:

                group_y = encoder.transform(y.fillna('MISSING'))
                group_z = encoder.transform(z.fillna('MISSING'))

        elif category == True and encode == False:

            if y.nunique() == 0 or z.nunique() == 0:
                return -10

            else:

                min_all = min(min(y), min(z))

                group_y = y.fillna(min_all-1)
                group_z = z.fillna(min_all-1)


        else:

            '''Calculate fit metrics for given target and predicted outcomes.'''

            min_all = min(min(y), min(z))
            max_all = max(max(y), max(z))

            if math.isnan(min_all):
                min_all = 0

            if math.isnan(max_all):
                max_all = 0

            y = y.fillna(min_all-1)
            z = z.fillna(min_all-1)
            everything = pd.concat([y, z])

            if max_all - min_all != 0:

                everything = pd.concat([y, z])

                if groupped_s in ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']:

                    try:
                        sel_bins = histogram_bin_edges(everything, bins = groupped_s)
                        keys = sel_bins[:-1]

                        yy = pd.DataFrame()
                        zz = pd.DataFrame()

                        counter_y = histogram(y, sel_bins, density = False)[0]
                        counter_z = histogram(z, sel_bins, density = False)[0]

                        yy['val_y'] = counter_y/sum(counter_y)
                        yy.index = keys
                        zz['val_z'] = counter_z/sum(counter_z)
                        zz.index = keys

                        del(sel_bins)
                        gc.collect()

                    except MemoryError:
                        print('MemoryError - hists')
                        group_y = y
                        group_z = z

                        counter_y = collections.Counter(group_y)
                        counter_z = collections.Counter(group_z)

                        yy = pd.DataFrame()
                        yy['val_y'] = counter_y.values()
                        yy['val_y'] = yy['val_y'] /len(y)
                        yy.index = counter_y.keys()
                        #yy = yy.loc[~yy.index.duplicated(keep='first')]

                        zz = pd.DataFrame()
                        zz['val_z'] = counter_z.values()
                        zz['val_z'] = zz['val_z'] /len(z)
                        zz.index = counter_z.keys()
                        #zz = zz.loc[~zz.index.duplicated(keep='first')]

                if groupped_s == 'No bin':

                    #k = (max_all - min_all)/ N
                    group_y = y
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = z
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N number bins':

                    k = (max_all - min_all)/ N

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N number obs':

                    number = round(everything.shape[0]/N)
                    k = (max_all - min_all)/ number

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N share unique' and N >= 0 and N <= 1:
                    number = round(len(np.unique(everything))*N)
                    k = (max_all - min_all)/ number

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N share unique' and (N < 0 or N > 1):
                    raise ValueError('N is a share of number of unique values! It should be in a range [0, 1]')

            else:
                return 0

        if not (groupped_s in ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] and category == False): 

            counter_y = collections.Counter(group_y)
            counter_z = collections.Counter(group_z)

            yy = pd.DataFrame()
            yy['val_y'] = counter_y.values()
            yy['val_y'] = yy['val_y'] /len(y)
            yy.index = counter_y.keys()
            #yy = yy.loc[~yy.index.duplicated(keep='first')]

            zz = pd.DataFrame()
            zz['val_z'] = counter_z.values()
            zz['val_z'] = zz['val_z'] /len(z)
            zz.index = counter_z.keys()
            #zz = zz.loc[~zz.index.duplicated(keep='first')]

        dd_all = pd.concat([yy, zz], axis=1)
        dd_all = dd_all.fillna(0)
        dd_all['abs'] = abs(dd_all['val_y'] - dd_all['val_z'])/2

        return sum(dd_all['abs'])


def get_PSI_stat(y, z, N=None, category=False, encode = False, groupped_s = 'auto'):
    
    """
    Оценить PSI_stat 
    y, z - векторы для оценки
    N - количество наблюдений в бине
    category - категориальные переменные или нет? категориальные переменные не бьются на куски
    encode - делать LabelEncoder для категориальных переменных или нет? 
    groupped_s - как бинить нумерические переменные. Варианты: 
    'No bin' - отсутствие бининга
    'N number bins' - N означает количество бинов
    'N number obs' - N означает количество наблюдений в одном бине (количество наблюдений делится на N - получается количество бинов)
    'N share unique' - N означает долю от количества уникальных значений переменной. Количество бинов рассчитывается как round(len(np.unique(everything))*N)
    ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] - методы подбора в функции numpy.histogram_bin_edges¶
    Ссылка: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    
    """
    
    if y.dropna().shape[0] == 0 or z.dropna().shape[0] == 0:
        return -10
    else:
    
        if category == True and encode == True:

            everything = pd.concat([y, z])
            encoder = LabelEncoder().fit(everything.fillna('MISSING'))

            if len(encoder.classes_) == 0:
                return -10

            else:

                group_y = encoder.transform(y.fillna('MISSING'))
                group_z = encoder.transform(z.fillna('MISSING'))

        elif category == True and encode == False:

            if y.nunique() == 0 or z.nunique() == 0:
                return -10

            else:

                min_all = min(min(y), min(z))

                group_y = y.fillna(min_all-1)
                group_z = z.fillna(min_all-1)
        else:

            '''Calculate fit metrics for given target and predicted outcomes.'''

            min_all = min(min(y), min(z))
            max_all = max(max(y), max(z))

            if math.isnan(min_all):
                min_all = 0

            if math.isnan(max_all):
                max_all = 0

            y = y.fillna(min_all-1)
            z = z.fillna(min_all-1)

            if max_all - min_all != 0:

                everything = pd.concat([y, z])

                if groupped_s in ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']:

                    try:
                        sel_bins = histogram_bin_edges(everything, bins = groupped_s)
                        keys = sel_bins[:-1]

                        yy = pd.DataFrame()
                        zz = pd.DataFrame()

                        counter_y = histogram(y, sel_bins, density = False)[0]
                        counter_z = histogram(z, sel_bins, density = False)[0]

                        yy['val_y'] = counter_y/sum(counter_y)
                        yy.index = keys
                        zz['val_z'] = counter_z/sum(counter_z)
                        zz.index = keys

                        yy.loc[yy['val_y'] == 0, 'val_y'] = 0.000001
                        zz.loc[zz['val_z'] == 0, 'val_z'] = 0.000001

                        del(sel_bins)
                        gc.collect()

                    except MemoryError:
                        print('MemoryError - hists')
                        group_y = y
                        group_z = z

                        counter_y = collections.Counter(group_y)
                        counter_z = collections.Counter(group_z)

                        yy = pd.DataFrame()
                        yy['val_y'] = counter_y.values()
                        yy['val_y'] = yy['val_y'] /len(y)
                        yy.index = counter_y.keys()
                        #yy = yy.loc[~yy.index.duplicated(keep='first')]

                        zz = pd.DataFrame()
                        zz['val_z'] = counter_z.values()
                        zz['val_z'] = zz['val_z'] /len(z)
                        zz.index = counter_z.keys()
                        #zz = zz.loc[~zz.index.duplicated(keep='first')]

                if groupped_s == 'No bin':

                    #k = (max_all - min_all)/ N
                    group_y = y
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = z
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N number bins':

                    k = (max_all - min_all)/ N

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N number obs':

                    number = round(everything.shape[0]/N)
                    k = (max_all - min_all)/ number

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N share unique' and N >= 0 and N <= 1:
                    number = round(len(np.unique(everything))*N)
                    k = (max_all - min_all)/ number

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N share unique' and (N < 0 or N > 1):
                    raise ValueError('N is a share of number of unique values! It should be in a range [0, 1]')

            else:
                return 0

        if not (groupped_s in ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] and category == False): 

            counter_y = collections.Counter(group_y)
            counter_z = collections.Counter(group_z)

            yy = pd.DataFrame()
            yy['val_y'] = counter_y.values()
            yy['val_y'] = yy['val_y'] /len(y)
            yy.index = counter_y.keys()
            #yy = yy.loc[~yy.index.duplicated(keep='first')]

            zz = pd.DataFrame()
            zz['val_z'] = counter_z.values()
            zz['val_z'] = zz['val_z'] /len(z)
            zz.index = counter_z.keys()
            #zz = zz.loc[~zz.index.duplicated(keep='first')]

        dd_all = pd.concat([yy, zz], axis=1)
        dd_all = dd_all.fillna(0.000001)
        dd_all['PSI'] = (dd_all['val_z'] - dd_all['val_y'])*np.log(dd_all['val_z']/dd_all['val_y'])

        return sum(dd_all['PSI'])


def get_chi_stat(y, z, N=None, category=False, encode = False, groupped_s = 'auto'):
    
    """
    Оценить Chi_stat 
    y, z - векторы для оценки
    N - количество наблюдений в бине
    category - категориальные переменные или нет? категориальные переменные не бьются на куски
    encode - делать LabelEncoder для категориальных переменных или нет? 
    groupped_s - как бинить нумерические переменные. Варианты: 
    'No bin' - отсутствие бининга
    'N number bins' - N означает количество бинов
    'N number obs' - N означает количество наблюдений в одном бине (количество наблюдений делится на N - получается количество бинов)
    'N share unique' - N означает долю от количества уникальных значений переменной. Количество бинов рассчитывается как round(len(np.unique(everything))*N)
    ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] - методы подбора в функции numpy.histogram_bin_edges¶
    Ссылка: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    
    """
    
    if y.dropna().shape[0] == 0 or z.dropna().shape[0] == 0:
        return -10
    else:
    
        if category == True and encode == True:

            everything = pd.concat([y, z])
            encoder = LabelEncoder().fit(everything.fillna('MISSING'))

            if len(encoder.classes_) == 0:
                return -10

            else:

                group_y = encoder.transform(y.fillna('MISSING'))
                group_z = encoder.transform(z.fillna('MISSING'))

        elif category == True and encode == False:

            if y.nunique() == 0 or z.nunique() == 0:
                return -10

            else:

                min_all = min(min(y), min(z))

                group_y = y.fillna(min_all-1)
                group_z = z.fillna(min_all-1)
        else:

            '''Calculate fit metrics for given target and predicted outcomes.'''

            min_all = min(min(y), min(z))
            max_all = max(max(y), max(z))

            y = y.fillna(min_all-1)
            z = z.fillna(min_all-1)

            if max_all - min_all != 0:

                everything = pd.concat([y, z])

                if groupped_s in ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']:

                    try:
                        sel_bins = histogram_bin_edges(everything, bins = groupped_s)
                        keys = sel_bins[:-1]

                        yy = pd.DataFrame()
                        zz = pd.DataFrame()

                        counter_y = histogram(y, sel_bins, density = False)[0]
                        counter_z = histogram(z, sel_bins, density = False)[0]

                        yy['val_y'] = counter_y/sum(counter_y)
                        yy.index = keys
                        zz['val_z'] = counter_z/sum(counter_z)
                        zz.index = keys

                        del(sel_bins)
                        gc.collect()

                    except MemoryError:
                        print('MemoryError - hists')
                        group_y = y
                        group_z = z

                        counter_y = collections.Counter(group_y)
                        counter_z = collections.Counter(group_z)

                        yy = pd.DataFrame()
                        yy['val_y'] = counter_y.values()
                        yy['val_y'] = yy['val_y'] /len(y)
                        yy.index = counter_y.keys()
                        #yy = yy.loc[~yy.index.duplicated(keep='first')]

                        zz = pd.DataFrame()
                        zz['val_z'] = counter_z.values()
                        zz['val_z'] = zz['val_z'] /len(z)
                        zz.index = counter_z.keys()
                        #zz = zz.loc[~zz.index.duplicated(keep='first')]
                
				# Исправлено
                if groupped_s == 'No bin':

                    #k = (max_all - min_all)/ N
                    group_y = y
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = z
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N number bins':

                    k = (max_all - min_all)/ N

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N number obs':

                    number = round(everything.shape[0]/N)
                    k = (max_all - min_all)/ number

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N share unique' and N >= 0 and N <= 1:
                    number = round(len(np.unique(everything))*N)
                    k = (max_all - min_all)/ number

                    group_y = [np.trunc((x - min_all)/k) for x in y]
                    #group_val_y = [np.trunc((x - min_all)/k)*k + min_all for x in y]

                    group_z = [np.trunc((x - min_all)/k) for x in z]
                    #group_val_z = [np.trunc((x - min_all)/k)*k + min_all for x in z]

                if groupped_s == 'N share unique' and (N < 0 or N > 1):
                    raise ValueError('N is a share of number of unique values! It should be in a range [0, 1]')

            else:
                return 0

        if not (groupped_s in ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] and category == False): 

            counter_y = collections.Counter(group_y)
            counter_z = collections.Counter(group_z)

            yy = pd.DataFrame()
            yy['val_y'] = counter_y.values()
            yy['val_y'] = yy['val_y'] /len(y)
            yy.index = counter_y.keys()
            #yy = yy.loc[~yy.index.duplicated(keep='first')]

            zz = pd.DataFrame()
            zz['val_z'] = counter_z.values()
            zz['val_z'] = zz['val_z'] /len(z)
            zz.index = counter_z.keys()
            #zz = zz.loc[~zz.index.duplicated(keep='first')]

        dd_all = pd.concat([yy, zz], axis=1)
        dd_all = dd_all.fillna(0)
		# Исправлено
        dd_all['chi'] = (dd_all['val_y'] - dd_all['val_z'])**2/(z.shape[0]+y.shape[0])
        chi_squared_stat = sum(dd_all['chi'])


        p_value = 1-stats.chi2.cdf(x=chi_squared_stat, df = N-1)

        return chi_squared_stat, p_value


def get_stats_by_month(data, columns, time, cut, N, dropna='min', category_list = None, encode = False, 
                       groupped_s = 'auto', n_jobs=2, verbose=20):
    
    """
    Посчитать статистики стабильности по месяцам - (S-stat, Ks-stat, PSI-stat). Данные не агрегируются.
    data - данные
    columns - переменные
    time - переменная, которая отвечает за время
    cut - обрезать ли переменные по 99 перцентилю
    N - количество наблюдений в бине (для S и PSI статистик)
    dropna - метод импутации пропусков:
        min - минимальное значение на весь период. data[i].min()-1
        max - максимальное значение на весь период. data[i].max()+1
        какое-то значение
    category_list - список категориальных переменных
    encode - делать ли encoding для категориальных переменных перед расчетом статистик (в методах S и PSI они False)
    groupped_s - как бинить нумерические переменные. Варианты: 
    'No bin' - отсутствие бининга
    'N number bins' - N означает количество бинов
    'N number obs' - N означает количество наблюдений в одном бине (количество наблюдений делится на N - получается количество бинов)
    'N share unique' - N означает долю от количества уникальных значений переменной. Количество бинов рассчитывается как round(len(np.unique(everything))*N)
    ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] - методы подбора в функции numpy.histogram_bin_edges¶
    Ссылка: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    
    """
    
    unique = np.sort(data[time].unique())
    time_len = len(unique) - 1
    Statistics = []
    # k_test = 0
    def col_preproc(o):
    # for o in range(len(columns)):
        # nonlocal k_test
        nonlocal Statistics
        # k_test = k_test+1
        # if k_test ==1:
        #     tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
        #     print ('Number of finished repetitions:', k_test , '| time: ' , tm)
            
        # if k_test % 10 ==0:
        #     tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
        #     print ('Number of finished repetitions:', k_test , '| time: ' , tm)
        
        for i in range(len(unique)):
            
            if i == time_len:
                break
            else:
                
                if data[columns[o]].dtype == 'object' or (type(category_list) != type(None) and columns[o] in category_list and encode == True):
                    
                    x = data.loc[data[time] == unique[i], columns[o]]
                    y = data.loc[data[time] == unique[i+1], columns[o]]
                    
                    everything = pd.concat([x, y])
                    encoder = LabelEncoder().fit(everything.fillna('MISSING'))                    
                    
                    x_label = pd.Series(encoder.transform(x.fillna('MISSING')))
                    y_label = pd.Series(encoder.transform(y.fillna('MISSING')))
                    
                    KS = stats.ks_2samp(x_label, y_label)
                    
                    #mis_i = data.loc[data[time] == unique[i], 
                                     #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i], columns[o]])
                    #mis_j = data.loc[data[time] == unique[i+1], 
                                     #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i+1], columns[o]])
                    #delta = mis_j-mis_i
                    
                    Statistics.append([columns[o], unique[i], unique[i+1], #delta, 
                                       get_s_stat(x_label, y_label, N, category = True, encode = False, groupped_s = groupped_s)*100, 
									   # Исправлено
                                       get_PSI_stat(x_label, y_label, N, category = True, encode = False, groupped_s = groupped_s)*100,
                                       KS.statistic*100])
                    gc.collect()
                    
                elif type(category_list) != type(None) and columns[o] in category_list and encode == False:
                    
                    x = data.loc[data[time] == unique[i], columns[o]]
                    y = data.loc[data[time] == unique[i+1], columns[o]]
                    
                    everything = pd.concat([x, y])
                               
                    x_label = x.fillna(everything.min()-1)
                    y_label = y.fillna(everything.min()-1)
                    
                    KS = stats.ks_2samp(x_label, y_label)
                    
                    #mis_i = data.loc[data[time] == unique[i], 
                                     #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i], columns[o]])
                    #mis_j = data.loc[data[time] == unique[i+1], 
                                     #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i+1], columns[o]])
                    #delta = mis_j-mis_i
                    
                    Statistics.append([columns[o], unique[i], unique[i+1], #delta, 
                                       get_s_stat(x_label, y_label, N, category = True, encode = False, groupped_s = groupped_s)*100, 
                                       get_PSI_stat(x_label, y_label, N, category = True, encode = False, groupped_s = groupped_s)*100,
                                       KS.statistic*100])
                    gc.collect()
                    
                    
                else:
                
                    if dropna == 'min':

                        if cut == True:
                            
                            #minimum = data[columns[o]].min() - 1

                            x = data.loc[data[time] == unique[i], columns[o]]
                            y = data.loc[data[time] == unique[i+1], columns[o]]
                            
                            everything = pd.concat([x, y])
                            minimum = everything.min()-1
                            
                            perc_x = np.nanpercentile(x, 99)
                            perc_y = np.nanpercentile(y, 99)
                            
                            x = x.fillna(minimum)
                            x.loc[x > perc_x] = perc_x
                            y = y.fillna(minimum)
                            y.loc[y > perc_y] = perc_y
                            
                            
                        elif cut == False:
                            
                            #minimum = data[columns[o]].min() - 1
                                
                            x = data.loc[data[time] == unique[i], columns[o]]
                            y = data.loc[data[time] == unique[i+1], columns[o]]
                            
                            everything = pd.concat([x, y])
                            minimum = everything.min()-1
                            
                            x = x.fillna(minimum)
                            y = y.fillna(minimum)

                    elif dropna == 'max':

                        if cut == True:
                            
                            #maximum = data[columns[o]].max() + 1

                            x = data.loc[data[time] == unique[i], columns[o]]
                            y = data.loc[data[time] == unique[i+1], columns[o]]
                            
                            everything = pd.concat([x, y])
                            maximum = everything.max()+1
                            
                            perc_x = np.nanpercentile(x, 99)
                            perc_y = np.nanpercentile(y, 99)
                            
                            x = x.fillna(maximum)
                            x.loc[x > perc_x] = perc_x
                            y = y.fillna(maximum)
                            y.loc[y > perc_y] = perc_y

                        elif cut == False:
                            
                            #maximum = data[columns[o]].max() + 1
                            
                            x = data.loc[data[time] == unique[i], columns[o]]
                            y = data.loc[data[time] == unique[i+1], columns[o]]
                            
                            everything = pd.concat([x, y])
                            maximum = everything.max()+1
                            
                            x = x.fillna(maximum)
                            y = y.fillna(maximum)
                        
                    else:

                        if cut == True:

                            x = data.loc[data[time] == unique[i], columns[o]]
                            y = data.loc[data[time] == unique[i+1], columns[o]]
                            
                            perc_x = np.nanpercentile(x, 99)
                            perc_y = np.nanpercentile(y, 99)
                            
                            x = x.fillna(dropna)
                            x.loc[x > perc_x] = perc_x
                            y = y.fillna(dropna)
                            y.loc[y > perc_y] = perc_y
                            

                        elif cut == False:
                            x = data.loc[data[time] == unique[i], columns[o]]
                            x = x.fillna(dropna)
                            y = data.loc[data[time] == unique[i+1], columns[o]]
                            y = y.fillna(dropna)
                        
                    KS = stats.ks_2samp(x, y)
                    
                    #mis_i = data.loc[data[time] == unique[i], 
                                     #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i], columns[o]])
                    #mis_j = data.loc[data[time] == unique[i+1], 
                                     #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i+1], columns[o]])
                    #delta = mis_j-mis_i
                    
                    Statistics.append([columns[o], unique[i], unique[i+1], #delta, 
                                       get_s_stat(x, y, N, category = False, encode = False, groupped_s = groupped_s)*100, 
                                       get_PSI_stat(x, y, N, category = False, encode = False, groupped_s = groupped_s)*100,
                                       KS.statistic*100])
                    gc.collect()
                   
    parallel = Parallel(n_jobs=n_jobs, require='sharedmem', verbose = verbose)
    # dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    # print ('Now', dd1)
    with parallel:
        par_res = parallel((delayed(col_preproc)(o) for o in tqdm(range(len(columns)), 
                                                                    total=len(columns), 
                                                                    desc = 'Get_stats_by_month')))


    # dd2 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    # print ('Now', dd2)

    
    # print('Time spent in hours:', (datetime.strptime(dd2,"%d.%m.%Y %H:%M:%S")-datetime.strptime(dd1,"%d.%m.%Y %H:%M:%S")))
         
    
    labels = ['variable', 'i', 'j', #'Delta Share of missing', 
              'S-stat', 'psi', 
              'KS_statistic']
    definition = pd.DataFrame.from_records(Statistics, columns = labels)
    
    return definition


# In[ ]:


def get_stats(data,
              columns,
              time,
              cut,
              N,
              dropna='min',
              category_list=None,
              encode=False,
              groupped_s='auto',
              group=True,
              n_jobs=2,
              verbose=20):
    
    """
    Посчитать статистики стабильности по укрупненным временным периодам (получается максимальное значение статистик
    по всем значениям периода) (S-stat, Ks-stat, PSI-stat)
    data - данные
    columns - переменные
    time - переменная, которая отвечает за время
    cut - обрезать ли переменные по 99 перцентилю
    N - количество наблюдений в бине (для S и PSI статистик)
    dropna - метод импутации пропусков:
        min - минимальное значение на весь период. data[i].min()-1
        max - максимальное значение на весь период. data[i].max()+1
        какое-то значение
    category_list - список категориальных переменных
    encode - делать ли encoding для категориальных переменных перед расчетом статистик (в методах S и PSI они False)
    groupped_s - как бинить нумерические переменные. Варианты: 
    'No bin' - отсутствие бининга
    'N number bins' - N означает количество бинов
    'N number obs' - N означает количество наблюдений в одном бине (количество наблюдений делится на N - получается количество бинов)
    'N share unique' - N означает долю от количества уникальных значений переменной. Количество бинов рассчитывается как round(len(np.unique(everything))*N)
    ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] - методы подбора в функции numpy.histogram_bin_edges¶
    Ссылка: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    
    """
    
    Statistics = []
    unique = data[time].unique()
    # k_test = 0
    def col_preproc(o):
        #     for o in range(len(columns)):
        # nonlocal k_test
        nonlocal Statistics
        # k_test = k_test+1
        # if k_test ==1:
        #     tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
        #     print ('Number of finished repetitions:', k_test , '| time: ' , tm)
            
        # if k_test % 10 ==0:
        #     tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
        #     print ('Number of finished repetitions:', k_test , '| time: ' , tm)
            
        for i in range(len(unique)):
            for j in range(len(unique)):   
                if i > j:

                    if data[columns[o]].dtype == 'object' or (type(category_list) != type(None) and columns[o] in category_list and encode == True):

                        x = data.loc[data[time] == unique[i], columns[o]]
                        y = data.loc[data[time] == unique[j], columns[o]]

                        everything = pd.concat([x, y])
                        encoder = LabelEncoder().fit(everything.fillna('MISSING'))                    

                        x_label = pd.Series(encoder.transform(x.fillna('MISSING')))
                        y_label = pd.Series(encoder.transform(y.fillna('MISSING')))

                        KS = stats.ks_2samp(x_label, y_label)

                        #mis_i = data.loc[data[time] == unique[i], 
                                         #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i], columns[o]])
                        #mis_j = data.loc[data[time] == unique[j], 
                                         #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[j], columns[o]])
                        #delta = mis_j-mis_i

                        Statistics.append([columns[o], unique[i], unique[j], #delta, 
                                          get_s_stat(x_label, y_label, N, category = True, encode = False, groupped_s = groupped_s)*100, 
                                          get_PSI_stat(x_label, y_label, N, category = True, encode = False, groupped_s = groupped_s)*100, 
                                          KS.statistic*100])
                        
                    elif type(category_list) != type(None) and columns[o] in category_list and encode == False:
                    
                        x = data.loc[data[time] == unique[i], columns[o]]
                        y = data.loc[data[time] == unique[j], columns[o]]

                        everything = pd.concat([x, y])

                        x_label = x.fillna(everything.min()-1)
                        y_label = y.fillna(everything.min()-1)

                        KS = stats.ks_2samp(x_label, y_label)

                        #mis_i = data.loc[data[time] == unique[i], 
                                         #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i], columns[o]])
                        #mis_j = data.loc[data[time] == unique[i+1], 
                                         #columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i+1], columns[o]])
                        #delta = mis_j-mis_i

                        Statistics.append([columns[o], unique[i], unique[j], #delta, 
                                          get_s_stat(x_label, y_label, N, category = True, encode = False, groupped_s = groupped_s)*100, 
                                          get_PSI_stat(x_label, y_label, N, category = True, encode = False, groupped_s = groupped_s)*100, 
                                          KS.statistic*100])
                    

                    else:

                        if dropna == 'min':

                            if cut == True:

                                x = data.loc[data[time] == unique[i], columns[o]]
                                y = data.loc[data[time] == unique[j], columns[o]]

                                everything = pd.concat([x, y])
                                minimum = everything.min()-1

                                perc_x = np.nanpercentile(x, 99)
                                perc_y = np.nanpercentile(y, 99)

                                x = x.fillna(minimum)
                                x.loc[x > perc_x] = perc_x
                                y = y.fillna(minimum)
                                y.loc[y > perc_y] = perc_y

                            elif cut == False:

                                #minimum = data[columns[o]].min() - 1

                                x = data.loc[data[time] == unique[i], columns[o]]
                                y = data.loc[data[time] == unique[j], columns[o]]

                                everything = pd.concat([x, y])
                                minimum = everything.min()-1

                                x = x.fillna(minimum)
                                y = y.fillna(minimum)

                        elif dropna == 'max':

                            if cut == True:

                                #maximum = data[columns[o]].max() + 1

                                x = data.loc[data[time] == unique[i], columns[o]]
                                y = data.loc[data[time] == unique[j], columns[o]]

                                everything = pd.concat([x, y])
                                maximum = everything.max()+1

                                perc_x = np.nanpercentile(x, 99)
                                perc_y = np.nanpercentile(y, 99)

                                x = x.fillna(maximum)
                                x.loc[x > perc_x] = perc_x
                                y = y.fillna(maximum)
                                y.loc[y > perc_y] = perc_y

                            elif cut == False:

                                #maximum = data[columns[o]].max() + 1
                            
                                x = data.loc[data[time] == unique[i], columns[o]]
                                y = data.loc[data[time] == unique[j], columns[o]]

                                everything = pd.concat([x, y])
                                maximum = everything.max()+1

                                x = x.fillna(maximum)
                                y = y.fillna(maximum)

                        else:

                            if cut == True:

                                x = data.loc[data[time] == unique[i], columns[o]]
                                y = data.loc[data[time] == unique[j], columns[o]]

                                perc_x = np.nanpercentile(x, 99)
                                perc_y = np.nanpercentile(y, 99)

                                x = x.fillna(dropna)
                                x.loc[x > perc_x] = perc_x
                                y = y.fillna(dropna)
                                y.loc[y > perc_y] = perc_y

                            elif cut == False:
                                x = data.loc[data[time] == unique[i], columns[o]]
                                x = x.fillna(dropna)
                                y = data.loc[data[time] == unique[j], columns[o]]
                                y = y.fillna(dropna)

                        KS = stats.ks_2samp(x, y)

                        #mis_i = data.loc[data[time] == unique[i], 
                          #               columns[o]].isnull().sum()/len(data.loc[data[time] == unique[i], columns[o]])
                        #mis_j = data.loc[data[time] == unique[j], 
                         #                columns[o]].isnull().sum()/len(data.loc[data[time] == unique[j], columns[o]])
                        #delta = mis_j-mis_i

                        Statistics.append([columns[o], unique[i], unique[j], 
                                           #delta, 
                                           get_s_stat(x, y, N, category = False, encode = False, groupped_s = groupped_s)*100, 
                                           get_PSI_stat(x, y, N, category = False, encode = False, groupped_s = groupped_s)*100, 
                                           KS.statistic*100])
                        
    parallel = Parallel(n_jobs=n_jobs, require='sharedmem', verbose = verbose)
    # print('data shape =', data_check.shape)
    # dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    # print ('Now', dd1)
    # t1= time.time()
    with parallel:
        par_res = parallel((delayed(col_preproc)(o) for o in tqdm(range(len(columns)), 
                                                                    total=len(columns), 
                                                                    desc = 'Get_stats')))

    # dd2 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    # print ('Now', dd2)
    # t2 = time.time()
    
    # print('Time spent in hours:', (datetime.strptime(dd2,"%d.%m.%Y %H:%M:%S")-datetime.strptime(dd1,"%d.%m.%Y %H:%M:%S")))
         
    
    labels = ['variable', 'i', 'j', #'Delta Share of missing', 
              'S-stat', 'psi', 
              'KS_statistic']
    definition = pd.DataFrame.from_records(Statistics, columns = labels)
                
    res = definition[definition['i'] != definition['j']]
    
    if group == True:
        res_max = res.groupby('variable').aggregate({'S-stat': 'max',
                                                 'KS_statistic': 'max',
                                                 'psi': 'max'})

        res_max = res_max.reset_index()

        return res_max
    else:
        return res



def stable_unstable(data, light_decline=15, hard_decline=25, light_treshold=1, hard_treshold=2):
    
    """
    рассчитывает флаги стабильности на матрице со значениями s-stat, PSI, KS 
    (работает на результататх функции get_stats/ get_stats_by_month. Если stable_unstable применяется к get_stats, 
    дополнительная агрегация не нужна, в противном случае - stable_unstable_by_month_divide
         light_decline/hard_decline - пороги для каждого критерия стабильности
         light_treshold / hard_treshold - пороги для суммарной стабильности, рассчитанной как сумма по 
         3 критериям стабильности
    
    """
    
    data.loc[(data['S-stat'] >= light_decline) & (data['S-stat'] < hard_decline), 'S-stat score'] = 0.5 
    data.loc[(data['S-stat'] >= hard_decline), 'S-stat score'] = 1 
    data.loc[(data['S-stat'] < light_decline) & (data['S-stat'] >= 0), 'S-stat score'] = 0 
    data.loc[(data['S-stat'] < 0), 'S-stat score'] = 2 
    
    data.loc[(data['KS_statistic'] >= light_decline) & (data['KS_statistic'] < hard_decline), 'KS score'] = 0.5 
    data.loc[(data['KS_statistic'] >= hard_decline), 'KS score'] = 1 
    data.loc[(data['KS_statistic'] < light_decline) & (data['KS_statistic'] >= 0), 'KS score'] = 0 
    data.loc[(data['KS_statistic'] < 0), 'KS score'] = 2
    
    data.loc[(data['psi'] >= light_decline) & (data['psi'] < hard_decline), 'PSI score'] = 0.5 
    data.loc[(data['psi'] >= hard_decline), 'PSI score'] = 1 
    data.loc[(data['psi'] < light_decline) & (data['psi'] >= 0), 'PSI score'] = 0 
    data.loc[(data['psi'] < 0), 'PSI score'] = 2
    
    data['Sum'] = data['S-stat score'] + data['KS score'] + data['PSI score']
    
    data.loc[(data['Sum'] < light_treshold), 'Result'] = 'Stable'
    data.loc[(data['Sum'] < light_treshold), 'Result_num'] = 0
    data.loc[(data['Sum'] >= light_treshold) & (data['Sum'] < hard_treshold), 'Result'] = 'Light instability'
    data.loc[(data['Sum'] >= light_treshold) & (data['Sum'] < hard_treshold), 'Result_num'] = 0.5
    data.loc[(data['Sum'] >= hard_treshold), 'Result'] = 'Unstable'
    data.loc[(data['Sum'] >= hard_treshold), 'Result_num'] = 1
    return data


def stable_unstable_by_month_divide(data, light_treshold=1, hard_treshold=1.5):
    
    """
    stable_unstable_by_month_divide(data, light_treshold=1, hard_treshold=1.5) -
    агрегирует результаты stable_unstable (после применения к get_stats_by_month)
    
    """
    
    grouped_res = data.groupby('variable').aggregate({'i': 'count','Sum': 'sum'})
    grouped_res = grouped_res.reset_index()
    grouped_res['Divide'] = grouped_res['Sum']/grouped_res['i']    
    
    
    grouped_res.loc[(grouped_res['Divide'] < light_treshold), 'Result'] = 'Stable'
    grouped_res.loc[(grouped_res['Divide'] < light_treshold), 'Result_num'] = 0
    
    grouped_res.loc[(grouped_res['Divide'] >= light_treshold) & 
                    (grouped_res['Divide'] < hard_treshold), 'Result'] = 'Light instability'
    grouped_res.loc[(grouped_res['Divide'] >= light_treshold) & 
                    (grouped_res['Divide'] < hard_treshold), 'Result_num'] = 0.5
    
    grouped_res.loc[(grouped_res['Divide'] >= hard_treshold), 'Result'] = 'Unstable'
    grouped_res.loc[(grouped_res['Divide'] >= hard_treshold), 'Result_num'] = 1
    
    return grouped_res


# In[ ]:


def union_datas(datas, weights, light_treshold=0.2, hard_treshold=0.6, strategy = 'max'):
    
    """
    объединяет проверки стабильности разными методами 
    (объединяет результаты get_stats_by_month и get_stats после агрегации результатов 
    stable_unstable и stable_unstable_by_month_divide)
    strategy - mean или max. По умолчанию max
    
    """
    
    if len(datas) != len(weights):
        return 'Len of data list and weight list must be equal!'
    
    else:
    
        if strategy == 'mean':
            
            lists = range(len(datas))

            data_check = pd.DataFrame(datas[0]['variable'])

            for data, w, n in zip(datas, weights, lists):
                data['Res_num_w'] = data['Result_num']*w
                data = data[['variable', 'Res_num_w']]

                data_check = pd.merge(data_check, data, on = 'variable', suffixes = ["", n])

            cols = [i for i in data_check.columns if i != 'variable']

            data_check['Sum'] = 0

            for i in cols:
                data_check['Sum'] += data_check[i]
                
        elif strategy == 'max':
            
            lists = range(len(datas))

            data_check = pd.DataFrame(datas[0]['variable'])

            for data, w, n in zip(datas, weights, lists):
                data['Res_num_w'] = data['Result_num']
                data = data[['variable', 'Res_num_w']]

                data_check = pd.merge(data_check, data, on = 'variable', suffixes = ["", n])

            cols = [i for i in data_check.columns if i != 'variable']

            data_check['Sum'] = 0
            for i in cols:
                data_check.loc[data_check['Sum'] <= data_check[i], 'Sum'] = data_check[i]
                data_check.loc[data_check['Sum'] > data_check[i], 'Sum'] = data_check['Sum']
            
        data_check.loc[(data_check['Sum'] < light_treshold), 'Result'] = 'Stable'
        data_check.loc[(data_check['Sum'] < light_treshold), 'Result_num'] = 0
        
        data_check.loc[(data_check['Sum'] >= light_treshold) & 
                        (data_check['Sum'] < hard_treshold), 'Result'] = 'Light instability'
        data_check.loc[(data_check['Sum'] >= light_treshold) & 
                        (data_check['Sum'] < hard_treshold), 'Result_num'] = 0.5
        
        data_check.loc[(data_check['Sum'] >= hard_treshold), 'Result'] = 'Unstable'
        data_check.loc[(data_check['Sum'] >= hard_treshold), 'Result_num'] = 1
        
        data_check = data_check.reset_index()
        
        grouped_res= data_check[['variable', 'Result', 'Result_num']]
        
        return grouped_res



# In[ ]:


def statistics_with_target(data, columns, target, category_list, category_target = True):
    
    """
    Считает статистики связи с таргетом
    рассчитывает p-value тестов взаимосвязи переменных из columns с переменной target. 
    
    data - данные
    columns - колонки, для которых хотим получить статистику
    category_list - список категориальных переменных
    category_target - флаг True/False (отмечает, является ли таргет категориальным. Бинарный таргет входит в категориальные!). Исходя из 
    этого флага рассчитываются статистики.
    
        Если таргет бинарный:
        Для непрерывных переменных - t-статистика (для нормально распределенных) или тест Манна_Уитни. 
        Для категориальных переменных - хи статистика, для бинарных переменных - хи-статистика(с учетом миссингов) и 
        точная статистика Фишера (если перед подачей заменить в них миссинги на 0). 

        Если таргет непрерывный:
        Для непрерывных переменных - коэффициент корреляции Пирсона, для которого считается его p-value
        Для бинарных категорий - t-статистика (для нормально распределенного таргета) или тест Манна_Уитни (для не ненормально
        распределенного таргета)
        Для категориальных переменных (более 2 категорий) - дисперсионный анализ ANOVA

        Если таргет категориальный (более 2 категорий):
        Для непрерывных переменных - дисперсионный анализ ANOVA
        Для бинарных переменных - хи-статистика
        Для категориальных переменных - хи-статистика
    
    Помимо этого, функция считает статистики корреляции с таргетом. 
    
        Если таргет бинарный:
        Для непрерывных переменных - корреляция Пирсона 
        Для категориальных переменных - фи-статистика 
        np.sqrt(chi2_contingency(table)[0]/(chi2_contingency(table)[0]+x.shape[0]))

        Если таргет непрерывный:
        Для непрерывных переменных - корреляция Пирсона
        Для бинарных переменных - корреляция Пирсона
        Для категориальных переменных - коэффициент F статистики, по которой считается значение pvalue для ANOVA.

        Если таргет категориальный (более 2 категорий):
        Для непрерывных переменных - коэффициент F статистики, по которой считается значение pvalue для ANOVA
        в такой ситуации
        Для категориальных и бинарных переменных - фи-статистика
    
    !!ВАЖНО!! По своему построению значения фи статистики отличаются от классической 
    корреляции! Максимально возможные значения фи статистики - 0.707,
    это следует учесть при сравнении корреляций непрерывных и категориальных переменных! 
    Помимо этого следует учесть, что значения фи статистики не могут быть отрицательные!
    
    !!ВАЖНО!! Для пары категориальная (более 2 категорий) - непрерывная переменная, если переменная не является порядковой, нельзя 
    рассчитать коэффициент корреляции или связи. Это поле заполняется статистикой F, для которых и считается pvalue теста ANOVA. 
    Этот коэффициент НЕЛЬЗЯ сравнивать с коэффициентами, так как он принципиально другой по своему построению. Более того, коэффициент
    зависит от количества категорий категориальной переменной и от объема наблюдений в каждой. 
    
    !!ВАЖНО!! Коэффициент F ANOVA рассчитывается только для тех категорий, количество наблюдений в которых более 5. В противном случае 
    ломается F распределение. Так как учет редких категорий бинов делается на этапе make_standart, здесь эти очень редкие категории 
    исключаются из анализа.
    
    """
    
    if category_target == True:
        if type(target) == str:
            unique_target_values = data[target].unique()
        else:
            unique_target_values = target.unique()
    
    statistics = []
    
    k = 0
    
    if type(target) == str:
        ys = data[target]
    else:
        ys = target
    
    if category_target ==  True and len(unique_target_values) == 2:
        y = ys.fillna(0)
    else:
        y = ys.fillna(ys.min()-1)
        
    #"""Бинарный таргет!"""
    if category_target ==  True and len(unique_target_values) == 2:
        for i in tqdm(columns, desc='Statistics_with_target', total=len(columns)):
            # k +=1
            # if k % 30 ==0:
            #     tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            #     print ('Number of finished repetitions:', k , '| time: ' , tm)

            #Категориальные переменные
            if data[i].dtype == 'object' or i in category_list: 
                
                if data[i].dtype == 'object':

                    x = data[i].astype('str')
                    x = x.fillna('MISSING')
                    encoder = LabelEncoder().fit(x)
                    x = encoder.transform(x)

                elif i in category_list:
                    x = data[i].copy()
                    x = x.fillna(x.min()-1)

                if len(np.unique(x)) > 1:
                    table = pd.crosstab(x, y)
                    statistics.append([i, chi2_contingency(table)[1], 
                                np.sqrt(chi2_contingency(table)[0]/(chi2_contingency(table)[0]+x.shape[0])),
                                          'phi'])
                else:
                    statistics.append([i, -10, -10, 'One value'])
                
            #Непрерывные переменные
            else:
                minimum = data[i].min()-1
                xs = data[i].copy() 
                x = xs.fillna(minimum)
                
                #"""Непрерывные переменные - возможные бины"""
                if len(data[i].unique()) == 2:
                    table = pd.crosstab(x, y)
                    statistics.append([i, chi2_contingency(table)[1], 
                            np.sqrt(chi2_contingency(table)[0]/(chi2_contingency(table)[0]+x.shape[0])),
                                          'phi'])        
                    
                #"""Непрерывные переменные - истинно непрерывные"""
                else:
                    a = x[y == unique_target_values[0]]
                    b = x[y == unique_target_values[1]]
                    
                    if len(a.unique()) > 1 and len(b.unique()) > 1:
                        x_normal = normaltest(a, nan_policy = 'omit').pvalue
                        y_normal = normaltest(b, nan_policy = 'omit').pvalue
                        x_ks = kstest(a, 'norm').pvalue
                        y_ks = kstest(b, 'norm').pvalue
                        
                        if x_normal < 0.01 or y_normal < 0.01 or x_ks < 0.01 or y_ks < 0.01:
                            statistics.append([i, mannwhitneyu(a, b).pvalue, xs.corr(ys), 'corr'])
                        else:
                            statistics.append([i, ttest_ind(a, b, nan_policy = 'omit').pvalue, 
                                                   xs.corr(ys), 'corr'])
                    else:
                        statistics.append([i, -10, -10, 'One value'])
                
    #"""Категориальный таргет!"""
    
    elif category_target ==  True and len(unique_target_values) > 2:
        for i in tqdm(columns, desc='Statistics_with_target', total=len(columns)):
            # k +=1
            # if k % 30 ==0:
            #     tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            #     print ('Number of finished repetitions:', k , '| time: ' , tm)
                
            #"""Категориальные переменные"""
            if data[i].dtype == 'object' or i in category_list: 
                
                if data[i].dtype == 'object':

                    x = data[i].astype('str')
                    x = x.fillna('MISSING')
                    encoder = LabelEncoder().fit(x)
                    x = encoder.transform(x)

                elif i in category_list:
                    x = data[i].copy()
                    x = x.fillna(x.min()-1)

                if len(x.unique()) > 1:
                   
                    table = pd.crosstab(x, y)
                    statistics.append([i, chi2_contingency(table)[1], 
                                np.sqrt(chi2_contingency(table)[0]/(chi2_contingency(table)[0]+x.shape[0])),
                                          'phi'])
                else:
                    statistics.append([i, -10, -10, 'One value'])

                #"""Непрерывные переменные - возможные бины"""
            else:
                minimum = data[i].min()-1
                xs = data[i].copy() 
                x = xs.fillna(minimum)
                    
                if len(data[i].unique()) == 2:
                
                    table = pd.crosstab(x, y)
                    statistics.append([i, chi2_contingency(table)[1], 
                                    np.sqrt(chi2_contingency(table)[0]/(chi2_contingency(table)[0]+x.shape[0])),
                                                  'phi'])      
                else:
                
                #"""Непрерывные переменные - истинно непрерывные"""
                
                    samples_for_anova = []
                    check_for_normality_ks = []

                    for value in unique_target_values:
                        if x[y == value].count() > 5:
                            samples_for_anova.append(x[y == value])
                            check_for_normality_ks.append(kstest(x[y == value], 'norm').pvalue)
                            
                    if len(samples_for_anova) > 1:
                        if max(check_for_normality_ks) < 0.01:
                            statistics.append([i, kruskal(*samples_for_anova).pvalue, kruskal(*samples_for_anova).statistic, 
                                               'ANOVA'])  

                        else:
                            statistics.append([i, f_oneway(*samples_for_anova).pvalue, f_oneway(*samples_for_anova).statistic, 
                                               'ANOVA'])

                    else:
                        statistics.append([i, np.nan, np.nan, 'One big category, others less than 5 obs'])
                        
            
    #"""Непрерывный таргет!"""
    
    elif category_target == False:
        for i in tqdm(columns, desc='Statistics_with_target', total=len(columns)):
            # k +=1
            # if k % 30 ==0:
            #     tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            #     print ('Number of finished repetitions:', k , '| time: ' , tm)
                
            #Категориальные переменные
            
            if data[i].dtype == 'object' or i in category_list: 
                
                if data[i].dtype == 'object':

                    x = data[i].astype('str')
                    x = x.fillna('MISSING')
                    encoder = LabelEncoder().fit(x)
                    x = encoder.transform(x)

                elif i in category_list:
                    xs = data[i].copy()
                    if sorted(xs.unique())[0] == 0 and sorted(xs.unique())[1] == 1:
                        x = xs.fillna(0)
                    else:
                        x = xs.fillna(xs.min()-1)

                #Бинарные переменные в списке категориальных
                if len(x.unique()) == 2:
                    a = y[x==x.unique()[0]]
                    b = y[x==x.unique()[1]]
                    
                    if len(a.unique()) > 1 and len(b.unique()) > 1:
                        x_normal = normaltest(a, nan_policy = 'omit').pvalue
                        y_normal = normaltest(b, nan_policy = 'omit').pvalue
                        x_ks = kstest(a, 'norm').pvalue
                        y_ks = kstest(b, 'norm').pvalue
                        
                        if x_normal < 0.01 or y_normal < 0.01 or x_ks < 0.01 or y_ks < 0.01:
                            statistics.append([i, mannwhitneyu(a, b).pvalue, x.corr(ys), 'corr'])
                        else:
                            statistics.append([i, ttest_ind(a, b, nan_policy = 'omit').pvalue, 
                                                   x.corr(ys), 'corr'])
                    else:
                        statistics.append([i, -10, -10, 'One value'])
                            
                #Истинно категориальные переменные
                
                else: 
                    samples_for_anova = []
                    check_for_normality = []
                    check_for_normality_ks = []
                    unique_feature_values = x.unique()

                    for value in unique_feature_values:
                        if y[x == value].count() > 5:
                            samples_for_anova.append(y[x == value])
                            check_for_normality_ks.append(kstest(y[x == value], 'norm').pvalue)

                    if len(samples_for_anova) > 1:
                        if max(check_for_normality_ks) < 0.01:
                            statistics.append([i, kruskal(*samples_for_anova).pvalue, kruskal(*samples_for_anova).statistic,
                                               'Was ANOVA'])  

                        else:
                            statistics.append([i, f_oneway(*samples_for_anova).pvalue, f_oneway(*samples_for_anova).statistic, 
                                               'Was ANOVA'])
                            
                    else:
                        statistics.append([i, np.nan, np.nan, 'One big category, others less than 5 obs'])
                        
            #Непрерывные переменные
            else:
                xs = data[i].copy()
                if sorted(xs.unique())[0] == 0 and sorted(xs.unique())[1] == 1:
                    x = xs.fillna(0)
                else:
                    x = xs.fillna(xs.min()-1)
                
                #Бинарные переменные в списке непрерывных
                if len(x.unique()) == 2:
                    a = y[x==x.unique()[0]]
                    b = y[x==x.unique()[1]]
                    
                    if len(a.unique()) > 1 and len(b.unique()) > 1:
                        x_normal = normaltest(a, nan_policy = 'omit').pvalue
                        y_normal = normaltest(b, nan_policy = 'omit').pvalue
                        x_ks = kstest(a, 'norm').pvalue
                        y_ks = kstest(b, 'norm').pvalue
                        
                        if x_normal < 0.01 or y_normal < 0.01 or x_ks < 0.01 or y_ks < 0.01:
                            statistics.append([i, mannwhitneyu(a, b).pvalue, x.corr(ys), 'corr'])
                        else:
                            statistics.append([i, ttest_ind(a, b, nan_policy = 'omit').pvalue, 
                                                   x.corr(ys), 'corr'])
                    else:
                        statistics.append([i, -10, -10, 'One value'])
                            
                else:
                    #Истинно непрерывные переменные
                    if len(x.unique()) > 1 and len(y.unique()) > 1:
                        statistics.append([i, pearsonr(x, y)[1], xs.corr(ys), 'corr'])
                    else:
                        statistics.append([i, -10, -10, 'One value'])
                
                
    labels = ['variable', 'stat pvalue', 'corr', 'corr name']
    
    definition = pd.DataFrame.from_records(statistics, columns = labels)
    
    gc.collect()
    
    return definition


# In[ ]:


def find_doubles(data, columns, categorial_list= None, target_statistics = None, lvl = 0.90, light_unstable = None):
    
    """
    Долгая версия find_doubles, которая по корреляции ищет дублированные переменные. Работает дольше, чем связка 
    receive_correlations+find_doubles_corr
    
    """
    
    if type(light_unstable) != type(None):
        if type(light_unstable) != list:
            light_unstable = list(light_unstable)
    
    k = 0
    
    cols_num = []
    cols_cat = []

    if categorial_list == None:
        for i in columns:
            if (data[i].dtype == 'object') or (len(data[i].dropna().unique()) == 2):
                cols_cat.append(i)
            else:
                cols_num.append(i)
        
    else:
        for i in columns:
            if (data[i].dtype == 'object') or (i in categorial_list):
                cols_cat.append(i)
            else:
                cols_num.append(i)
    
    double_num = []
    firts_num = []
    
    for ii in np.arange(0, len(cols_num)):
        i = cols_num[ii]
        k = k+1
        if  k % 10 ==0 or k == 1:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Numeric', i, k , '| time: ' , tm)
            
        for jj in np.arange(ii, len(cols_num)):
            
            j = cols_num[jj]
            if i != j:

                if abs(data[i].corr(data[j])) >= lvl:
                    print('FOUND', i, j, abs(data[i].corr(data[j])))
                    if type(light_unstable) != type(None):
                            if i in light_unstable and j not in light_unstable:
                                double_num.append(i)
                                firts_num.append(j)
                            elif j in light_unstable and i not in light_unstable:
                                double_num.append(j)
                                firts_num.append(i)
                            else:
                                if type(target_statistics) == type(None):
                                    if data[i].isnull().sum() <= data[j].isnull().sum():
                                        double_num.append(j)
                                        firts_num.append(i)
                                    else:
                                        double_num.append(i)
                                        firts_num.append(j)
                                else:
                                    if abs(list(target_statistics[target_statistics['variable'] == 
                                                i]['corr'])[0]) >= abs(list(target_statistics[target_statistics['variable'] ==
                                                                                    j]['corr'])[0]):
                                        double_num.append(j)
                                        firts_num.append(i)
                                    else:
                                        double_num.append(i)
                                        firts_num.append(j)    
                    else:
                        if type(target_statistics) == type(None):
                            if data[i].isnull().sum() <= data[j].isnull().sum():
                                double_num.append(j)
                                firts_num.append(i)
                            else:
                                double_num.append(i)
                                firts_num.append(j)
                        else:
                            if abs(list(target_statistics[target_statistics['variable'] == 
                                            i]['corr'])[0]) >= abs(list(target_statistics[target_statistics['variable'] ==
                                                                            j]['corr'])[0]):
                                double_num.append(j)
                                firts_num.append(i)
                            else:
                                double_num.append(i)
                                firts_num.append(j)                        

                        
                        
                            
    double_cat = []
    firts_cat = []
    k2=0
    
    for ii in np.arange(0, len(cols_cat)):
        i = cols_cat[ii]
        k2 = k2+1
        if  k2 % 10 ==0 or k2 == 1:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Categorial', i, k2,  '| time: ' , tm)
            
        for jj in np.arange(ii, len(cols_cat)):
            j = cols_cat[jj]
            if i != j:
                
                data1 = data[[i, j]].dropna()
                
                x = data1[i].astype('str')
                y = data1[j].astype('str')
                
                if x.nunique() == 1 or y.nunique() == 1:
                    print('ALARM! One value', i, x.nunique(), j, y.nunique())
                    continue
                
                else:
                    
                    everything = pd.concat([x, y])
                    encoder = LabelEncoder().fit(everything.fillna('MISSING')) 
                    x_label = encoder.transform(x.fillna('MISSING'))
                    y_label = encoder.transform(y.fillna('MISSING'))


                    if abs(spearmanr(x_label, y_label)[0]) >= lvl:
                        print('FOUND', i, j, abs(spearmanr(x_label, y_label)[0]))
                        if type(light_unstable) != type(None):
                            if i in light_unstable and j not in light_unstable:
                                double_cat.append(i)
                                firts_cat.append(j)
                            elif j in light_unstable and i not in light_unstable:
                                double_cat.append(j)
                                firts_cat.append(i)
                            else:
                                if type(target_statistics) == type(None):
                                    if data[i].isnull().sum() <= data[j].isnull().sum():
                                        double_cat.append(j)
                                        firts_cat.append(i)
                                    else:
                                        double_cat.append(i)
                                        firts_cat.append(j)
                                else:
                                    if abs(list(target_statistics[target_statistics['variable'] == 
                                                i]['corr'])[0]) >= abs(list(target_statistics[target_statistics['variable'] ==
                                                                                j]['corr'])[0]):
                                        double_cat.append(j)
                                        firts_cat.append(i)
                                    else:
                                        double_cat.append(i)
                                        firts_cat.append(j)    
                        else:
                            if type(target_statistics) == type(None):
                                if data[i].isnull().sum() <= data[j].isnull().sum():
                                    double_cat.append(j)
                                    firts_cat.append(i)
                                else:
                                    double_cat.append(i)
                                    firts_cat.append(j)
                            else:
                                if abs(list(target_statistics[target_statistics['variable'] == 
                                                i]['corr'])[0]) >= abs(list(target_statistics[target_statistics['variable'] ==
                                                                                j]['corr'])[0]):
                                    double_cat.append(j)
                                    firts_cat.append(i)
                                else:
                                    double_cat.append(i)
                                    firts_cat.append(j)    

                        
    columns2 = [i for i in columns if i not in double_num]
    columns2 = [i for i in columns2 if i not in double_cat]
    result_all_num = pd.DataFrame()
    result_all_cat = pd.DataFrame()
    result_all_num['double_num'] = double_num
    result_all_num['firts_num'] = firts_num
    result_all_cat['double_cat'] = double_cat
    result_all_cat['firts_cat'] = firts_cat
    
    import gc
    gc.collect()
    
    return columns2 , result_all_num, result_all_cat


# In[ ]:


def receive_correlations(data, categorial_list = None):
    
    """
    Рассчитать корреляции. Для непрерывных - Пирсон, для категориальных - Спирман
    
    data - данные
    categorial_list - список категориальных
    
    """
    
    cat_list = []
    num_list = []
    
    if categorial_list == None:
        for i in data.columns:
            if (data[i].dtype == 'object') or (len(data[i].dropna().unique()) == 2):
                cat_list.append(i)
            else:
                num_list.append(i)
        
    else:
        for i in data.columns:
            if (data[i].dtype == 'object') or (i in categorial_list):
                cat_list.append(i)
            else:
                num_list.append(i)
    
    data1 = data.copy()
    data1[cat_list] = data[cat_list].fillna(data[cat_list].min()-1)
    
    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Начало:' , tm)

    corr_num = data1[num_list].corr()
    
    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Корреляционная матрица по числовым переменным посчитана:', tm)
    
    corr_cat = data1[cat_list].corr(method = 'spearman')

    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Корреляционная матрица по категориальным переменным посчитана:', tm)
    
    return corr_num, corr_cat


# In[ ]:


def find_doubles_corr(data, colls, CORR, definition = None, lvl = 0.90, light_unstable = None, silent = False):
    
    """
    Найти "дубли" фичей по корреляционной матрице. 
    data - данные
    colls - переменные
    CORR - матрица корреляций
    lvl - уровень корреляции, по которому идет отсечение.
    definition - матрица статистик с таргетом
    light_unstable - список "легко нестабильных переменных"
    
    Определение дублированных переменных зависит от таблиц, которые подают в функцию:
    - Если подать light_unstable, и одна из переменных в паре находится в этом списке, то она сразу же попадает в 
    список "дублей". Если же ни одной из переменных нет в списке или же есть обе переменные, то отбор происходит либо по 
    definition (если она подана), либо по количеству пропусков в переменных.
    - Если definition подана, то сравниваются модули их корреляций. "Дублем" считается та переменная, у которой он меньше.
    - Если definition не подана, то сравниваются количество пропусков в переменной. "Дублем" считается та переменная, у 
    которой пропусков больше.
    
    Возвращает таблицу с парами "дубль - оставшаяся переменная", список дублированных переменных, список оставшихся
    переменных, словарь соответствия "дубль - оставшаяся переменная"
    
    
    """

    if type(light_unstable) != type(None):
        if type(light_unstable) != list:
            light_unstable = list(light_unstable)
    
    double_col = []
    firts_col = []
    double_dic = {}
    if type(definition) != type(None):
        target_statistics = definition.set_index('variable')['corr']
    for i, c1 in enumerate(CORR.columns):
        for c2 in CORR.columns[i+1:]:
            if abs(CORR[c1][c2]) >= lvl:
                
                if type(light_unstable) != type(None) and c1 not in light_unstable and c2 in light_unstable:
                    d, f = c2, c1
                elif type(light_unstable) != type(None) and c1 in light_unstable and c2 not in light_unstable:   
                    d, f = c1, c2
                else:
                    if type(definition) == type(None):
                        if data[c1].isnull().sum() <= data[c2].isnull().sum():
                            d, f = c2, c1
                        else:
                            d, f = c1, c2
                    else:
                        if abs(target_statistics[c1]) >= abs(target_statistics[c2]):
                            d, f = c2, c1
                        else:
                            d, f = c1, c2
                            
                double_col.append(d)
                firts_col.append(f)
                double_dic[f] = double_dic.get(f,[])
                double_dic[f].append(d)     
                    

    non_doubles = [i for i in CORR.columns if i not in double_col]
    col_doubles = [i for i in CORR.columns if i in double_col]
    result_tab = pd.DataFrame()
    result_tab['double'] = double_col
    result_tab['firts'] = firts_col
    if silent is not True:
        print('Порог =', lvl, 'Осталось фичей =', len(non_doubles), 'Коррелир.фичей =', len(col_doubles) )    
    return result_tab, non_doubles, col_doubles, double_dic


#Функция бинит категории 
def make_standard(data,
                  columns,
                  target,
                  attribute_list,
                  technical_list=None,
                  categorial_list=None,
                  label_encoder=None,
                  string_list=None,
                  mis_value=None,
                  small_treshold=10):
    
    """
    Бинит категориальные переменные, создает _bin переменные для непрерывных переменных
    
    data - данные
    columns - переменные
    target - название таргета
    attribute_list - данные со статистикой по переменным
    
       
    technical_list - список технических переменных
    Таргет и технические переменные исключаются из работы, таргет используется еще и для определения категорий, которые 
    будут сливаться в _Other
    
    categorial_list - список категориальных переменных для разбиения на дамми переменные
    label_encoder - обученная функция LabelEncoder, по которой текстовые переменные были переведены в числовые
    string_list - список текстовых переменных, которые были трансформированы LabelEncoder
    mis_value - чем были при LabelEncoder заменены пропуски?
    small_treshold - порог, по которому определяем категории, которые будут сливаться в _Other. Проверка отличается в зависимости от 
    таргета. Если таргет [0, 1], то small_treshold обозначает минимальное количество goods (target = 1) для категории. Если таргет иной
    (категориальная с большим списком категорий или непрерывная), то small_treshold обозначает количество наблюдений в категории.
    
    Возвращает измененные данные и список изменений следующего формата:
    
    - Название новой переменной
    - Название истинной переменной
    - Правило, по которому была изменена переменная (_bin, _nan, _название категории, значение категории)
    
    Пример: 
    PRSMRTLSTSTYPE_AP_Other | PRSMRTLSTSTYPE_AP | _Other | [4.0]
    
    """
    
    list_numeric = []
    list_categ = []
    
    for i in columns:
        if data[i].dtype == 'object' or (type(categorial_list) != type(None) and i in categorial_list):
            list_categ.append(i)
        else:
            list_numeric.append(i)
            
    if target in list_numeric:
        list_numeric.remove(target)
    
    if target in list_categ:
        list_categ.remove(target)
        
    for tech in technical_list:
        if tech in list_numeric:
            list_numeric.remove(tech)
        elif tech in list_categ:
            list_categ.remove(tech)
            
    list_new = []
    dd = []
    features_new = []
    dd_count = 0
    gc.collect()
    
    for cat in list_categ:
        
        print(cat)
        if (type(label_encoder) != type(None) and type(string_list) == type(None)) or (type(label_encoder) == type(None) 
                                                and type(string_list) != type(None)):
            raise ValueError('Both LabelEncoder and string_list should be passed!')
        elif type(label_encoder) != type(None) and type(string_list) != type(None):
            if cat in string_list:
                if mis_value != None:
                    print(np.where(label_encoder.classes_ == mis_value))
                    if data[cat].dropna().shape[0]==data[cat].shape[0]:
                        data[cat] = label_encoder.inverse_transform(data[cat].astype(int))
                    else:
                        data[cat] = label_encoder.inverse_transform(data[cat].fillna(np.where(label_encoder.classes_ == mis_value)[0][0]).astype(int))
                    data[cat].replace(to_replace = mis_value, value = np.nan, inplace = True)
                else:
                    data[cat] = label_encoder.inverse_transform(data[cat].astype(int))
        
        if data[cat].count() == data.shape[0]:
            test_3 = pd.get_dummies(data[cat])
            test_3 = test_3.drop(test_3.columns[0], axis=1)
            for new_var in test_3.columns:
                features_new.append([cat + '_'+str(new_var), cat, '_'+str(new_var), new_var])
            test_3.columns = [cat + '_'+str(x) for x in test_3.columns]
            list_new.extend(test_3.columns.tolist())
            data = pd.concat([data, test_3], axis=1)
            
            gc.collect()
            
        else:
            test_3 = pd.get_dummies(data[cat])
            for x in test_3.columns:
                features_new.append([cat + '_'+str(x), cat, '_'+str(x), x])
            test_3.columns = [cat + '_'+str(x) for x in test_3.columns]
            list_new.extend(test_3.columns.tolist())
            data = pd.concat([data, test_3], axis=1) 
            gc.collect()

        list_new_2 = []     
        for cat_del in test_3.columns.tolist():
            if len(data[target].unique()) == 2 and sorted(data[target].unique())[0] == 0 and sorted(data[target].unique()) == 1:
                if sum(data[data[cat_del] == 1][target]) <= small_treshold:
                    list_new_2.append(cat_del)
                    print(cat_del, sum(data[data[cat_del] == 1][target]))
            else: 
                if sum(data[cat_del]) <= small_treshold:
                    list_new_2.append(cat_del)
                    print(cat_del, sum(data[cat_del]))
                    
        print(cat, '| count_bin = ', len(test_3.columns), '| del_bin = ' , len(list_new_2))
        list_new_values = []
        for i in list_new_2:
            if data[cat].dtype == 'object':
                list_new_values.append(i.replace(cat+'_', ''))
            else:
                list_new_values.append(eval(i.replace(cat+'_', '')))
            
        dd.extend(list_new_2)
        if len(list_new_2) > 0:
            dd_count = dd_count + 1
            cat_new = cat + '_' + 'Other'
            list_new.append(cat_new)
            features_new.append([cat_new, cat, '_Other', list_new_values])
            data[cat_new] = data[list_new_2].max(axis = 1)
            data = data.drop(list_new_2, axis = 1)
            
    data= data.drop(list_categ, axis = 1)
    
    list_new_bin = []
    list_bin = []
    count_all = data.shape[0]
    for i in attribute_list['attribute']:
        if i in data.columns:
            if attribute_list[attribute_list['attribute'] == i]['count_val'].tolist()[0] < 0.95*count_all:
                col = i + '_bin'
                features_new.append([col, i, '_bin', np.nan])
                data[col] = pd.Series(np.where(pd.isnull(data[i]) == True , 0, 1), index=data.index)
                list_new_bin.append(col)
                list_bin.append(i) 
            
    columns_of_changed_list = ['new variable', 'genuine variable', 'rule', 'values']
    data_of_changes = pd.DataFrame.from_records(features_new, columns = columns_of_changed_list)
    
    indexes_for_delete = []
    other_data = data_of_changes[data_of_changes['rule'] == '_Other']
    
    for i in data_of_changes.index:
        if data_of_changes.loc[i, 'genuine variable'] in list(other_data['genuine variable']):
            values_list = list(other_data.loc[other_data['genuine variable'] == data_of_changes.loc[i, 
                                                 'genuine variable'],  'values'])[0]
            if type(data_of_changes.loc[i, 'values']) != list and data_of_changes.loc[i, 'values'] in values_list:
                indexes_for_delete.append(i)
                
    data_of_changes.drop(indexes_for_delete, axis = 0, inplace = True)
    data.columns = list(map(str.upper, data.columns))
    data_of_changes['new variable'] = data_of_changes['new variable'].str.upper()
    data_of_changes['genuine variable'] = data_of_changes['genuine variable'].str.upper()
    return data, data_of_changes


def data_preprocessing_train(data,
                             target,
                             technical_values,
                             categorial_list=None,
                             drop_technical=False,
                             yeo_johnson=False,
                             attribute_list=None,
                             var_col=None,
                             scale='mean',
                             median='median',
                             high_outlier=None,
                             low_outlier=None,
                             check_percentile=1,
                             cols_outlier=None,
                             cut_non_out_9999=True):
    
    """
    Проводит препроцессинг для train выборки.
    
    data - данные
    target - название таргета или таргет array
    technical_values - список технических переменных
    target и technical_values исключены из анализа. Если технических переменных нет, можно задать пустой список.
    drop_technical - True/False. Удалять или не удалять технические переменные из выборки. По дефолту False. 
     
    yeo_johnson - проводить ли нормализацию Йео-Джонсона (приведение распределения данных к нормальному
    виду). По дефолту False
    attribute_list - данные attribute_list. Могут быть None. По дефолту None
    var_col - в каком поле attribute_list находятся названия фичей. Если attribute_list не задан, то в var_col нет нужды.
    По дефолту None.
    scale - проводить ли стандартизацию StandardScaler/MinMaxScaler. По дефолту 'mean'.
    
    median - импутация пропусков. Возможные принимаемые значения:
        - 'median' - тогда на train куске рассчитываются медианы, и импутация ими
        - 'min-1' - тогда на train куске рассчитываются минимальные значения - 1, и импутация ими
        - число - если задать число, то пропуски будут заполняться этим числом
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
        - None. Тогда пропуски не импутируются
        По дефолту задано значение 'median'
    high_outlier - импутация выбросов вверх. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 99 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
        - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
        - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
        - None. Тогда выбросы вверх не импутируются.
    low_outlier - импутация выбросов вниз. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 1 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
        - None. Тогда выбросы вниз не импутируются. 
     
     Возвращает измененные данные и обученный Scaler(или Йео_Джонсон)
     
    """
    
    xtrain = data.copy()
    
    cols = list(xtrain.columns)
    
    if drop_technical == True:
        for i in technical_values:
            if i in xtrain.columns:
                xtrain.drop(i, axis = 1, inplace= True)
                
    if type(technical_values)!=type(None): 
        for i in technical_values:
            if i in cols:
                cols.remove(i)
    
    if type(target) == str:
        if target in cols:
            cols.remove(target)
    
    if type(categorial_list) != type(None):
        categorial_cols = categorial_list
        numeric_cols = cols.copy() 
        for i in categorial_cols:
            if i in numeric_cols:
                numeric_cols.remove(i)
        
    else:
        categorial_cols = []
        for cc in xtrain.columns:
            if xtrain[cc].nunique() == 2:
                if sorted(xtrain[cc].unique())[0] == 0 and sorted(xtrain[cc].unique())[1] == 1:
                    categorial_cols.append(cc) 
        numeric_cols = cols.copy()
        for i in categorial_cols:
            if i in numeric_cols:
                numeric_cols.remove(i)
    
    if type(cols_outlier) == type(None):
        cols_outlier = xtrain.columns
                    
    for oo in tqdm(cols):

        if median != None:
            if median == 'median':
                medians = xtrain[oo].median(skipna = True)
            elif median == 'min-1':
                medians = xtrain[oo].min(skipna = True)-1
            elif type(attribute_list) != type(None) and median in attribute_list.columns:
                medians = list(attribute_list.loc[attribute_list[var_col] == oo, median])[0]
            else:
                medians = median
                
        if high_outlier != None:
            if oo in cols_outlier:
                if type(attribute_list) != type(None) and high_outlier in attribute_list.columns:
                    to_replace_high = list(attribute_list.loc[attribute_list[var_col] == oo, high_outlier])[0]
                    
                elif high_outlier == 'IQR':
                    check_25 = np.nanpercentile(xtrain[oo], 25)
                    check_75 = np.nanpercentile(xtrain[oo], 75)
                    check_99 = np.nanpercentile(xtrain[oo], 100-check_percentile)
                    maximum = xtrain[oo].max()
                    
                    if check_25 != check_75:
                        q_25 = np.nanpercentile(xtrain[oo], 25)
                        q_75 = np.nanpercentile(xtrain[oo], 75)
                        iqr = q_75-q_25
                        right_border = q_75+iqr*1.5
                    else:
                        x = xtrain.loc[xtrain[oo] != check_25, oo]
                        q_25 = np.nanpercentile(x, 25)
                        q_75 = np.nanpercentile(x, 75)
                        iqr = q_75-q_25
                        right_border = q_75+iqr*1.5
                    
                    if right_border > maximum:
                        to_replace_high = maximum
                    elif check_99 > right_border:
                        to_replace_high = check_99
                    else:
                        to_replace_high = right_border
                            
                elif high_outlier == 'z-score':
                    to_replace_high = 3*xtrain[oo].std()+xtrain[oo].mean()
                else:
                    to_replace_high = np.nanpercentile(xtrain[oo], high_outlier)
                    
            elif oo not in cols_outlier:
                if oo in numeric_cols:
                    if cut_non_out_9999 == True:
                        to_replace_high = np.nanpercentile(xtrain[oo], 99.99)
                    else: 
                        to_replace_high = max(xtrain[oo])
                else:
                    to_replace_high = max(xtrain[oo])
                        
        elif high_outlier == None:
            to_replace_high = None

        if low_outlier != None:
            if oo in cols_outlier:
                if type(attribute_list) != type(None) and low_outlier in attribute_list.columns:
                    to_replace_low = list(attribute_list.loc[attribute_list[var_col] == oo, low_outlier])[0]
                    
                elif low_outlier == 'IQR':
                    check_25 = np.nanpercentile(xtrain[oo], 25)
                    check_75 = np.nanpercentile(xtrain[oo], 75)
                    check_1 = np.nanpercentile(xtrain[oo], check_percentile)
                    minimum = xtrain[oo].min()
                    if check_25 != check_75:
                        q_25 = np.nanpercentile(xtrain[oo], 25)
                        q_75 = np.nanpercentile(xtrain[oo], 75)
                        iqr = q_75-q_25
                        left_border = q_25-iqr*1.5
                    else:
                        x = xtrain.loc[xtrain[oo] != check_25, oo]
                        q_25 = np.nanpercentile(x, 25)
                        q_75 = np.nanpercentile(x, 75)
                        iqr = q_75-q_25
                        left_border = q_25-iqr*1.5
                        
                    if left_border < minimum:
                        to_replace_low = minimum
                    elif check_1 < left_border:
                        to_replace_low = check_1
                    else:
                        to_replace_low = left_border
                        
                elif low_outlier == 'z-score':
                    to_replace_low = (-1)*3*xtrain[oo].std()+xtrain[oo].mean()
                else:
                    to_replace_low = np.nanpercentile(xtrain[oo], low_outlier)
                    
        elif low_outlier == None:
            to_replace_low = None
                        
        if median != None:
            xtrain[oo] = xtrain[oo].fillna(medians)
        if to_replace_high != None:
            if oo in cols_outlier:
                xtrain.loc[xtrain[oo] > to_replace_high, oo] = to_replace_high
        if to_replace_low != None:
            if oo in cols_outlier:
                xtrain.loc[xtrain[oo] < to_replace_low, oo] = to_replace_low
    
    
    if yeo_johnson == False and scale != False:
        if scale == 'mean':
            pr = preprocessing.StandardScaler()
            pr.fit(xtrain[numeric_cols])
            xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
            return xtrain, pr
        elif scale == 'minmax':
            pr = preprocessing.MinMaxScaler()
            pr.fit(xtrain[numeric_cols])
            xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
            return xtrain, pr
        
    elif yeo_johnson == True:
        power = PowerTransformer(method = 'yeo-johnson', standardize = False).fit(xtrain[numeric_cols])
        xtrain[numeric_cols] = power.transform(xtrain[numeric_cols])
        pr = power
        
        pr2 = preprocessing.StandardScaler()
        pr2.fit(xtrain[numeric_cols])
        xtrain[numeric_cols] = pr2.transform(xtrain[numeric_cols])
 
        return xtrain, pr, pr2
    else:
        return xtrain


def data_preprocessing_test(data,
                            target,
                            technical_values,
                            categorial_list=None,
                            drop_technical=False,
                            attribute_list=None,
                            var_col=None,
                            median=None,
                            high_outlier=None,
                            low_outlier=None,
                            scale=None,
                            yeo_johnson=None,
                            cols_outlier=None):

    """
    Проводит препроцессинг для test выборки.
    
    data - данные
    target - название таргета или таргет array
    technical_values - список технических переменных
    target и technical_values исключены из анализа. Если технических переменных нет, можно задать пустой список.
    drop_technical - True/False. Удалять или не удалять технические переменные из выборки. По дефолту False. 
    attribute_list - данные attribute_list. Могут быть None. По дефолту None
    var_col - в каком поле attribute_list находятся названия фичей. Если attribute_list не задан, то в var_col нет нужды.
    По дефолту None.
    
    median - импутация пропусков. Возможные принимаемые значения:
        - число - если задать число, то пропуски будут заполняться этим числом
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
        - None. Тогда пропуски не импутируются
        Если будет значение 'median' или  'min-1', то функция выдаст ошибку (потому что на тестовом куске вычислять статистики нельзя!)
        По дефолту задано значение None
    high_outlier - импутация выбросов вверх. Возможные принимаемые значения:
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
        - None. Тогда выбросы вверх не импутируются.
        Если дано значение от 0 до 100, то функция выдаст ошибку (потому что на тестовом куске вычислять статистики 
        нельзя!)
        По умолчанию None
    low_outlier - импутация выбросов вниз. Возможные принимаемые значения:
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
        - None. Тогда выбросы вниз не импутируются. 
        Если дано значение от 0 до 100, то функция выдаст ошибку (потому что на тестовом куске вычислять статистики 
        нельзя!)
        По умолчанию None
    
    yeo_johnson - если None, то Йео-Джонсон не проводится. Если надо, чтобы Йео-Джонсон был проведен, то надо подать 
    обученного Йео-Джонсонса. По дефолту None.
    scale - если None, то Стандартизация не проводится. Если надо, чтобы стандартизация была проведена, то надо подать 
    обученный StandardScaler. По дефолту None.
    
    Возвращает измененные данные
     
    """
    
    xtest = data.copy()
    
    cols = list(xtest.columns)
    
    if drop_technical == True:
        for i in technical_values:
            if i in xtest.columns:
                xtest.drop(i, axis = 1, inplace= True)
        
    for i in technical_values:
        if i in cols:
            cols.remove(i)
    
    if type(target) == str:
        if target in cols:
            cols.remove(target)
        
    if type(categorial_list) != type(None):
        categorial_cols = categorial_list
        numeric_cols = cols.copy()
        for i in categorial_cols:
            if i in numeric_cols:
                numeric_cols.remove(i)
    else:
        categorial_cols = []
        for cc in xtest.columns:
            if xtest[cc].nunique() == 2:
			# Исправлено
                if sorted(xtest[cc].unique())[0] == 0 and sorted(xtest[cc].unique())[1] == 1:
                    categorial_cols.append(cc) 
        numeric_cols = cols.copy()
        for i in categorial_cols:
            if i in numeric_cols:
                numeric_cols.remove(i)
    
    if type(cols_outlier) == type(None):
        cols_outlier = xtest.columns
    
    xtest = xtest.astype('float64')
    
    for oo in tqdm(cols):
        if median != None:
            if median == 'median':
                raise ValueError('Test data cannot be used for statistics calculation!')
            elif median == 'min-1':
                raise ValueError('Test data cannot be used for statistics calculation!')
            elif type(attribute_list) != type(None) and median in attribute_list.columns:
                medians = list(attribute_list.loc[attribute_list[var_col] == oo, median])[0]
            else:
                medians = median
                
        if high_outlier != None:
            if oo in cols_outlier:
                if type(attribute_list) != type(None) and high_outlier in attribute_list.columns:
                    to_replace_high = list(attribute_list.loc[attribute_list[var_col] == oo, high_outlier])[0]
                else:
                    raise ValueError('Test data cannot be used for statistics calculation!')
        elif high_outlier == None:
            to_replace_high = None
                        
        if low_outlier != None:
            if oo in cols_outlier:
                if type(attribute_list) != type(None) and low_outlier in attribute_list.columns:
                    to_replace_low = list(attribute_list.loc[attribute_list[var_col] == oo, low_outlier])[0]
                else:
                    raise ValueError('Test data cannot be used for statistics calculation!')
        elif low_outlier == None:
            to_replace_low = None
                        
        if median != None:
            xtest[oo] = xtest[oo].fillna(medians)
        if oo in cols_outlier:    
            if to_replace_high != None:
                xtest.loc[xtest[oo] > to_replace_high, oo] = to_replace_high
            if to_replace_low != None:
                xtest.loc[xtest[oo] < to_replace_low, oo] = to_replace_low
            
    if type(yeo_johnson) == type(None) and type(scale) != type(None):
        xtest[numeric_cols] = scale.transform(xtest[numeric_cols])
        
    elif type(yeo_johnson) != type(None) and type(scale) != type(None):
        xtest[numeric_cols] = yeo_johnson.transform(xtest[numeric_cols])
        xtest[numeric_cols] = scale.transform(xtest[numeric_cols])
        
    elif type(yeo_johnson) != type(None) and type(scale) == type(None):
        xtest[numeric_cols] = yeo_johnson.transform(xtest[numeric_cols])
        
    return xtest


# In[ ]:



def two_forests(data1, y1, data2, y2, param_dict, task = 'binary', use_metric = None, treshold_metric = True, 
                higher_is_better = True, features_list = None, 
                n_samples_our = None, several_perms = None, random_state = None):
    
    """
    Считает важности с использованием двух-лесового метода (пермутированная важность).
    
    Метод основан на делении выборки на 2 куска ***!РАВНОГО!*** размера и вычислении пермутированной важности на 
    на отложенной выборке для каждого из равных кусков.
    Далее вычисляется среднее значение важности по этим двум кускам. 
    Использовать можно различные метрики, но следует учесть, что пороговые метрики все считаются для порога 50%. Если происходит
    пермутирование, то оптимальное значение порога для метрики может сместиться, что может сместить результаты значимости переменных.
    Поэтому лучше смотреть интегральные метрики и мерить importance по ним.
    Для интегральных метрик большая часть переменных будет иметь хорошее значение p_value, так что можно отбирать еще и по количеству 
    переменных, которое можно взять в модель, однако если переменная имеет высокое значение p value, она вряд ли будет информативной, так
    что по p value стоит отрезать первичный сет переменных.
    
    data1 - данные первого куска
    y1 - таргет для первого куска (array)
    data2 - данные второго куска
    y2 - таргет для второго куска (array)
    param_dict - словарь с параметрами для RandomForest, включая random_state
    classifier - флаг задачи классификации или регрессии. True - классификация, False - регрессия. Если использется задача регрессии, то
    по дефолту используется метрика R^2. Другие метрики также задаются
    use_metric - вызываемая метрика, по которой считается importance, если не подать, то считаться будет accuracy
    treshold_metric - True, если используется пороговая метрика, False - если используется интегральная метрика
    higher_is_better - True если большие значения метрики - означают лучший классификатор. False если цель сделать метрику меньше. 
    features_list - список переменных, которые надо пермутировать (если подать часть переменных в списке, то они будут пермутироваться
    вместе). 
    n_samples_our - количество наблюдений, которые надо пермутировать
    several_perms - Сколько раз пермутировать данные? Если None, то будет выполняться одна пермутация, и выводиться будет importance
    для одной пермутации. Если several_perms - любое число кроме 0, то это число означает количество ДОПОЛНИТЕЛЬНЫХ пермутаций, по которым
    будет происходить усреднение. В таком случае функция выдаст импортансы для первой пермутации и усреднения для всех остальных.
    random_state - random_state для пермутаций
    
    Возвращает: если several_perms не задан, то return importance[['Feature', 'Importance']], import1, import2
    Если several_perms задан, то return importance[['Feature', 'Importance', 'p_value']], import1, import2, 
    importance_grouped[['Feature', 'Importance', 'p_value']], several_perms1, several_perms2
    
    """
    
    if callable(use_metric):
        def metric_our(model, X_valid, y_valid, sample_weights_list, treshold_metric, task):

            if ((treshold_metric == True) and (task in ['binary', 'multiclass'])) or (task == 'numeric'):
                prediction = model.predict(X_valid)
            elif (treshold_metric == False) and (task == 'binary'):
                prediction = model.predict_proba(X_valid)[:, 1]
            elif (treshold_metric == False) and (task == 'multiclass'):
                prediction = model.predict_proba(X_valid)
            else:
                print('Проверьте правильность заполнения параметров task и treshold_metric')
                print(f'task:{task}, threshold_metric:{treshold_metric}')
            scores = use_metric(y_valid, prediction)
            return scores
    else:
        metric_our = None
        
        
    def sample(X_valid, y_valid, n_samples, sample_weights=None, random_state = None):
        if n_samples < 0: n_samples = len(X_valid)
        n_samples = min(n_samples, len(X_valid))
        if n_samples < len(X_valid):
            ix = np.random.RandomState(random_state).choice(len(X_valid), n_samples)
            X_valid = X_valid.iloc[ix].copy(deep=False)  # shallow copy
            y_valid = y_valid.iloc[ix].copy(deep=False)
            if sample_weights is not None: sample_weights = sample_weights.iloc[ix].copy(deep=False)
        return X_valid, y_valid, sample_weights


    def sample_rows(X, n_samples, random_state = None):
        if n_samples < 0: n_samples = len(X)
        n_samples = min(n_samples, len(X))
        if n_samples < len(X):
            ix = np.random.RandomState(random_state).choice(len(X), n_samples)
            X = X.iloc[ix].copy(deep=False)  # shallow copy
        return X

    def importances(model, X_valid, y_valid, features=None, n_samples=5000, sort=True, 
                    metric=None, sample_weights = None, higher_is_better = True, random_state = None, treshold_metric=None, task='binary'):
        """
        Compute permutation feature importances for scikit-learn models using
        a validation set.
        Given a Classifier or Regressor in model
        and validation X and y data, return a data frame with columns
        Feature and Importance sorted in reverse order by importance.
        The validation data is needed to compute model performance
        measures (accuracy or R^2). The model is not retrained.
        You can pass in a list with a subset of features interesting to you.
        All unmentioned features will be grouped together into a single meta-feature
        on the graph. You can also pass in a list that has sublists like:
        [['latitude', 'longitude'], 'price', 'bedrooms']. Each string or sublist
        will be permuted together as a feature or meta-feature; the drop in
        overall accuracy of the model is the relative importance.
        The model.score() method is called to measure accuracy drops.
        This version that computes accuracy drops with the validation set
        is much faster than the OOB, cross validation, or drop column
        versions. The OOB version is a less vectorized because it needs to dig
        into the trees to get out of examples. The cross validation and drop column
        versions need to do retraining and are necessarily much slower.
        This function used OOB not validation sets in 1.0.5; switched to faster
        test set version for 1.0.6. (breaking API change)
        :param model: The scikit model fit to training data
        :param X_valid: Data frame with feature vectors of the validation set
        :param y_valid: Series with target variable of validation set
        :param features: The list of features to show in importance graph.
                         These can be strings (column names) or lists of column
                         names. E.g., features = ['bathrooms', ['latitude', 'longitude']].
                         Feature groups can overlap, with features appearing in multiple.
        :param n_samples: How many records of the validation set to use
                          to compute permutation importance. The default is
                          5000, which we arrived at by experiment over a few data sets.
                          As we cannot be sure how all data sets will react,
                          you can pass in whatever sample size you want. Pass in -1
                          to mean entire validation set. Our experiments show that
                          not too many records are needed to get an accurate picture of
                          feature importance.
        :param sort: Whether to sort the resulting importances
        :param metric: Metric in the form of callable(model, X_valid, y_valid, sample_weights) to evaluate for,
                        if not set default's to model.score()
        :param sample_weights: set if a different weighting is required for the validation samples
        return: A data frame with Feature, Importance columns
        SAMPLE CODE
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        X_train, y_train = ..., ...
        X_valid, y_valid = ..., ...
        rf.fit(X_train, y_train)
        imp = importances(rf, X_valid, y_valid)
        """

        def flatten(features, random_state = None):
            all_features = set()
            for sublist in features:
                if isinstance(sublist, str):
                    all_features.add(sublist)
                else:
                    for item in sublist:
                        all_features.add(item)
            return all_features

        if features is None:
            # each feature in its own group
            features = X_valid.columns.values
        else:
            req_feature_set = flatten(features)
            model_feature_set = set(X_valid.columns.values)
            # any features left over?
            other_feature_set = model_feature_set.difference(req_feature_set)
            if len(other_feature_set) > 0:
                # if leftovers, we need group together as single new feature
                features.append(list(other_feature_set))

        X_valid, y_valid, sample_weights = sample(X_valid, y_valid, n_samples, sample_weights=sample_weights, random_state = random_state)
        X_valid = X_valid.copy(deep=False)  # we're modifying columns

        if callable(metric):
            baseline = metric(model, X_valid, y_valid, sample_weights, treshold_metric, task)
        else:
            baseline = model.score(X_valid, y_valid, sample_weights)

        imp = []
        # k = 0

        for group in tqdm(features, desc='Two Forests', total=len(features)):
            # k = k+1
            # if k % 30 ==0:
            #     tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            #     print ('Number of finished repetitions:', k , '| time: ' , tm)

            if isinstance(group, str):
                save = X_valid[group].copy()
                X_valid[group] = np.random.RandomState(random_state).permutation(X_valid[group])
                if callable(metric):
                    m = metric(model, X_valid, y_valid, sample_weights, treshold_metric, task)
                else:
                    m = model.score(X_valid, y_valid, sample_weights)
                X_valid[group] = save
            else:
                save = {}
                for col in group:
                    save[col] = X_valid[col].copy()
                for col in group:
                    X_valid[col] = np.random.RandomState(random_state).permutation(X_valid[col])

                if callable(metric):
                    m = metric(model, X_valid, y_valid, sample_weights, treshold_metric, task)
                else:
                    m = model.score(X_valid, y_valid, sample_weights)
                for col in group:
                    X_valid[col] = save[col]
            if higher_is_better == True:
                imp.append(baseline - m)
            else:
                imp.append(m-baseline)



        # Convert and groups/lists into string column names
        labels = []
        for col in features:
            if isinstance(col, list):
                labels.append('\n'.join(col))
            else:
                labels.append(col)

        I = pd.DataFrame(data={'Feature': labels, 'Importance': np.array(imp)})
        I = I.set_index('Feature')
        if sort:
            I = I.sort_values('Importance', ascending=False)
        return I
    
    if task in ['binary', 'multiclass']:
        rf = RandomForestClassifier(**param_dict)
    elif task == 'numeric':
        rf = RandomForestRegressor(**param_dict)  # Исправлено
    
    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Start of random forest fit', '| time: ' , tm)
    
    rf1 = rf.fit(data1, y1)
    f1_score1 = metric_our(rf1, data2, y2, None, treshold_metric, task)
    
    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('First random forest end', '| time: ' , tm, 
           'Score train:', round(metric_our(rf1, data1, y1, None, treshold_metric, task),4), 
           'Score test:', round(f1_score1, 4))
    
    rf2 = rf.fit(data2, y2)
    f1_score2 = metric_our(rf2, data1, y1, None, treshold_metric, task)
    
    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Second random forest end', '| time: ' , tm, 
           'Score train:', round(metric_our(rf2, data2, y2, None, treshold_metric, task),4), 
           'Score test:', round(f1_score2, 4))
    
    print('\nFIRST IMPORTANCE:')
    
    import1 = importances(rf1, data2, y2, metric=metric_our, n_samples = n_samples_our, features = features_list,
                         higher_is_better = higher_is_better, random_state = random_state, 
                         treshold_metric=treshold_metric, task=task)
    # tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    # print ('First importance end', '| time: ' , tm)
    
    print('SECOND IMPORTANCE:')
    
    import2 = importances(rf2, data1, y1, metric=metric_our, n_samples = n_samples_our, features = features_list,
                         higher_is_better = higher_is_better, random_state = random_state, 
                         treshold_metric=treshold_metric, task=task)
    # tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    # print ('Second importance end', '| time: ' , tm)
    
    import1 = import1.reset_index()
    import2 = import2.reset_index()
    
    importance = pd.merge(import1, import2, on = 'Feature', suffixes = ["1", "2"])

    importance['Importance'] = (importance['Importance1'] + importance['Importance2'])/2

    imp_neg = importance[importance['Importance'] < 0]['Importance']
    imp_zer = importance[importance['Importance'] == 0]['Importance']
    imp_pos = imp_neg*(-1)

    #imp1_neg = import1[import1['Importance'] <= 0]['Importance']
    #imp1_zer = import1[import1['Importance'] == 0]['Importance']
    #imp1_pos = imp1_neg*(-1)

    #imp2_neg = import2[import2['Importance'] <= 0]['Importance']
    #imp2_zer = import2[import2['Importance'] == 0]['Importance']
    #imp2_pos = imp2_neg*(-1)

    #all_negs = sorted(pd.concat([imp1_neg, imp1_pos, imp1_zer, imp2_neg, imp2_pos, imp2_zer]), reverse = True)

    all_negs = sorted(pd.concat([imp_neg, imp_pos, imp_zer]), reverse = True)
    all_len = len(np.unique(all_negs))
    
    if all_len in [0,1]:

        return importance[['Feature', 'Importance']], import1, import2

    else:

        all_negs_cdf = edf.ECDF(all_negs)
        slope_changes = sorted(set(all_negs))

        sample_edf_values_at_slope_changes = [all_negs_cdf(item) for item in slope_changes]
        inverted_edf = interp1d(slope_changes, sample_edf_values_at_slope_changes)
        
        for i, v in enumerate(importance['Feature']):

            imp_number = importance[importance['Feature'] == v]['Importance']
            if imp_number[i]> max(all_negs):
                quest = max(all_negs)
            else:
                quest = imp_number[i]
            importance.loc[importance['Feature'] == v, 'p_value'] = 1-inverted_edf(quest)

        if several_perms == None:
        
            return importance[['Feature', 'Importance', 'p_value']], import1, import2
    
        else: 
        
            several_perms1 = import1.copy()
            several_perms2 = import2.copy()

            several_perms1['Number_of_perms'] = 0
            several_perms2['Number_of_perms'] = 0

            n = 0

            for i in range(several_perms):

                n+=1

                print(30*'-', 'NUMBER OF PERMUTATION', i, 30*'-')

                print(20*'-', 'START OF FIRST IMPORTANCE', 20*'-')

                import_new1 = importances(rf1, data2, y2, metric=metric_our, n_samples = n_samples_our, 
                                         features = features_list, higher_is_better = higher_is_better, 
                                         random_state = random_state+i+1, treshold_metric=treshold_metric, task=task)
                tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
                print ('First importance end', '| time: ' , tm)

                print(20*'-', 'START OF SECOND IMPORTANCE', 20*'-')

                import_new2 = importances(rf2, data1, y1, metric=metric_our, n_samples = n_samples_our, 
                                         features = features_list, higher_is_better = higher_is_better, 
                                         random_state = random_state+i+1, treshold_metric=treshold_metric, task=task)
                tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
                print ('Second importance end', '| time: ' , tm)

                import_new1.reset_index(inplace = True)
                import_new2.reset_index(inplace = True)

                import_new1['Number_of_perms'] = i+1
                import_new2['Number_of_perms'] = i+1

                several_perms1 = pd.concat([several_perms1, import_new1], axis = 0)
                several_perms2 = pd.concat([several_perms2, import_new2], axis = 0)

            grouped_several_perms1 = several_perms1.groupby('Feature').aggregate({'Importance': 'mean'})
            grouped_several_perms2 = several_perms2.groupby('Feature').aggregate({'Importance': 'mean'})

            grouped_several_perms1.reset_index(inplace = True)
            grouped_several_perms2.reset_index(inplace = True)

            importance_grouped = pd.merge(grouped_several_perms1, 
                                          grouped_several_perms2, on = 'Feature', suffixes = ["1", "2"])

            importance_grouped['Importance'] = (importance_grouped['Importance1'] + importance_grouped['Importance2'])/2

            imp_neg = importance_grouped[importance_grouped['Importance'] < 0]['Importance']
            imp_zer = importance_grouped[importance_grouped['Importance'] == 0]['Importance']
            imp_pos = imp_neg*(-1)    

            all_negs = sorted(pd.concat([imp_neg, imp_pos, imp_zer]), reverse = True)
            all_len = len(all_negs)
            
            all_negs_cdf = edf.ECDF(all_negs)
            slope_changes = sorted(set(all_negs))

            sample_edf_values_at_slope_changes = [all_negs_cdf(item) for item in slope_changes]
            inverted_edf = interp1d(slope_changes, sample_edf_values_at_slope_changes)

            for i, v in enumerate(importance_grouped['Feature']):

                imp_number = importance_grouped[importance_grouped['Feature'] == v]['Importance']
                if imp_number[i]> max(all_negs):
                    quest = max(all_negs)
                else:
                    quest = imp_number[i]
                importance_grouped.loc[importance_grouped['Feature'] == v, 'p_value'] = 1-inverted_edf(quest)         
            
        
            return importance[['Feature', 'Importance', 'p_value']], import1, import2, importance_grouped[['Feature', 'Importance', 'p_value']], several_perms1, several_perms2



# In[ ]:


def turn_variables(data1, target, rules_list, lvl =10, oth_list = None):
    
    """
    Функция оставляет в выборке только интересующие нас переменные. 
    data - данные
    target - таргет. Может быть как array, так и str
    rules_list - данные соответствия новые-старые переменные
    lvl - сколько максимально должно быть наблюдений с y=1 в категории, чтобы она была объединена в Other? используется только если 
    oth_list = None
    oth_list - если oth_list не None, то используется для определения _Other
    
    """
        
    data = data1.copy()
    
    if oth_list != None:
        if type(oth_list) == list:
            other_data = pd.DataFrame.from_records(oth_list, columns = ['variable', 'values'])
        elif type(oth_list) == pd.core.frame.DataFrame:
            other_data = oth_list
            for oth_d_col in oth_list.columns:
                if type(other_data.loc[0, oth_d_col]) == str:
                    other_data.rename(columns = {oth_d_col: 'variable'}, inplace = True)
                elif type(other_data.loc[0, oth_d_col]) == list:
                    other_data.rename(columns = {oth_d_col: 'values'}, inplace = True)
    
    new_variables = rules_list['new variable'].to_list()
    if type(target) == str:
        positive_index = np.where(data[target] == 1)[0]
    else:    
        positive_index = np.where(target == 1)[0]
    list_of_other = []
    
    for i in rules_list.index:
        if rules_list.loc[i, 'rule'] == '_bin':
            data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(pd.isnull(data[rules_list.loc[i, 
                                                                       'genuine variable']]) == True , 0, 1), 
                                                                index=data.index)
            
        elif rules_list.loc[i, 'rule'] == 'Missing':
            continue
        elif rules_list.loc[i, 'rule'] == '_nan':
            data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(pd.isnull(data[rules_list.loc[i, 
                                                    'genuine variable']]) == True , 1, 0), index=data.index)
            
        elif rules_list.loc[i, 'rule'] == '_Other' and oth_list == None:
            counts_of_dist_values = data.iloc[positive_index][rules_list.loc[i, 'genuine variable']].value_counts()
            found_values = counts_of_dist_values.keys()[np.where(counts_of_dist_values < lvl)[0]]
            data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(data[rules_list.loc[i, 
                                        'genuine variable']].isin(found_values), 1, 0), index=data.index)
            list_of_other.append([rules_list.loc[i, 'genuine variable'], found_values])
            
        elif rules_list.loc[i, 'rule'] == '_Other' and type(oth_list) != type(None):
            found_values = other_data.loc[other_data['variable'] == rules_list.loc[i, 'genuine variable'], 'values']
            data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(data[rules_list.loc[i, 
                                        'genuine variable']].isin(found_values), 1, 0), index=data.index)
                
        else:
            red_rule = rules_list.loc[i, 'rule'].replace('_', '')
            counts_of_dist_values = data[rules_list.loc[i, 'genuine variable']].value_counts()
            for o in counts_of_dist_values.keys():
                if type(o) == str:
                    if str(o) == red_rule:
                        number = o
                elif type(o) == int or type(o) == float:
                    if o == eval(red_rule):
                        number = o

                    data[rules_list.loc[i, 'new variable']] = pd.Series(np.where((data[rules_list.loc[i, 
                                                    'genuine variable']] == number), 1, 0), index=data.index)
            
    data = data[new_variables]
    return data, list_of_other


# In[ ]:


def turn_variables_with_values(data, rules_list):
        
    """
    Функция оставляет в выборке только интересующие нас переменные. 
    data - данные
    rules_list - DataFrame с информацией о фичах, которые оставляем. Структура датафрейма:
        
    new variable - новая переменная, которая попадет в модель
    genuine variable - старая переменная, которую надо изменить
    rule - правило, по которому было изменение (постфикс).
    values - значения, которые принимала истинная переменная (для бинов на категории или Others)
    
    Возвращает измененные данные
    
    !ВАЖНО!
    Перед подачей данных следует убедиться, что в поле "values" находятся значения правильного типа! При загрузке данных rules_list из csv 
    все значения values могут превратиться в текст, что нарушит работу алгоритма!
    
    """
        
    data = data.copy()
  
    new_variables = rules_list['new variable'].to_list()
    
    for i in rules_list.index:
        if rules_list.loc[i, 'rule'] == '_bin':
            data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(pd.isnull(data[rules_list.loc[i, 
                                                                       'genuine variable']]) == True , 0, 1), 
                                                                index=data.index)
            
        elif rules_list.loc[i, 'rule'] == 'Missing':
            continue
        elif rules_list.loc[i, 'rule'] == '_nan':
            data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(pd.isnull(data[rules_list.loc[i, 
                                                    'genuine variable']]) == True , 1, 0), index=data.index)
            
        elif rules_list.loc[i, 'rule'] == '_Other':
            found_values = rules_list.loc[i, 'values']
            if type(found_values) == type('One'):    
                try:
                    found_val = eval(found_values)
                except SyntaxError:
                    found_val = found_values
                except NameError:
                    found_val = found_values
            else:
                found_val = found_values
            
            
            data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(data[rules_list.loc[i, 
                                        'genuine variable']].isin(found_val), 1, 0), index=data.index)
                
        else:
            number1 = rules_list.loc[i, 'values']
            if type(number1) == type('One'):            
                try:
                    number = eval(number1)
                except SyntaxError:
                    number = number1
                except NameError:
                    number = number1
            else:
                number = number1
            
            
            if type(number) != list:
                data[rules_list.loc[i, 'new variable']] = pd.Series(np.where((data[rules_list.loc[i, 
                                                                    'genuine variable']] == number), 1, 0), index=data.index)
            else:
                data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(data[rules_list.loc[i, 
                                            'genuine variable']].isin(number), 1, 0), index=data.index)
            
    data = data[new_variables]
    
    return data


# In[ ]:


def find_meta_params(X, Y, params_dictionary, params_to_model, pass_model, 
                     sort_by_var, list_of_vars_for_strat, n_folds, second_target, yeo_johnson, 
                     attribute_list, var_col, categorial_list = None, cols_outlier = None, need_business = True, 
                     draw = True, draw_by_approval_rate = False,
                     simple_b_score = None, business_dict = None, business_dict_sec = None,
                     scale = 'mean', median = 'median',
                     high_outlier = None, 
                     low_outlier = None, check_percentile = 1, random_state = None, task = 'binary', k_logs = 10,
                     cut_non_out_9999 = True):
    
    """
    Функция find_meta_params получается meta файл, в котором содержится brute-force поиск по сетке. Функция может применяться к различным моделям и для различных параметров, аналогично GridSearchCV. Параметры:

    - X, Y - матрица данных и таргет. Матрица X не должна быть предобработана, так как предобработка делается внутри функции! 

    - params_dictionary - словарь с параметрами, аналогично GridSearchCV. Пример: params_dictionary = {'C': [0.05, 0.1, 0.2], 'weight_0': [0.01, 0.015], 'regularization': 'l2', 'random_state': 241, 'solver': 'liblinear', 'max_iter': 300}. ***!Важно!*** Если подбирается параметр веса класса, ключ имеет значение:

        - Если параметр веса задается в виде списка или np.array веса одного из классов, следует указать какого именно. Например, если производится поиск веса для класса 0, примеры возможных названий: 'weight_0', '0_weight', 'var_0_weight', 'weight_var_0'. В любом случае должны присутстовать '0' и 'weight'. Если ищется класс 1, то должны присутствовать '1' и 'weight'.
        - Если параметр веса с самого начала задается словарем аля [{0: 0.07, 1:1}, {0: 0.14, 1:1}], то ключ словаря обязан иметь 'class_weight'

    - params_to_model - так как ***params_dictionary*** ключи словаря могут отличаться от обозначений в функции, следует задать словарь соответствия. Пример: params_to_model = {'C': 'C', 'weight_0': 'class_weight', 'regularization':'penalty', 'random_state': 'random_state', 'solver': 'solver', 'max_iter': 'max_iter'}

    - pass_model - вызываемая функция. Пример: pass_model = LogisticRegression или pass_model = DecisionTreeClassifier

    - sort_by_var - переменная для деления (пример - id клиента, клиенты должны попасть либо в тест, либо в трейн)
    
    - list_of_vars_for_strat - список переменных для стратификации. Пример: распределение по регионам, распределение по месяцам
    
    - n_folds - количество фолдов

    - second_target - используется ли второй таргет (как в research для модели CRM)

    - yeo_johnson - используется ли преобразование Йео-Джонсона. Если его нет, делается стандартизация StandardScaler
    
    - attribute_list - аттрибут лист для использования его в импутациях пропусков и обрезании выбросов
    
    - var_col - имя поля, в котором в attribute_list находятся названия переменных
    
    - need_business - флажок True/False. Считать ли бизнес метрику. По умолчанию True
    
    - draw - флажок True/False. Рисовать ли картинки. По умолчанию True
    
    - draw_by_approval_rate - флажок True/False. Рисовать ли картинки по Approval_rate вместо treshold. По умолчанию False
    
    - simple_b_score - функция, которая расчитывает бизнес метрику
    
    - business_dict - словарь с параметрами для бизнес метрики. Пример для CRM:business_dictionary = {'t0': 0.1, 'm_s': 19000, 'fund': 1, 'k': 20, 'c': 3}
    
    - business_dic_sec - словарь с параметрами для бизнес метрики для второго таргета, если он используется
    
    - scale - делать ли стандартизацию. По дефолту True
    
    - median - импутация пропусков. Возможные принимаемые значения:
        - 'median' - тогда на train куске рассчитываются медианы, и импутация ими
        - 'min-1' - тогда на train куске рассчитываются минимальные значения - 1, и импутация ими
        - число - если задать число, то пропуски будут заполняться этим числом
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
        - None. Тогда пропуски не импутируются
        По дефолту задано значение 'median'
    - high_outlier - импутация выбросов вверх. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 99 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
        - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
        - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
        - None. Тогда выбросы вверх не импутируются.
    - low_outlier - импутация выбросов вниз. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 1 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
        - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
        - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
        - None. Тогда выбросы вниз не импутируются. 
        
    - task - задача регрессии или классификации. binar для задачи классификации, numeric - для задачи регрессии. По умолчанию binar
    - k_logs - как часто отображать результат. По умолчанию отображать каждую десятую итерацию
        
    ВАЖНО!!!!!!!!!!!! Для поиска по сетке не нужно подавать данные с импутированными пропусками/выбросами! Более того, не стоит 
    использовать attribute_list, так как на каждом разбиении должна считаться собственная импутация.
    
    random_state - random_state для биения данных на куски. random_state для модели следует подавать в словаре для обучения!!!!

    Возвращает таблицу meta с сеткой и показателями. 
    """
    
    def train_and_receive_stats_binar(model, xtrain, ytrain, xtest, ytest, scores, 
                                      second_target = None, y_train_2 = None, y_test_2 = None,
                                     draw = True, draw_by_approval_rate = False):
        model.fit(xtrain, ytrain)
        yhat_test = model.predict(xtest)
        
        yhat_test_proba = model.predict_proba(xtest)[:,1]
        yhat_train_proba = model.predict_proba(xtrain)[:,1]
        
        scores['ScoreF1'].append(metrics.f1_score(ytest, yhat_test))
        scores['Acc'].append(metrics.accuracy_score(ytest, yhat_test))
        scores['Pre'].append(metrics.precision_score(ytest, yhat_test))
        scores['Rec'].append(metrics.recall_score(ytest, yhat_test)) 
        scores['APS'].append(metrics.average_precision_score(ytest, yhat_test_proba))
        scores['Brier_score'].append(metrics.brier_score_loss(ytest, yhat_test_proba))
        scores['AUC'].append(metrics.roc_auc_score(ytest, yhat_test_proba))
        scores['Bad_Rate'].append(ytest.value_counts()[1]/len(ytest))
        # находим лучший cut-off по трейн и применяем его для тест!!
        #best_score_max, cut_off_max, best_score_thr, cut_off_thr
        
        if need_business == True:
            b_best_train_max, cutoff_train_max, b_best_max = b_score_train_and_test(ytrain,
                                                                        yhat_train_proba, ytest, 
                                                                    yhat_test_proba, simple_b_score, business_dict)
            if draw == True:
                b_score_array, approval_rate, cutoff, best_sc, best_cutoff = max_prof_corve(ytest, yhat_test_proba, simple_b_score,
                                                                                                    business_dict)
                b_score_array_train, approval_rate_train, cutoff_train, best_sc_train, best_cutoff_train = max_prof_corve(ytrain, 
                                                                                                                yhat_train_proba, 
                                                                                                                simple_b_score,
                                                                                                                business_dict)
                if draw_by_approval_rate == False:
                    x_plot = cutoff
                    y_plot = b_score_array
                    c = next(color) 
                    x_plot_train = cutoff_train
                    y_plot_train = b_score_array_train #/len(y_test)

                    if k/k_logs == int(k/k_logs) or k == 1:
                        ax_each.scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.1)
                        ax_each.scatter(x_plot_train, y_plot_train, s = 0.1, color=c, alpha=0.1)
                        ax_each.plot([best_cutoff_train, best_cutoff_train], [0, best_sc_train], '--', color=c, alpha=0.8)
                        ax_each.plot([best_cutoff, best_cutoff], [0, best_sc], '--', color=c, alpha=0.8)

                        #axs[k].scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.5, linewidths = 0.2)
                        #axs[k].tick_params(labelsize=2, which='both', labelbottom=True, labelleft=True, width = 0.2)

                    axs[k].scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.1, linewidth=0.2)
                    axs[k].scatter(x_plot_train, y_plot_train, s = 0.1, color=c, alpha=0.1, linewidth=0.2)
                    axs[k].plot([best_cutoff_train, best_cutoff_train], [0, best_sc_train], '--', linewidth=0.2, color=c, alpha=0.8)
                    axs[k].plot([best_cutoff, best_cutoff], [0, best_sc], '--', linewidth=0.2, color=c, alpha=0.8)
                    axs[k].tick_params(labelsize=2, which='both', labelbottom=True, labelleft=True, width = 0.2)

                else:
                    x_plot = approval_rate
                    y_plot = b_score_array
                    c = next(color) 
                    x_plot_train = approval_rate_train
                    y_plot_train = b_score_array_train #/len(y_test)

                    if k/k_logs == int(k/k_logs) or k == 1:
                        ax_each.scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.1)
                        ax_each.scatter(x_plot_train, y_plot_train, s = 0.1, color=c, alpha=0.1)

                    axs[k].scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.1, linewidth=0.2)
                    axs[k].scatter(x_plot_train, y_plot_train, s = 0.1, color=c, alpha=0.1, linewidth=0.2)
                    axs[k].tick_params(labelsize=2, which='both', labelbottom=True, labelleft=True, width = 0.2)

            scores['b_best'].append(b_best_max)
            scores['cutoff'].append(cutoff_train_max)
                                    
        if type(second_target) != type(None):
            
            scores['ScoreF1_second_target'].append(metrics.f1_score(y_test_2, yhat_test))
            scores['Acc_second_target'].append(metrics.accuracy_score(y_test_2, yhat_test))
            scores['Pre_second_target'].append(metrics.precision_score(y_test_2, yhat_test))
            scores['Rec_second_target'].append(metrics.recall_score(y_test_2, yhat_test)) 
            scores['APS_second_target'].append(metrics.average_precision_score(y_test_2, yhat_test_proba))
            scores['Brier_score_second_target'].append(metrics.brier_score_loss(y_test_2, yhat_test_proba))
            scores['AUC_second_target'].append(metrics.roc_auc_score(y_test_2, yhat_test_proba))
            scores['Bad_Rate_second_target'].append(y_test_2.value_counts()[1]/len(y_test_2))
            # находим лучший cut-off по трейн и применяем его для тест!!
            #best_score_max, cut_off_max, best_score_thr, cut_off_thr
                    
            if need_business == True:
                b_best_train_max, cutoff_train_max, b_best_max  = b_score_train_and_test(y_train_2, 
                                                            yhat_train_proba, y_test_2, yhat_test_proba, simple_b_score,
                                                                                     business_dict_sec)           

                scores['b_best_second_target'].append(b_best_max)
                scores['cutoff_second_target'].append(cutoff_train_max)
        
        if need_business == True:
            if draw == True:
                if draw_by_approval_rate == False:
                    if k/10 == int(k/10) or k == 1:
                        ax_each.set_xlabel('Treshold')
                        ax_each.set_ylabel('Profit')
                        ax_each.set_title(parameters)
                        plt.show()

                    axs[k].set_xlabel('Treshold', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_ylabel('Profit', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_title(parameters, fontdict = {'fontsize': 2, 'fontweight' : 2})

                else:
                    if k/10 == int(k/10) or k == 1:
                        ax_each.set_xlabel('Approval Rate')
                        ax_each.set_ylabel('Profit')
                        ax_each.set_title(parameters)
                        plt.show()

                    axs[k].set_xlabel('Approval Rate', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_ylabel('Profit', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_title(parameters, fontdict = {'fontsize': 2, 'fontweight' : 2})
        
        plt.close()
        
        return model, scores
    
    def train_and_receive_stats_numeric(model, xtrain, ytrain, xtest, ytest, 
                                        scores, second_target = None, y_train_2 = None, y_test_2 = None):
        
        model.fit(xtrain, ytrain)
        yhat_test = model.predict(xtest)
        
        scores['R2'].append(metrics.r2_score(ytest, yhat_test))
        scores['MSE'].append(metrics.mean_squared_error(ytest, yhat_test))
        scores['MAE'].append(metrics.mean_absolute_error(ytest, yhat_test))
        scores['MedianAE'].append(metrics.median_absolute_error(ytest, yhat_test))
        #scores['MSLE'].append(metrics.mean_squared_log_error(ytest, yhat_test))
        scores['RMSE'].append(np.sqrt(metrics.mean_squared_error(ytest, yhat_test)))
        #scores['RMSLE'].append(np.sqrt(metrics.mean_squared_log_error(ytest, yhat_test)))
            
        if type(second_target) != type(None):
		    # Исправлено
			# Исправлено
            scores['R2_second_target'].append(metrics.r2_score(y_test_2, yhat_test))
            scores['MSE_second_target'].append(metrics.mean_squared_error(y_test_2, yhat_test))
            scores['MAE_second_target'].append(metrics.mean_absolute_error(y_test_2, yhat_test))
            scores['MedianAE_second_target'].append(metrics.median_absolute_error(y_test_2, yhat_test))
            #scores['MSLE_second_target'].append(metrics.mean_squared_log_error(y_test_2, yhat_test))
            scores['RMSE_second_target'].append(np.sqrt(metrics.mean_squared_error(y_test_2, yhat_test)))
            #scores['RMSLE_second_target'].append(np.sqrt(metrics.mean_squared_log_error(y_test_2, yhat_test)))
            # находим лучший cut-off по трейн и применяем его для тест!!
            #best_score_max, cut_off_max, best_score_thr, cut_off_thr
            
        return model, scores
    
    def data_preprocessing_meta(X_1, y_1, X_2, y_2, technical_values, categorial_list = None, yeo_johnson = False, 
                                attribute_list = None, var_col = None, scale = 'mean', median = 'median',
                                high_outlier = None, low_outlier = None, cols_outlier = None, cut_non_out_9999 = True, 
                                check_percentile = 1):
    
      
        """
        Проводит препроцессинг для train и test выборки.

        X_1, y_1, X_2, y_2 - данные
        technical_values - список технических переменных
        technical_values исключены из анализа и удаляются из выборки. Если технических переменных нет, можно задать пустой список. 

        yeo_johnson - проводить ли нормализацию Йео-Джонсона (приведение распределения данных к нормальному
        виду). По дефолту False
        attribute_list - данные attribute_list. Могут быть None. По дефолту None
        var_col - в каком поле attribute_list находятся названия фичей. Если attribute_list не задан, то в var_col нет нужды.
        По дефолту None.
        scale - проводить ли стандартизацию StandardScaler. По дефолту True.

        median - импутация пропусков. Возможные принимаемые значения:
            - 'median' - тогда на train куске рассчитываются медианы, и импутация ими и train и test кусков
            - 'min-1' - тогда на train куске рассчитываются минимальные значения - 1, и импутация ими
            - число - если задать число, то пропуски будут заполняться этим числом
            - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
            - None. Тогда пропуски не импутируются
            По дефолту задано значение 'median'
        high_outlier - импутация выбросов вверх. Возможные принимаемые значения:
            - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 99 перцентиль
            - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
            - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
            - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
            - None. Тогда выбросы вверх не импутируются.
        low_outlier - импутация выбросов вниз. Возможные принимаемые значения:
            - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 1 перцентиль
            - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
            - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
            - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
            - None. Тогда выбросы вниз не импутируются. 

         Возвращает измененные данные и обученный Scaler(или Йео_Джонсон)

        """

        xtrain = X_1.copy()
        xtest = X_2.copy()
        ytrain = y_1.copy()
        ytest = y_2.copy()

        for i in technical_values:
            if i in xtrain.columns:
                xtrain.drop(i, axis = 1, inplace = True)
            if i in xtest.columns:
                xtest.drop(i, axis = 1, inplace= True)

        if type(categorial_list) != type(None):
            categorial_cols = categorial_list
            numeric_cols = list(xtrain.columns) 
            for i in categorial_cols:
                if i in numeric_cols:
                    numeric_cols.remove(i)
        else:
            categorial_cols = []
            for cc in xtrain.columns:
                if xtrain[cc].nunique() == 2:
                    if sorted(xtrain[cc].unique())[0] == 0 and sorted(xtrain[cc].unique())[1] == 1:
                        categorial_cols.append(cc) 
            numeric_cols = list(xtrain.columns) 
            for i in categorial_cols:
                if i in numeric_cols:
                    numeric_cols.remove(i)

        if type(cols_outlier) == type(None):
            cols_outlier = xtrain.columns
            
        test_ind = xtest.index

        for oo in xtrain.columns:
            if median != None:
                if median == 'median':
                    medians = xtrain[oo].median(skipna = True)
                elif median == 'min-1':
                    medians = xtrain[oo].min(skipna = True)-1
                elif type(attribute_list) != type(None) and median in attribute_list.columns:
                    medians = list(attribute_list.loc[attribute_list[var_col] == oo, median])[0]
                else:
                    medians = median

            if high_outlier != None:
                if oo in cols_outlier:
                    if type(attribute_list) != type(None) and high_outlier in attribute_list.columns:
                        to_replace_high = list(attribute_list.loc[attribute_list[var_col] == oo, high_outlier])[0]
                    elif high_outlier == 'IQR':
                        check_25 = np.nanpercentile(xtrain[oo], 25)
                        check_75 = np.nanpercentile(xtrain[oo], 75)
                        check_99 = np.nanpercentile(xtrain[oo], 100-check_percentile)
                        maximum = xtrain[oo].max()

                        if check_25 != check_75:
                            q_25 = np.nanpercentile(xtrain[oo], 25)
                            q_75 = np.nanpercentile(xtrain[oo], 75)
                            iqr = q_75-q_25
                            right_border = q_75+iqr*1.5
                        else:
                            x = xtrain.loc[xtrain[oo] != check_25, oo]
                            q_25 = np.nanpercentile(x, 25)
                            q_75 = np.nanpercentile(x, 75)
                            iqr = q_75-q_25
                            right_border = q_75+iqr*1.5

                        if right_border > maximum:
                            to_replace_high = maximum
                        elif check_99 > right_border:
                            to_replace_high = check_99
                        else:
                            to_replace_high = right_border
                        
                        
                    elif high_outlier == 'z-score':
                        to_replace_high = 3*xtrain[oo].std()+xtrain[oo].mean()
                    else:
                        to_replace_high = np.nanpercentile(xtrain[oo], high_outlier)
                elif oo not in cols_outlier:
                    if oo in numeric_cols:
                        if cut_non_out_9999 == True:
                            to_replace_high = np.nanpercentile(xtrain[oo], 99.99)
            elif high_outlier == None:
                to_replace_high = None

            if low_outlier != None:
                if oo in cols_outlier:
                    if type(attribute_list) != type(None) and low_outlier in attribute_list.columns:
                        to_replace_low = list(attribute_list.loc[attribute_list[var_col] == oo, low_outlier])[0]
                        
                    elif low_outlier == 'IQR':
                        check_25 = np.nanpercentile(xtrain[oo], 25)
                        check_75 = np.nanpercentile(xtrain[oo], 75)
                        check_1 = np.nanpercentile(xtrain[oo], check_percentile)
                        minimum = xtrain[oo].min()
                        if check_25 != check_75:
                            q_25 = np.nanpercentile(xtrain[oo], 25)
                            q_75 = np.nanpercentile(xtrain[oo], 75)
                            iqr = q_75-q_25
                            left_border = q_25-iqr*1.5
                        else:
                            x = xtrain.loc[xtrain[oo] != check_25, oo]
                            q_25 = np.nanpercentile(x, 25)
                            q_75 = np.nanpercentile(x, 75)
                            iqr = q_75-q_25
                            left_border = q_25-iqr*1.5

                        if left_border < minimum:
                            to_replace_low = minimum
                        elif check_1 < left_border:
                            to_replace_low = check_1
                        else:
                            to_replace_low = left_border
                        
                    elif low_outlier == 'z-score':
                        to_replace_low = (-1)*3*xtrain[oo].std()+xtrain[oo].mean()
                    else:
                        to_replace_low = np.nanpercentile(xtrain[oo], low_outlier)
                        
                elif oo not in cols_outlier:
                    if oo in numeric_cols:
                        to_replace_low = min(xtrain[oo])
            elif low_outlier == None:
                to_replace_low = None

            if median != None:
                xtrain[oo] = xtrain[oo].fillna(medians)
                xtest[oo] = xtest[oo].fillna(medians)
            if to_replace_high != None:
                if oo in cols_outlier:
                    xtrain.loc[xtrain[oo] > to_replace_high, oo] = to_replace_high
                    xtest.loc[xtest[oo] > to_replace_high, oo] = to_replace_high
            if to_replace_low != None:
                if oo in cols_outlier:
                    xtrain.loc[xtrain[oo] < to_replace_low, oo] = to_replace_low
                    xtest.loc[xtest[oo] < to_replace_low, oo] = to_replace_low


        if yeo_johnson == False and scale != False:
            if scale == 'mean':
                pr = preprocessing.StandardScaler()
                pr.fit(xtrain[numeric_cols])
                xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
                xtest[numeric_cols] = pr.transform(xtest[numeric_cols])
                return xtrain, xtest, ytrain, ytest
            elif scale == 'minmax':
                pr = preprocessing.MinMaxScaler()
                pr.fit(xtrain[numeric_cols])
                xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
                xtest[numeric_cols] = pr.transform(xtest[numeric_cols])

                return xtrain, xtest, ytrain, ytest

        elif yeo_johnson == True:
            power = PowerTransformer(method = 'yeo-johnson', standardize = False).fit(xtrain[numeric_cols])
            xtrain[numeric_cols] = power.transform(xtrain[numeric_cols])
            xtest[numeric_cols] = power.transform(xtest[numeric_cols])
            pr = power

            pr2 = preprocessing.StandardScaler()
            pr2.fit(xtrain[numeric_cols])
            xtrain[numeric_cols] = pr2.transform(xtrain[numeric_cols])
            xtest[numeric_cols] = pr2.transform(xtest[numeric_cols])

            return xtrain, xtest, ytrain, ytest
        else:
            return xtrain, xtest, ytrain, ytest
    
    def max_prof_corve(y_true, y_score, simple_score, business_dictionary, pos_label=None, sample_weight=None):
    
        """

        Создает векторы для отрисовки кривой max_profit.
        Пример использования:

        b_auc, tp_fps_auc, cut_auc, best_auc, cutoff_auc = max_prof_corve(y_2, auc_test_pred, simple_b_score_crm, business_dictionary)

        plt.figure(figsize = (10, 10))

        plt.title('Business score test')

        plt.plot(tp_fps_auc, b_auc, color='green',
                 lw=lw, label='maxProfit AUC model')

        """

        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score = b_score.max()
        cut_off = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
        #return  tns, fns, fps, tps, y_score[threshold_idxs]
        return b_score, tp_fp, y_score[threshold_idxs], best_score, cut_off
    
    def b_score_train_and_test(y_true, y_score, y_test, y_test_score, simple_score, business_dictionary, pos_label=None, 
                               sample_weight=None):
        """
        ----------
        y_true : array, shape = [n_samples]
            True targets of binary classification
        y_score : array, shape = [n_samples]
            Estimated probabilities or decision function
        y_test: True TEST targets
        y_test_score: predictions on target
        pos_label : int or str, default=None
            The label of the positive class
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        Returns

            !!!!ВАЖНО!!!!

            В функцию надо подавать как прогнозы и истинные метки трейна, так и прогнозы и истинные метки теста, так как на трейне
            рассчитывается оптимальный порог, по которому считается бизнес метрика, но итоговые результаты и выводы о бизнес метрике надо
            делать на ТЕСТЕ!

        """
        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps # Оставить, часть матрицы сопряженности. может пригодиться
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score_max = b_score.max()
        cut_off_max = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
        #idx_max = max(np.where(tp_fp <= t0)[0])
        #best_score_thr = b_score[idx_max]
        #cut_off_thr = y_score[threshold_idxs][idx_max]
        #return  tns, fns, fps, tps, y_score[threshold_idxs]

        y_best_test_max = pd.Series(np.where(y_test_score >= cut_off_max , 1, 0))
        #y_best_test_thr = pd.Series(np.where(y_test_score >= cut_off_thr  , 1, 0))

        _tn, _fp, _fn, _tp = metrics.confusion_matrix(y_test, y_best_test_max).ravel()
                    #m_s*fund*tps - c*k*tp_fp*y_true.size 
        b_best_max = simple_score(_tn = _tn, _fp = _fp, _fn = _fn, _tp = _tp, **business_dictionary) 

        return best_score_max, cut_off_max, b_best_max#, best_score_thr, cut_off_thr # b_score , tns, fns, fps, tps, y_score[threshold_idxs]
    
    data = X.join(Y)
    
    target = data.columns[-1]
    
    if type(cols_outlier) == type(None):
        cols_outlier = list(data.columns)
        
    for i in list_of_vars_for_strat:
        if i in cols_outlier:
            cols_outlier.remove(i)
    if target in cols_outlier:
        cols_outlier.remove(target)
    if sort_by_var in cols_outlier:
        cols_outlier.remove(sort_by_var)
    
    max_target = data.groupby(sort_by_var).aggregate({target: 'max'})
    max_target = max_target.reset_index()
    
    data = pd.merge(data, max_target, on = sort_by_var, suffixes = ["", "_max"])  
    
    target1 = target+"_max"
    
    list_of_vars_for_strat1 = list_of_vars_for_strat.copy()
    if task == 'binary':
        if len(list_of_vars_for_strat1) == 0:
            list_of_vars_for_strat1 = [target1]
        if target in list_of_vars_for_strat1:
            list_of_vars_for_strat1.remove(target)
            list_of_vars_for_strat1.append(target1)
        else:
            list_of_vars_for_strat1.append(target1)
                    
    for i in list_of_vars_for_strat1:
        if i == list_of_vars_for_strat1[0]:
            data['For_stratify'] = data[i].astype('str')
        else:
            data['For_stratify'] += data[i].astype('str')

    data_nodup = data[[sort_by_var, 'For_stratify', target1]].drop_duplicates(subset = sort_by_var)
    
    cross_val = StratifiedKFold(n_splits=n_folds, shuffle = True, random_state = random_state)
    meta_container = pd.DataFrame()
    meta_model = []
    
    for key in params_dictionary.keys():
        if type(params_dictionary[key]) != np.ndarray and type(params_dictionary[key]) != list:
            check = []
            check.append(params_dictionary[key])
            params_dictionary[key] = check
    
    print(params_dictionary)
    
    vals = params_dictionary.values()
    
    import itertools

    combs = list(itertools.product(*vals))
    
    k = 0
          
    if draw == True:
        fig, axs = plt.subplots(len(combs), 1, figsize=(2.2, len(combs)), sharey='all', sharex='all', 
                                constrained_layout=True)
        #fig = plt.figure(figsize = (5, len(combs)))
        fig.suptitle('Graphs of Max profit', fontsize=3)
        plt.close(fig)
    
    for combination in combs:
        outputs = list(combination)
        dicts = {}
        for position in range(len(combination)):
            dicts[list(params_dictionary.keys())[position]] = combination[position]
        
        parameters = {}
        
        for key in dicts.keys():
            if len(re.findall('weight', key)) == 0:
                parameters[params_to_model[key]] = dicts[key]
            elif len(re.findall('class_weight', key)) > 0:
                parameters[params_to_model[key]] = dicts[key]
            else:
                if len(re.findall('0', key)) > 0:
                    parameters[params_to_model[key]] = {0: dicts[key], 1:1}
                if len(re.findall('1', key)) > 0:
                    parameters[params_to_model[key]] = {0:1, 1:dicts[key]}

        if not callable(pass_model):
            return 'Error! Model should be callable'
        else:
            model = pass_model(**parameters)
            
        if type(second_target) != type(None) and task == 'binary':
            if need_business == False:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],    
                'AUC': [],
                'Bad_Rate': [],
                'ScoreF1_second_target': [],
                'Acc_second_target': [],
                'Pre_second_target': [],
                'Rec_second_target': [] ,
                'APS_second_target': [],
                'Brier_score_second_target': [],
                'AUC_second_target': [],
                'Bad_Rate_second_target': []
                }
                
            else:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],    
                'AUC': [],
                'b_best' : [] ,
                'cutoff' : [] ,
                'Bad_Rate': [],
                'ScoreF1_second_target': [],
                'Acc_second_target': [],
                'Pre_second_target': [],
                'Rec_second_target': [] ,
                'APS_second_target': [],
                'Brier_score_second_target': [],
                'AUC_second_target': [],
                'b_best_second_target' : [] ,
                'cutoff_second_target' : [] ,
                'Bad_Rate_second_target': []
                }
    
        elif type(second_target) == type(None) and task == 'binary':
            if need_business == True:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],
                'AUC': [],
                'b_best' : [] ,
                'cutoff' : [],
                'Bad_Rate': []
            }
            else:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],
                'AUC': [],
                'Bad_Rate': []
            }
                
        elif type(second_target) == type(None) and task == 'numeric':
            scores = {
                'R2': [],
                'MSE':[],
                'MAE':[],
                'MedianAE': [],
                #'MSLE': [],
                'RMSE': [],
                #'RMSLE': []
            }
        elif type(second_target) != type(None) and task == 'numeric':    
            
            scores = {
                'R2': [],
                'MSE':[],
                'MAE':[],
                'MedianAE': [],
                #'MSLE': [],
                'RMSE': [],
                #'RMSLE': [],
                'R2_second_target': [],
                'MSE_second_target':[],
                'MAE_second_target':[],
                'MedianAE_second_target': [],
                #'MSLE_second_target': [],
                'RMSE_second_target': [],
                #'RMSLE_second_target': []
            }
        
        
        if draw == True:
            color=iter(cm.rainbow(np.linspace(0, 1, n_folds))) # Оставить
            fig_each, ax_each = plt.subplots(1, 1, figsize=(10, 5))    
                
        for idx_train, idx_test in cross_val.split(data_nodup[sort_by_var], data_nodup['For_stratify']):
                
            xtrain_id, xtest_id = data_nodup.iloc[idx_train][sort_by_var], data_nodup.iloc[idx_test][sort_by_var]
            xtrain = data[data[sort_by_var].isin(xtrain_id)].copy()
            train_index = xtrain.index
            ytrain = data.iloc[train_index][target].copy()

            xtest = data[data[sort_by_var].isin(xtest_id)].copy()
            test_index = xtest.index
            ytest = data.iloc[test_index][target].copy()

            if type(second_target) != type(None):
                y_test_2 = data.iloc[test_index][second_target].copy()
            
            xtrain.drop(list_of_vars_for_strat1, axis = 1, inplace = True)
            xtrain.drop(sort_by_var, axis = 1, inplace = True)
            xtrain.drop(target, axis = 1, inplace = True)
            if target1 in xtrain.columns:
                xtrain.drop(target1, axis = 1, inplace = True)
            xtrain.drop('For_stratify', axis = 1, inplace = True)

            xtest.drop(list_of_vars_for_strat1, axis = 1, inplace = True)
            xtest.drop(sort_by_var, axis = 1, inplace = True)
            xtest.drop(target, axis = 1, inplace = True)
            if target1 in xtest.columns:
                xtest.drop(target1, axis = 1, inplace = True)
            xtest.drop('For_stratify', axis = 1, inplace = True)
            
            if type(second_target) != type(None):
                y_train_2 = xtrain[second_target]
                y_test_2 = xtest[second_target]
                xtrain.drop(second_target, axis = 1, inplace = True)
                xtest.drop(second_target, axis = 1, inplace = True)
                 
            test_с = xtest.columns
            test_ind = xtest.index
            
            xtrain, xtest, ytrain, ytest = data_preprocessing_meta(xtrain, ytrain, xtest, ytest, technical_values = [], 
                                                                   categorial_list = categorial_list, 
                                                                   yeo_johnson = yeo_johnson, attribute_list = attribute_list, 
                                                                   var_col = var_col, scale = scale, median = median,
                                                                   high_outlier = high_outlier, 
                                                                   low_outlier = low_outlier, 
                                                                   check_percentile = check_percentile, 
                                                                   cols_outlier = cols_outlier,
                                                                   cut_non_out_9999 = cut_non_out_9999)
            
            if task == 'binary':
                model, scores = train_and_receive_stats_binar(model, xtrain, ytrain, xtest, ytest, 
                                                              scores, second_target = second_target, 
                                                              y_train_2 = None, y_test_2 = None, draw = draw, 
                                                              draw_by_approval_rate = draw_by_approval_rate)
            elif task == 'numeric': 
                model, scores = train_and_receive_stats_numeric(model, xtrain, ytrain, xtest, ytest, 
                                                                scores, second_target = second_target, 
                                                              y_train_2 = None, y_test_2 = None)
            
        scores = pd.DataFrame(scores)
        outputs.extend(scores.mean().tolist())
        outputs.extend(scores.std().tolist())
        score_cols = scores.columns.tolist()
        score_cols_std = [c+'_std' for c in score_cols]
        cols = list(params_dictionary.keys()) + score_cols + score_cols_std
        outputs = pd.DataFrame([outputs], columns=cols)
        meta_container = meta_container.append(outputs)
        meta_model.append(model)
        #print(k)
        if k/k_logs == int(k/k_logs) or k == 1:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
                
            if need_business == True and task == 'binary':
                print(20*'-',tm, k, 20*'-', '\n',
                  'Параметры:', parameters, '\n', 'Среднее значение бизнес метрики =',
                  scores['b_best'].mean(), '\n', 'Среднее значение AUC =', scores['AUC'].mean())
            elif need_business == False and task == 'binary':
                print(20*'-',tm, k, 20*'-', '\n',
                  'Параметры:', parameters, '\n', 'Среднее значение APS =',
                  scores['APS'].mean(), '\n', 'Среднее значение AUC =', scores['AUC'].mean())
            elif task == 'numeric':
                print(20*'-',tm, k, 20*'-', '\n',
                  'Параметры:', parameters, '\n', 'Среднее значение R2 =',
                  scores['R2'].mean())

        k += 1
                      
    meta_container.reset_index(drop=True, inplace=True)
    if draw == True:
        fig.savefig('All_Max_Profit.png', dpi = 300)
    if task == 'binary':
        return meta_container
    elif task == 'numeric':
        return meta_container, meta_model

# In[ ]:


def data_preprocessing(X_1, y_1, X_2, y_2, technical_values, categorial_list = None, yeo_johnson = False, attribute_list = None, 
                       var_col = None,
                       scale = 'mean', median = 'median',
                      high_outlier = None, low_outlier = None, cols_outlier = None, check_percentile = 1, cut_non_out_9999 = True):
    
      
    """
    Проводит препроцессинг для train и test выборки.
    
    X_1, y_1, X_2, y_2 - данные
    technical_values - список технических переменных
    technical_values исключены из анализа и удаляются из выборки. Если технических переменных нет, можно задать пустой список. 
    
    yeo_johnson - проводить ли нормализацию Йео-Джонсона (приведение распределения данных к нормальному
    виду). По дефолту False
    attribute_list - данные attribute_list. Могут быть None. По дефолту None
    var_col - в каком поле attribute_list находятся названия фичей. Если attribute_list не задан, то в var_col нет нужды.
    По дефолту None.
    scale - проводить ли стандартизацию StandardScaler. По дефолту True.
    
    median - импутация пропусков. Возможные принимаемые значения:
        - 'median' - тогда на train куске рассчитываются медианы, и импутация ими и train и test кусков
        - 'min-1' - тогда на train куске рассчитываются минимальные значения - 1, и импутация ими
        - число - если задать число, то пропуски будут заполняться этим числом
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
        - None. Тогда пропуски не импутируются
        По дефолту задано значение 'median'
    high_outlier - импутация выбросов вверх. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 99 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
        - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
        - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
        - None. Тогда выбросы вверх не импутируются.
    low_outlier - импутация выбросов вниз. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 1 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
        - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
        - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
        - None. Тогда выбросы вниз не импутируются. 
     
     Возвращает измененные данные и обученный Scaler(или Йео_Джонсон)
     
    """
    
    xtrain = X_1.copy()
    xtest = X_2.copy()
    ytrain = y_1.copy()
    ytest = y_2.copy()
    
    for i in technical_values:
        if i in xtrain.columns:
            xtrain.drop(i, axis = 1, inplace = True)
        if i in xtest.columns:
            xtest.drop(i, axis = 1, inplace= True)
    
    if type(categorial_list) != type(None):
        categorial_cols = categorial_list
        numeric_cols = list(xtrain.columns) 
        for i in categorial_cols:
            if i in numeric_cols:
                numeric_cols.remove(i)
    else:
        categorial_cols = []
        for cc in xtrain.columns:
            if xtrain[cc].nunique() == 2:
                if sorted(xtrain[cc].unique())[0] == 0 and sorted(xtrain[cc].unique())[1] == 1:
                    categorial_cols.append(cc) 
        numeric_cols = list(xtrain.columns) 
        for i in categorial_cols:
            if i in numeric_cols:
                numeric_cols.remove(i)
    
    if type(cols_outlier) == type(None):
        cols_outlier = xtrain.columns
    
    train_c = xtrain.columns
    test_c = xtrain.columns
    train_ind = xtrain.index
    test_ind = xtest.index
                
    for oo in tqdm(numeric_cols):
        if median != None:
            if median == 'median':
                medians = xtrain[oo].median(skipna = True)
            elif median == 'min-1':
                medians = xtrain[oo].min(skipna = True)-1
            elif type(attribute_list) != type(None) and median in attribute_list.columns:
                medians = list(attribute_list.loc[attribute_list[var_col] == oo, median])[0]
            else:
                medians = median
                
        if high_outlier != None:
            if oo in cols_outlier:
                if type(attribute_list) != type(None) and high_outlier in attribute_list.columns:
                    to_replace_high = list(attribute_list.loc[attribute_list[var_col] == oo, high_outlier])[0]
                elif high_outlier == 'IQR':
                    check_25 = np.nanpercentile(xtrain[oo], 25)
                    check_75 = np.nanpercentile(xtrain[oo], 75)
                    check_99 = np.nanpercentile(xtrain[oo], 100-check_percentile)
                    maximum = xtrain[oo].max()
                    
                    if check_25 != check_75:
                        q_25 = np.nanpercentile(xtrain[oo], 25)
                        q_75 = np.nanpercentile(xtrain[oo], 75)
                        iqr = q_75-q_25
                        right_border = q_75+iqr*1.5
                    else:
                        x = xtrain.loc[xtrain[oo] != check_25, oo]
                        q_25 = np.nanpercentile(x, 25)
                        q_75 = np.nanpercentile(x, 75)
                        iqr = q_75-q_25
                        right_border = q_75+iqr*1.5
                    
                    if right_border > maximum:
                        to_replace_high = maximum
                    elif check_99 > right_border:
                        to_replace_high = check_99
                    else:
                        to_replace_high = right_border
                        
                elif high_outlier == 'z-score':
                    to_replace_high = 3*xtrain[oo].std()+xtrain[oo].mean()
                else:
                    to_replace_high = np.nanpercentile(xtrain[oo], high_outlier)
                    
            elif oo not in cols_outlier: 
                if oo in numeric_cols:
                    if cut_non_out_9999 == True:
                        to_replace_high = np.nanpercentile(xtrain[oo], 99.99)
                    else:
                        to_replace_high = max(xtrain[oo])
                else:
                    to_replace_high = max(xtrain[oo])
                    
        elif high_outlier == None:
            to_replace_high = None

        if low_outlier != None:
            if oo in cols_outlier:
                if type(attribute_list) != type(None) and low_outlier in attribute_list.columns:
                    to_replace_low = list(attribute_list.loc[attribute_list[var_col] == oo, low_outlier])[0]
                    
                elif low_outlier == 'IQR':
                        check_25 = np.nanpercentile(xtrain[oo], 25)
                        check_75 = np.nanpercentile(xtrain[oo], 75)
                        check_1 = np.nanpercentile(xtrain[oo], check_percentile)
                        minimum = xtrain[oo].min()
                        if check_25 != check_75:
                            q_25 = np.nanpercentile(xtrain[oo], 25)
                            q_75 = np.nanpercentile(xtrain[oo], 75)
                            iqr = q_75-q_25
                            left_border = q_25-iqr*1.5
                        else:
                            x = xtrain.loc[xtrain[oo] != check_25, oo]
                            q_25 = np.nanpercentile(x, 25)
                            q_75 = np.nanpercentile(x, 75)
                            iqr = q_75-q_25
                            left_border = q_25-iqr*1.5

                        if left_border < minimum:
                            to_replace_low = minimum
                        elif check_1 < left_border:
                            to_replace_low = check_1
                        else:
                            to_replace_low = left_border
                            
                elif low_outlier == 'z-score':
                    to_replace_low = (-1)*3*xtrain[oo].std()+xtrain[oo].mean()
                else:
                    to_replace_low = np.nanpercentile(xtrain[oo], low_outlier)
                    
            elif oo not in cols_outlier:
                to_replace_low = min(xtrain[oo])
        elif low_outlier == None:
            to_replace_low = None
                        
        if median != None:
            xtrain[oo] = xtrain[oo].fillna(medians)
            xtest[oo] = xtest[oo].fillna(medians)
        if to_replace_high != None:
            if oo in cols_outlier:
                xtrain.loc[xtrain[oo] > to_replace_high, oo] = to_replace_high
                xtest.loc[xtest[oo] > to_replace_high, oo] = to_replace_high
        if to_replace_low != None:
            if oo in cols_outlier:
                xtrain.loc[xtrain[oo] < to_replace_low, oo] = to_replace_low
                xtest.loc[xtest[oo] < to_replace_low, oo] = to_replace_low
    
    
    if yeo_johnson == False and scale != False:
        if scale == 'mean':
            pr = preprocessing.StandardScaler()
            pr.fit(xtrain[numeric_cols])
            xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
            xtest[numeric_cols] = pr.transform(xtest[numeric_cols])
            return xtrain, xtest, ytrain, ytest, pr
        elif scale == 'minmax':
            pr = preprocessing.MinMaxScaler()
            pr.fit(xtrain[numeric_cols])
            xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
            xtest[numeric_cols] = pr.transform(xtest[numeric_cols])
            return xtrain, xtest, ytrain, ytest, pr
        
    elif yeo_johnson == True:
        power = PowerTransformer(method = 'yeo-johnson', standardize = False).fit(xtrain[numeric_cols])
        xtrain[numeric_cols] = power.transform(xtrain[numeric_cols])
        xtest[numeric_cols] = power.transform(xtest[numeric_cols])
        pr = power
            
        pr2 = preprocessing.StandardScaler()
        pr2.fit(xtrain[numeric_cols])
        xtrain[numeric_cols] = pr2.transform(xtrain[numeric_cols])
        xtest[numeric_cols] = pr2.transform(xtest[numeric_cols])
                
        return xtrain, xtest, ytrain, ytest, pr, pr2
    else:
        return xtrain, xtest, ytrain, ytest


# In[ ]:


def train_model_receive_stats(X_1, y_1, X_2, y_2, meta, by_var, params_dict, other_hyperparams, pass_model,
                             need_business = True, simple_b_score = None, business_dict = None, printed= True,
                             task = 'binary'):
    
    """
    Обучает модель на трейн данных и получает статистики на тест данных
    
    X_1, y_1, X_2, y_2 - данные трейн и тест
    meta - файл с поиском по сетке
    by_var - по какому показателю выбираем гиперпараметры? ('AUC', 'b_best' - берется из сетки)
    params_dict - параметры, которые отбираются и их соответствие классическим. {'weight_0':'class_weight', 'C': 'C', 'max_iter': 'max_iter'}
    other_hyperparams - гиперпараметры, которые зафиксированы насильственно. 
    Пример: {'max_iter': 300, 'penalty': 'l2', 'random_state': 241, 'solver': 'liblinear' }
    pass_model - модель, которую обучаем
    need_business - считать ли бизнес метрику True/False. По умолчанию True
    simple_b_score - функция бизнес метрики, которая считает оптимальный порог на трейне и возвращает на тесте значение бизнес метрики
    для этого порога
    business_dict - словарь гиперпараметров для бизнес метрики
    printed - отображать ли результат? по умолчанию True
    
    """
    
    
    def max_prof_corve(y_true, y_score, simple_score, business_dictionary, pos_label=None, sample_weight=None):
    
        """

        Создает векторы для отрисовки кривой max_profit.
        Пример использования:

        b_auc, tp_fps_auc, cut_auc, best_auc, cutoff_auc = max_prof_corve(y_2, auc_test_pred, simple_b_score_crm, business_dictionary)

        plt.figure(figsize = (10, 10))

        plt.title('Business score test')

        plt.plot(tp_fps_auc, b_auc, color='green',
                 lw=lw, label='maxProfit AUC model')

        """

        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score = b_score.max()
        cut_off = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
        #return  tns, fns, fps, tps, y_score[threshold_idxs]
        return b_score, tp_fp, y_score[threshold_idxs], best_score, cut_off
    
    def b_score_train_and_test(y_true, y_score, y_test, y_test_score, simple_score, business_dictionary, pos_label=None, 
                               sample_weight=None):
        """
        ----------
        y_true : array, shape = [n_samples]
            True targets of binary classification
        y_score : array, shape = [n_samples]
            Estimated probabilities or decision function
        y_test: True TEST targets
        y_test_score: predictions on target
        pos_label : int or str, default=None
            The label of the positive class
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        Returns

            !!!!ВАЖНО!!!!

            В функцию надо подавать как прогнозы и истинные метки трейна, так и прогнозы и истинные метки теста, так как на трейне
            рассчитывается оптимальный порог, по которому считается бизнес метрика, но итоговые результаты и выводы о бизнес метрике надо
            делать на ТЕСТЕ!

        """
        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score_max = b_score.max()
        cut_off_max = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
        #idx_max = max(np.where(tp_fp <= t0)[0])
        #best_score_thr = b_score[idx_max]
        #cut_off_thr = y_score[threshold_idxs][idx_max]
        #return  tns, fns, fps, tps, y_score[threshold_idxs]

        y_best_test_max = pd.Series(np.where(y_test_score >= cut_off_max , 1, 0))
        #y_best_test_thr = pd.Series(np.where(y_test_score >= cut_off_thr  , 1, 0))

        _tn, _fp, _fn, _tp = metrics.confusion_matrix(y_test, y_best_test_max).ravel()
                    #m_s*fund*tps - c*k*tp_fp*y_true.size 
        b_best_max = simple_score(_tn = _tn, _fp = _fp, _fn = _fn, _tp = _tp, **business_dictionary) 

        return best_score_max, cut_off_max, b_best_max#, best_score_thr, cut_off_thr # b_score , tns, fns, fps, tps, y_score[threshold_idxs]
    
    params = pd.DataFrame(meta.sort_values(by = by_var, 
                                    ascending = False).iloc[0][list(params_dict.keys())]).T.reset_index()
    
    new_dict = {}

    for i in params_dict.keys():
        if len(re.findall('weight', i)) == 0:
            new_dict[params_dict[i]] = params[i][0]
        elif len(re.findall('class_weight', i)) > 0:
            if type(params[i][0]) is not type(dict):
                import ast
                params[i][0]=ast.literal_eval(params[i][0])
            new_dict[params_dict[i]] = params[i][0]
        else:
            if len(re.findall('0', i)) > 0:
                new_dict['class_weight'] = {0: params[i][0], 1:1}
            elif len(re.findall('1', i)) > 0:
                new_dict['class_weight'] = {0:1, 1: params[i][0]}
    
    if other_hyperparams != None:
        new_dict = {**new_dict, **other_hyperparams}
    
    print(new_dict)
    
    model_to_train = pass_model(**new_dict)
        
    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")   
    print(tm, 'start of logit fit')
    model_to_train.fit(X_1, y_1)
    
    if task == 'binary':
        pred_proba_train = model_to_train.predict_proba(X_1)[:, 1]
    elif task == 'numeric':
        pred_train = model_to_train.predict(X_1)
    
    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print(tm, 'end of logit fit')
    
    if task == 'binary':
        pred_proba = model_to_train.predict_proba(X_2)[:, 1]
    elif task == 'numeric':
        pred = model_to_train.predict(X_2)
    
    if task == 'binary' and need_business ==  True:
        b_best_train, cutoff, b_best_max = b_score_train_and_test(y_1, pred_proba_train, y_2, pred_proba, simple_b_score, 
                                                                  business_dict)
    
        print('Business metric on train = ', b_best_train)
    
    if task == 'binary':
        p_f_score, r_f_score, treshold_f_score = metrics.precision_recall_curve(y_1, pred_proba_train)
        new_elements  = np.r_[0, treshold_f_score]
        f_score = (2*p_f_score*r_f_score)/(p_f_score+r_f_score)
        f_score = np.where(np.isnan(f_score), 0, f_score)

        cut_f = new_elements[np.argmax(f_score)]

        if need_business ==  True:
            pred = np.where(pred_proba>=cutoff, 1, 0)
        pred_f = np.where(pred_proba>=cut_f, 1, 0)
    
    
    if need_business == True and task == 'binary':
        scores = {
                'ScoreF1_b_best': [],
                'ScoreF1_f_score': [],
                'Acc_b_best': [],
                'Pre_b_best': [],
                'Rec_b_best': [] ,
                'Acc_f_score': [],
                'Pre_f_score': [],
                'Rec_f_score': [] , 
                'Approval_rate_b_best': [],
                'Approval_rate_f_score': [],
                'APS': [],
                'Brier_score': [],
                'AUC': [],
                'AUC train': [],
                'Gini': [],
                'Gini train':[],
                'b_best' : []
                }
        
    elif need_business == False and task == 'binary':
        scores = {
                'ScoreF1_f_score': [],
                'Acc_f_score': [],
                'Pre_f_score': [],
                'Rec_f_score': [] , 
                'Approval_rate_f_score': [],
                'APS': [],
                'Brier_score': [],
                'AUC': [],
                'AUC train': [],
                'Gini': [],
                'Gini train':[],
                }
        
    elif task == 'numeric':
        scores = {
                'R2': [],
                'MSE':[],
                'MAE':[],
                'MedianAE': [],
                #'MSLE': [],
                'RMSE': [],
                #'RMSLE': []
            }
    
    tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print(tm)
    print(20*'-', 'Statistics', 20*'-')
    
    if task == 'binary':
        if need_business ==  True:
            scores['Approval_rate_b_best'].append(sum(pred)/y_2.shape[0])
            scores['Acc_b_best'].append(metrics.accuracy_score(y_2, pred))
            scores['ScoreF1_b_best'].append(metrics.f1_score(y_2, pred))
            scores['Pre_b_best'].append(metrics.precision_score(y_2, pred))
            scores['Rec_b_best'].append(metrics.recall_score(y_2, pred))

        scores['Approval_rate_f_score'].append(sum(pred_f)/y_2.shape[0])
        scores['Acc_f_score'].append(metrics.accuracy_score(y_2, pred_f))
        scores['ScoreF1_f_score'].append(metrics.f1_score(y_2, pred_f))
        scores['Pre_f_score'].append(metrics.precision_score(y_2, pred_f))
        scores['Rec_f_score'].append(metrics.recall_score(y_2, pred_f))

        scores['APS'].append(metrics.average_precision_score(y_2, pred_proba))
        scores['Brier_score'].append(metrics.brier_score_loss(y_2, pred_proba))
        scores['AUC'].append(metrics.roc_auc_score(y_2, pred_proba))
        scores['AUC train'].append(metrics.roc_auc_score(y_1, pred_proba_train))
        scores['Gini'].append(metrics.roc_auc_score(y_2, pred_proba)*2-1)
        scores['Gini train'].append(metrics.roc_auc_score(y_1, pred_proba_train)*2-1)
    
        if need_business ==  True:
            _tn, _fp, _fn, _tp = metrics.confusion_matrix(y_2, pred_f).ravel()
            b_best_f = simple_b_score(_tn, _fp, _fn, _tp, **business_dict )

            scores['b_best'].append(b_best_max)
    
    
        if printed == True:
            print('Brier score = ', metrics.brier_score_loss(y_2, pred_proba))
            print(20*'-')
            print('AUC = ', metrics.roc_auc_score(y_2, pred_proba))
            print('AUC train =', metrics.roc_auc_score(y_1, pred_proba_train))

            print('Gini = ', metrics.roc_auc_score(y_2, pred_proba)*2-1)
            print('Gini train =', metrics.roc_auc_score(y_1, pred_proba_train)*2-1)

            print('Stability =', (metrics.roc_auc_score(y_2, pred_proba)*2-1)/(metrics.roc_auc_score(y_1, 
                                                                                                     pred_proba_train)*2-1))
            print('Average precision score = ', metrics.average_precision_score(y_2, pred_proba))

            if need_business ==  True:
                print(10*'-', 'Treshold by Business score', 10*'-')

                print('Accuracy = ', metrics.accuracy_score(y_2, pred))
                print('Score F1 = ', metrics.f1_score(y_2, pred))
                print('Precision = ', metrics.precision_score(y_2, pred))
                print('Recall = ', metrics.recall_score(y_2, pred))
                print('Approval rate = ', sum(pred)/y_2.shape[0])
                print('Business score on test = ', b_best_max)

            print(10*'-', 'Treshold by F_score', 10*'-')

            print('Accuracy = ', metrics.accuracy_score(y_2, pred_f))
            print('Score F1 = ', metrics.f1_score(y_2, pred_f))
            print('Precision = ', metrics.precision_score(y_2, pred_f))
            print('Recall = ', metrics.recall_score(y_2, pred_f))
            print('Approval rate = ', sum(pred_f)/y_2.shape[0])
            if need_business ==  True:
                print('Business score on test = ', b_best_f)
    
    if task == 'numeric':
        # Исправлено
        scores['R2'].append(metrics.r2_score(y_2, pred))
        scores['MSE'].append(metrics.mean_squared_error(y_2, pred))
        scores['MAE'].append(metrics.mean_absolute_error(y_2, pred))
        scores['MedianAE'].append(metrics.median_absolute_error(y_2, pred))
        #scores['MSLE'].append(metrics.mean_squared_log_error(y_2, pred))
        scores['RMSE'].append(np.sqrt(metrics.mean_squared_error(y_2, pred)))
        #scores['RMSLE'].append(np.sqrt(metrics.mean_squared_log_error(y_2, pred)))
    
        if printed == True:
            print('R2 score = ', metrics.r2_score(y_2, pred))
            print('R2 score train = ', metrics.r2_score(y_1, pred_train))
            print('Stability =', (metrics.r2_score(y_2, pred)*2-1)/(metrics.r2_score(y_1, pred_train)*2-1))
            
            print(20*'-')
            print('MSE = ', metrics.mean_squared_error(y_2, pred))
            print('MSE train =', metrics.mean_squared_error(y_1, pred_train))

            print('MedianAE = ', metrics.median_absolute_error(y_2, pred)*2-1)
            print('MedianAE train =', metrics.median_absolute_error(y_1, pred_train)*2-1)

    if task == 'binary':
        return model_to_train, scores, pred_proba
    elif task == 'numeric':
        return model_to_train, scores, pred


# In[ ]:


def plot_meta_2d(meta, first_dimention, second_dimention, b_best):
    
    """
    
    Рисует тепловую карту для meta. First_dimention, second_dimention - поля, в которых находятся параметры, которые хотим отрисовать.
    В любом из этим параметров можно задать список полей!
    b_best - метрика, по которой рисуется тепловая карта
    
    """
    
    #b_best - назание поля с бизнес-метрикой в матрице meta
    score = pd.DataFrame()
    
    if type(first_dimention) == str:
        score[first_dimention] = meta[first_dimention]
        first_d = first_dimention
    
    elif type(first_dimention) == list and len(first_dimention) == 1:
        score[first_dimention[0]] = meta[first_dimention[0]]
        first_d = first_dimention[0]
    
    elif type(first_dimention) == list and len(first_dimention) > 1:
        
        first_d = reduce(lambda right, left: str(right) + ' & ' 
                                +str(left), first_dimention)
        score[first_d] = reduce(lambda right, left: str(right) + '= ' +meta[right].astype('str') + ' '
                                +str(left)+'= '+meta[left].astype('str'), first_dimention)
        
    if type(second_dimention) == str:
        score[second_dimention] = meta[second_dimention]
        sec_d = second_dimention
    
    elif type(second_dimention) == list and len(second_dimention) == 1:
        score[second_dimention[0]] = meta[second_dimention[0]]
        sec_d = second_dimention[0]
        
    elif type(second_dimention) == list and len(second_dimention) > 1:
        sec_d = reduce(lambda right, left: str(right) + ' & ' 
                                +str(left), second_dimention)
        score[sec_d] = reduce(lambda right, left: str(right) + '= ' +meta[right].astype('str') + ' '
                                +str(left)+'= '+meta[left].astype('str'), second_dimention)    
        
    score['b_best'] = meta[b_best] 
    
    first_list = sorted(score[first_d].unique())
    sec_list = sorted(score[sec_d].unique())
    
    f_d = score[score.b_best ==score.b_best.max()][first_d].to_list()
    s_d = score[score.b_best ==score.b_best.max()][sec_d].to_list()
    # score.drop_duplicates(subset=['first_d','sec_d'], inplace=True)
    df = score.pivot(index=first_d, columns=sec_d, values='b_best')
    df = df.values
    plt.figure(figsize=(len(sec_list), len(first_list)))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    #'nearest' 'bilinear'
    im = plt.imshow(df, cmap=plt.cm.viridis)
    plt.xlabel(sec_d)
    plt.ylabel(first_d)
    plt.colorbar(im,fraction=0.02, pad=0.04)
    #plt.scatter(reg ,wei , s=40, color='black', marker="D")
    
    plt.xticks(np.arange(len(sec_list)), sec_list)
    plt.yticks(np.arange(len(first_list)), first_list)
    
    i, j = np.where(df == score.b_best.max())
    i = i[0]
    j = j[0]
    plt.scatter(j , i , s=40, color='black', marker="x")
    text = plt.text(j, i, round(df[i, j] , 2), ha="center", va="top", color="black", fontsize=12)
    plt.title('Grid Search' + b_best)
    #plt.savefig('outputs/meta_{}.png'.format(logit))
    return plt.show()


# In[ ]:


def plot_meta_mosaic(meta, first_dimention, second_dimention, third_dimention, b_best, row_length =5):
    
    """
    Рисует "мозаику" тепловых карт.
    meta - мета контейнер
    first_dimention, second_dimention, third_dimention - три измерения (три гиперпараметра), по которым рисуется тепловая карта
    row_length - сколько графиков в одной строке?
    По третьему строится мозаика.
    
    """
    
    score = pd.DataFrame()
    
    if type(first_dimention) == str:
        score[first_dimention] = meta[first_dimention]
        first_d = first_dimention
    
    elif type(first_dimention) == list and len(first_dimention) == 1:
        score[first_dimention[0]] = meta[first_dimention[0]]
        first_d = first_dimention[0]
    
    elif type(first_dimention) == list and len(first_dimention) > 1:
        
        first_d = reduce(lambda right, left: str(right) + ' & ' 
                                +str(left), first_dimention)
        score[first_d] = reduce(lambda right, left: str(right) + '= ' +meta[right].astype('str') + ' '
                                +str(left)+'= '+meta[left].astype('str'), first_dimention)
        
    if type(second_dimention) == str:
        score[second_dimention] = meta[second_dimention]
        sec_d = second_dimention
    
    elif type(second_dimention) == list and len(second_dimention) == 1:
        score[second_dimention[0]] = meta[second_dimention[0]]
        sec_d = second_dimention[0]
        
    elif type(second_dimention) == list and len(second_dimention) > 1:
        sec_d = reduce(lambda right, left: str(right) + ' & ' 
                                +str(left), second_dimention)
        score[sec_d] = reduce(lambda right, left: str(right) + '= ' +meta[right].astype('str') + ' '
                                +str(left)+'= '+meta[left].astype('str'), second_dimention)    
        
        
    if type(third_dimention) == str:
        score[third_dimention] = meta[third_dimention]
        third_d = third_dimention
    
    elif type(third_dimention) == list and len(third_dimention) == 1:
        score[third_dimention[0]] = meta[third_dimention[0]]
        third_d = third_dimention[0]
        
    elif type(third_dimention) == list and len(third_dimention) > 1:
        third_d = reduce(lambda right, left: str(right) + ' & ' 
                                +str(left), third_dimention)
        score[third_d] = reduce(lambda right, left: str(right) + '= ' +meta[right].astype('str') + ' '
                                +str(left)+'= '+meta[left].astype('str'), third_dimention)
        
    score['b_best'] = meta[b_best] 
    
    first_list = sorted(score[first_d].unique())
    sec_list = sorted(score[sec_d].unique()) 
    third_list = sorted(score[third_d].unique()) 
        
    if len(third_list)/row_length == int(len(third_list)/row_length):
        
        n_rows = int(len(third_list)/row_length)
    
        fig, axes = plt.subplots(n_rows,
                             row_length, figsize = (row_length*7, n_rows*7), dpi = 100, 
                             sharex = True, sharey = True)
        
    else:
        n_rows = int(len(third_list)/2)+1
        fig, axes = plt.subplots(n_rows,
                             row_length, figsize = (row_length*7, n_rows*7), dpi = 100, 
                             sharex = True, sharey = True)
    
    for i, v in enumerate(third_list):
        
        score_new = score[score[third_d] == v].copy()
        df = score_new.pivot(index=first_d, columns=sec_d, values='b_best')
        df = df.values
                
        axes[i].imshow(df, cmap=plt.cm.viridis)
        axes[i].set_xlabel(sec_d)
        axes[i].set_ylabel(first_d)
        axes[i].set_title([third_d + ' = ' + str(v)])
                
                
        plt.xticks(np.arange(len(sec_list)), sec_list)
        plt.yticks(np.arange(len(first_list)), first_list)
        
        x, y = np.where(df == score_new.b_best.max())
        x = x[0]
        y = y[0]
        axes[i].scatter(y, x , s=40, color='black', marker="x")
        text = axes[i].text(y, x, round(df[x, y] , 6), ha="center", va="top", color="black", fontsize=12)                                                              
                                                                            
    return plt.show()


# In[ ]:


def by_month_gini(model, time_period, X, y, good_bad_dict):
    
    """
    Бьет выборку по месяцам и считает помесячные значение Gini.
    model - модель
    time_period - поле, в котором находятся временные периоды
    X, y - данные (полностью подготовленные) и таргет
    good_bad_dict - словарь вида {'good': 1, 'bad': 0}, нужен для определения, на основании чего считать good_rate и bad_rate статистики
    
    """
    
    X.reset_index(inplace = True)
    X.drop('index', axis = 1, inplace = True)
    target = y.name
    y_new = y.reset_index()[target]
    
    time_periods = sorted(X[time_period].unique())
    
    scores = []
       
    
    for i in time_periods:
        X_month = X[X[time_period] == i].copy()
        X_month.drop(time_period, axis =1, inplace = True)
        X_index = X_month.index
        y_month = y.iloc[X_index]
        
        prediction = model.predict_proba(X_month)[:, 1]
        if y_month.sum()>0:
            gini = metrics.roc_auc_score(y_month, prediction)*2-1
        else:
            gini = np.nan
        bad_rate = len(y_month[y_month == good_bad_dict['bad']])/len(y_month)
        good_rate = len(y_month[y_month == good_bad_dict['good']])/len(y_month)
        scores.append([i, len(y_month), bad_rate, good_rate, gini])
        
    col_names = ['month_call', 'number', 'bad_rate', 'good_rate', 'Gini']
    scores = pd.DataFrame.from_records(scores, columns = col_names)
    return scores


# In[ ]:


def receive_hyperparams(meta, by_var, params_dict, other_hyperparams):
    
    """
    Для бутстрэпа. Объединяет зафиксированные параметры (other_hyperparams) и отбираемые (params_dict).    
    
    """
    
    params = pd.DataFrame(meta.sort_values(by = by_var, 
                                    ascending = False).iloc[0][list(params_dict.keys())]).T.reset_index()
    
    new_dict = {}

    for i in params_dict.keys():
        if len(re.findall('weight', i)) == 0:
            new_dict[params_dict[i]] = params[i][0]
        elif len(re.findall('class_weight', i)) > 0:
            new_dict[params_dict[i]] = params[i][0]
        else:
            if len(re.findall('0', i)) > 0:
                new_dict['class_weight'] = {0: params[i][0], 1:1}
            elif len(re.findall('1', i)) > 0:
                new_dict['class_weight'] = {0:1, 1: params[i][0]}
    
    if other_hyperparams != None:
        new_dict = {**new_dict, **other_hyperparams}
        
    return new_dict


# In[ ]:


def get_score_2(ytest, yhat_test , yhat_test_proba, cutoff_train, simple_b_score, business_dict):
    scores = OrderedDict()
    scores['count'] = len(ytest)
    scores['bads'] = sum(ytest)
    
    if sum(ytest) == 0:
        scores['ScoreF1'] = None
        scores['Pre'] = None
        scores['Acc'] = metrics.accuracy_score(ytest, yhat_test)
        #scores['Pre'] = metrics.precision_score(ytest, yhat_test)
        scores['Rec'] = metrics.recall_score(ytest, yhat_test) 
        scores['APS'] = None
        scores['AUC'] = None
        scores['GINI'] = None
        scores['_tn'] = None
        scores['_fn'] = None
        scores['b_best'] = None
        scores['cutoff'] = None
        scores['AR'] = None
        scores['def_6'] = None
        
    else:
        
        if sum(yhat_test) == 0:
            scores['ScoreF1'] = 0
            scores['Pre'] = 0
        else:
            scores['ScoreF1'] = metrics.f1_score(ytest, yhat_test)
            scores['Pre'] = metrics.precision_score(ytest, yhat_test)

        #scores['ScoreF1'] = metrics.f1_score(ytest, yhat_test)
        scores['Acc'] = metrics.accuracy_score(ytest, yhat_test)
        #scores['Pre'] = metrics.precision_score(ytest, yhat_test)
        scores['Rec'] = metrics.recall_score(ytest, yhat_test) 
        scores['APS'] = metrics.average_precision_score(ytest, yhat_test_proba)
        scores['AUC'] = metrics.roc_auc_score(ytest, yhat_test_proba)
        scores['GINI'] = 2*metrics.roc_auc_score(ytest, yhat_test_proba)-1
                    # находим лучший cut-off по трейн и применяем его для тест!!
                    #b_best_train, cutoff_train = b_score_cutoff(ytrain, yhat_train_proba, t0)
                    #y_best_test = pd.Series(np.where(yhat_test_proba >= cutoff_train , 1, 0))

        _tn, _fp, _fn, _tp = metrics.confusion_matrix(ytest, yhat_test).ravel()
        b_best_fin = simple_b_score(_tn, _fp, _fn, _tp, **business_dict )

        scores['_tp'] = _tp
        scores['_fp'] = _fp
        scores['_tn'] = _tn
        scores['_fn'] = _fn
        scores['b_best'] = b_best_fin
        scores['cutoff'] = cutoff_train
        #scores['AR'] = ar
        #scores['def_6'] = mob6
    scores = pd.DataFrame.from_dict(scores, orient='index').T
    return scores


# In[ ]:



def bootstrap(X_1, y_1, X_2, y_2, pass_model, 
              nreps, out, preprocessing_hyps, params_to_fit = None, meta = None, by_var = None,
              params_dict = None, other_hyperparams = None, target = None, preprocessing = True, 
              task = 'binary', need_business=True,
              simple_b_score = None, business_dict =None, use_splines=False, max_d = 2, compare = False,
              features_1 = None, features_2 = None, pass_model_2 = None, params_to_fit_2 = None, func_1 = None, func_2 = None, use_data = None, func_1_params = None, func_2_params = None,
              list_of_returned_vars_1 = None, list_of_returned_vars_2 = None, trained_model = None, trained_model_2 = None):
    """
    ----------
    compare: сравнение двух моделей в бутстрепе
    features_1 и features_2: набор фичей для сравнения первой и второй модели. Используются при сравнений отличий в фичах данных
    pass_model_2: вторая модель. Используются при сравнений отличий в моделях
    func_1 и func_2: функции для манипуляций с данными перед обучением, сравнение отличий манипуляций. Название функций.
    use_data: Какие наборы данных использовать для func_1 и func_2
    func_1_params и func_2_params: параметры функций без наборов данных
    list_of_returned_vars_1 и list_of_returned_vars_2: название возвращаемых переменных
        !!!!ВАЖНО!!!!

        В функцию надо подавать как прогнозы и истинные метки трейна, так и прогнозы и истинные метки теста, так как на трейне
        рассчитывается оптимальный порог, по которому считается бизнес метрика, но итоговые результаты и выводы о бизнес метрике надо
        делать на ТЕСТЕ!

    """
    
    def business_cutoff(y_true, y_score, simple_score, business_dictionary, pos_label=None, 
                               sample_weight=None):
        """
        ----------
        y_true : array, shape = [n_samples]
            True targets of binary classification
        y_score : array, shape = [n_samples]
            Estimated probabilities or decision function
        y_test: True TEST targets
        y_test_score: predictions on target
        pos_label : int or str, default=None
            The label of the positive class
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        Returns

            !!!!ВАЖНО!!!!

            В функцию надо подавать как прогнозы и истинные метки трейна, так и прогнозы и истинные метки теста, так как на трейне
            рассчитывается оптимальный порог, по которому считается бизнес метрика, но итоговые результаты и выводы о бизнес метрике надо
            делать на ТЕСТЕ!

        """
        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score_max = b_score.max()
        cut_off_max = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]

        return best_score_max, cut_off_max
    
    def get_score_2_binary(ytest, yhat_test , yhat_test_proba, need_business = True, cutoff_train = None,
                           simple_b_score = None, business_dict = None):
        scores = OrderedDict()
        scores['count'] = len(ytest)
        scores['bads'] = sum(ytest)
        if sum(ytest) == 0:
            scores['ScoreF1'] = None
            scores['Pre'] = None
            scores['Acc'] = metrics.accuracy_score(ytest, yhat_test)
            #scores['Pre'] = metrics.precision_score(ytest, yhat_test)
            scores['Rec'] = metrics.recall_score(ytest, yhat_test) 
            scores['APS'] = None
            scores['AUC'] = None
            scores['GINI'] = None
            scores['_tn'] = None
            scores['_fn'] = None
            if need_business == True:
                scores['b_best'] = None
                scores['cutoff'] = None
            #cores['AR'] = None
            #cores['def_6'] = None
        

        else:
            if sum(yhat_test) == 0:
                scores['ScoreF1'] = 0
                scores['Pre'] = 0
            else:
                scores['ScoreF1'] = metrics.f1_score(ytest, yhat_test)
                scores['Pre'] = metrics.precision_score(ytest, yhat_test)

            #scores['ScoreF1'] = metrics.f1_score(ytest, yhat_test)
            scores['Acc'] = metrics.accuracy_score(ytest, yhat_test)
            #scores['Pre'] = metrics.precision_score(ytest, yhat_test)
            scores['Rec'] = metrics.recall_score(ytest, yhat_test) 
            scores['APS'] = metrics.average_precision_score(ytest, yhat_test_proba)
            scores['AUC'] = metrics.roc_auc_score(ytest, yhat_test_proba)
            scores['GINI'] = 2*metrics.roc_auc_score(ytest, yhat_test_proba)-1
                        # находим лучший cut-off по трейн и применяем его для тест!!
                        #b_best_train, cutoff_train = b_score_cutoff(ytrain, yhat_train_proba, t0)
                        #y_best_test = pd.Series(np.where(yhat_test_proba >= cutoff_train , 1, 0))

            _tn, _fp, _fn, _tp = metrics.confusion_matrix(ytest, yhat_test).ravel()
      
            if need_business == True:
                b_best_fin = simple_b_score(_tn, _fp, _fn, _tp, **business_dict )

            scores['_tp'] = _tp
            scores['_fp'] = _fp
            scores['_tn'] = _tn
            scores['_fn'] = _fn
            if need_business == True:
                scores['b_best'] = b_best_fin
                scores['cutoff'] = cutoff_train
                
        scores = pd.DataFrame.from_dict(scores, orient='index').T
        return scores
    
    def get_score_2_numeric(ytest, yhat_test):
        scores = OrderedDict()

        scores['R2'] = metrics.r2_score(ytest, yhat_test)
        scores['MSE']= metrics.mean_squared_error(ytest, yhat_test)
        scores['MAE']= metrics.mean_absolute_error(ytest, yhat_test)
        scores['MedianAE']= metrics.median_absolute_error(ytest, yhat_test)
        #scores['MSLE']= metrics.mean_squared_log_error(ytest, yhat_test)
        scores['RMSE']= np.sqrt(metrics.mean_squared_error(ytest, yhat_test))
        #scores['RMSLE']= np.sqrt(metrics.mean_squared_log_error(ytest, yhat_test))
      
        scores = pd.DataFrame.from_dict(scores, orient='index').T
        return scores
    
    def receive_hyperparams(meta, by_var, params_dict, other_hyperparams = None):
    
        """
        Для бутстрэпа. Объединяет зафиксированные параметры (other_hyperparams) и отбираемые (params_dict).    

        """

        params = pd.DataFrame(meta.sort_values(by = by_var, 
                                        ascending = False).iloc[0][list(params_dict.keys())]).T.reset_index()

        new_dict = {}

        for i in params_dict.keys():
            if len(re.findall('weight', i)) == 0:
                new_dict[params_dict[i]] = params[i][0]
            elif len(re.findall('class_weight', i)) > 0:
                new_dict[params_dict[i]] = params[i][0]
            else:
                if len(re.findall('0', i)) > 0:
                    new_dict['class_weight'] = {0: params[i][0], 1:1}
                elif len(re.findall('1', i)) > 0:
                    new_dict['class_weight'] = {0:1, 1: params[i][0]}

        if other_hyperparams != None:
            new_dict = {**new_dict, **other_hyperparams}

        return new_dict
    import catboost as cat
    
    if type(target) == type(None) and type(y_1) != type(None):
        Xtrain = X_1.join(y_1)
        target1 = y_1.name
    elif type(target) == type(None) and type(y_1) == type(None):
        raise ValueError("Target should be passed!")
    elif type(target) != type(None) and type(y_1) == type(None):
        Xtrain = X_1.copy()
    elif type(target) != type(None) and type(y_1) != type(None):
        Xtrain = X_1.join(y_1)
        target1 = target
    
    if type(target) == type(None) and type(y_2) != type(None):
        Xtest = X_2.join(y_2)
        target1 = y_2.name
    elif type(target) == type(None) and type(y_2) == type(None):
        raise ValueError("Target should be passed!")
    elif type(target) != type(None) and type(y_2) == type(None):
        Xtest = X_2.copy()
    elif type(target) != type(None) and type(y_2) != type(None):
        Xtest = X_2.join(y_2)
        target1 = target
        
    betas_box = []
    
    for k in tqdm(range(1, 1+nreps), total=nreps, desc='Repetitions'):
        # if k % 5 == 0:
        #     dd = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
        #     print ('Number of finished repetitions:', k, '| time: ', dd)

        X_train_sh = resample(Xtrain,  replace=True, random_state=k+1).copy()
        X_test_sh = resample(Xtest,  replace=True, random_state=k+1).copy()

        y_train = X_train_sh[target1]
        X_train_sh.drop(target1, axis = 1, inplace = True)
        y_test = X_test_sh[target1]
        X_test_sh.drop(target1, axis = 1, inplace = True)
        
        if preprocessing == True:
            preproc_results = data_preprocessing(X_train_sh, y_train, X_test_sh,
                                                             y_test, **preprocessing_hyps)
            X_tr = preproc_results[0].copy()
            X_te = preproc_results[1].copy()
            y_tr = preproc_results[2].copy()
            y_te = preproc_results[3].copy()
        else:
            X_tr = X_train_sh.copy()
            X_te = X_test_sh.copy()
            y_tr = y_train.copy()
            y_te = y_test.copy()

        if use_splines == True:
            X_tr, X_te, split_points_data = for_splines(X_tr, X_te, y_tr, preprocessing_hyps['categorial_list'], max_d, 241, silent=True, strategy='recursive')
        if trained_model is not None:
            model = trained_model
        elif type(params_to_fit) != type(None):
            model = pass_model(**params_to_fit)
        elif type(params_to_fit) == type(None) and type(meta) != type(None) and type(by_var) != type(None) and params_dict is not None:
            params_to_fit = receive_hyperparams(meta = meta, by_var = by_var, params_dict = params_dict, other_hyperparams = other_hyperparams)
            model = pass_model(**params_to_fit)
        if trained_model_2 is not None:
            model_2 = trained_model_2
        elif type(params_to_fit_2) != type(None):
            model_2 = pass_model_2(**params_to_fit_2)
        
        datasets={"X_tr":X_tr,
                  "X_te":X_te,
                  "y_tr":y_tr,
                  "y_te":y_te}
                    
        if (compare):
            if type(features_1) is not type(None) and type(features_1) is list and type(features_1) is type(features_2):
                X_tr1=X_tr[features_1]
                X_tr2=X_tr[features_2]
                
                X_te1=X_te[features_1]
                X_te2=X_te[features_2]
                
                if pass_model_2 == cat.CatBoostClassifier:
                    
                    cat_features_names = [i for i in preprocessing_hyps['categorial_list'] if i in X_tr2[features_2].columns] # here we specify names of categorical features
                    cat_features = [X_tr2[features_2].columns.get_loc(col) for col in cat_features_names if col in X_tr2[features_2].columns]
                    # train1[[*features_lgbm,  *sys_no_target]].iloc[:,cat_features]=train1[[*features_lgbm,  *sys_no_target]].iloc[:,cat_features].astype(str)
                    X_tr2[cat_features_names]=X_tr2[cat_features_names].astype(str)
                    X_te2[cat_features_names]=X_te2[cat_features_names].astype(str)
                    
                data_one_feat=[X_tr1,X_tr2]
                test_one_feat=[X_te1, X_te2]
                
            elif type(func_1) is not type(None):
                datasets = {k: datasets[k] for k in use_data}
                cort1 = (func_1(*datasets.values(), **func_1_params))
                returns1 = {var: returned_var for var, returned_var in zip(list_of_returned_vars_1, cort1)}
                if type(func_2) is not type(None): #Сравнение двух функций предобработки данных
                    cort2 = (func_2(*datasets.values(), **func_2_params))
                    returns2 = {var: returned_var for var, returned_var in zip(list_of_returned_vars_2, cort2)}

                    data_one_feat=[returns1["X_tr1"],returns2["X_tr2"]]
                    test_one_feat=[returns1["X_te1"],returns2["X_te2"]]

                else: #Сравнение добавления функции предобработки данных
                    
                    data_one_feat=[returns1["X_tr1"],X_tr]
                    test_one_feat=[returns1["X_te1"],X_te]
                
                
            else:
                data_one_feat=[X_tr]
                test_one_feat=[X_te]
        else:
            data_one_feat=[X_tr]
            test_one_feat=[X_te]
        print(data_one_feat)
        for i in range(len(data_one_feat)):
            
            X_tr = data_one_feat[i]
            X_te = test_one_feat[i]
            if pass_model_2 is not None and i==(len(data_one_feat)-1) and trained_model_2 is None:
                m = model_2.fit(X_tr, y_tr)
            elif trained_model is None:
                m = model.fit(X_tr, y_tr)

            # if hasattr(m, 'coef_') and hasattr(m, 'intercept_'): 
            #     betas = [*m.coef_[0].tolist(), m.intercept_[0]]
            #     betas_box.append(betas)

            if task == 'binary':
                pred_proba_y = m.predict_proba(X_te)[:,1]
                pred_proba_train = m.predict_proba(X_tr)[:, 1]

            else:
                yhat_test = m.predict(X_te)
                yhat_train = m.predict(X_tr)
                
            if need_business == True and task == 'binary':

                b_best_train, cutoff_train = business_cutoff(y_tr, pred_proba_train, simple_b_score, 
                                                             business_dictionary = business_dict)

                y_best_test = pd.Series(np.where(pred_proba_y >= cutoff_train , 1, 0))   
                y_best_train = pd.Series(np.where(pred_proba_train >= cutoff_train , 1, 0))

                scores1 = get_score_2_binary(y_tr, y_best_train , pred_proba_train, 
                                            need_business, cutoff_train, simple_b_score, business_dict)
                scores2 = get_score_2_binary(y_te, y_best_test , pred_proba_y,
                                             need_business, cutoff_train, simple_b_score, business_dict)
                
            elif need_business == False and task == 'binary':
                
                p_f_score, r_f_score, treshold_f_score = metrics.precision_recall_curve(y_train, pred_proba_train)
                new_elements  = np.r_[0, treshold_f_score]
                
                f_score = (2*p_f_score*r_f_score)/(p_f_score+r_f_score)
                f_score = np.where(np.isnan(f_score), 0, f_score)

                cut_f = new_elements[np.argmax(f_score)]

                y_best_test = pd.Series(np.where(pred_proba_y >= cut_f , 1, 0))
                y_best_train = pd.Series(np.where(pred_proba_train >= cut_f , 1, 0))

                scores1 = get_score_2_binary(y_tr, y_best_train , pred_proba_train, need_business = False, cutoff_train =  None, 
                                             simple_b_score = None, business_dict = None)
                scores2 = get_score_2_binary(y_te, y_best_test , pred_proba_y, need_business = False, cutoff_train =  None, 
                                             simple_b_score = None, business_dict = None)
                
            elif task == 'numeric':
                scores1 = get_score_2_numeric(y_tr, yhat_train)
                scores2= get_score_2_numeric(y_te, yhat_test)
            
            if k == 1:
                scores1.to_csv(out + f'bootstrap_scores_train{i}.csv', index=False)
                scores2.to_csv(out + f'bootstrap_scores_test{i}.csv', index=False)
                if (i==0):
                    train_scores = scores1
                    test_scores = scores2   
                elif (compare):
                    train_scores_2 = scores1
                    test_scores_2 = scores2
                    
            else:
                with open(out + f'bootstrap_scores_train.csv{i}', 'a') as outf:
                    scores1.to_csv(outf, header=False, index=False)
                with open(out + f'bootstrap_scores_test{i}.csv', 'a') as outf:
                    scores2.to_csv(outf, header=False, index=False)
                if (i==0):
                    train_scores = pd.concat([train_scores, scores1])
                    test_scores = pd.concat([test_scores, scores2])  
                elif (compare):
                    train_scores_2 = pd.concat([train_scores_2, scores1])
                    test_scores_2 = pd.concat([test_scores_2, scores2])
                
        # if hasattr(m, 'coef_') and hasattr(m, 'intercept_'):
        #     betas_box1 = pd.DataFrame.from_records(betas_box, columns = [*X_tr.columns, 'Intercept'])
        #     return train_scores, test_scores, betas_box1
        # else:
    if (compare):
        train_scores.reset_index(inplace=True, drop=True)
        test_scores.reset_index(inplace=True, drop=True)
        train_scores_2.reset_index(inplace=True, drop=True)
        test_scores_2.reset_index(inplace=True, drop=True)
        return train_scores, test_scores, train_scores_2, test_scores_2
    else:
        train_scores.reset_index(inplace=True, drop=True)
        test_scores.reset_index(inplace=True, drop=True)
        return train_scores, test_scores


def individual_hists_all(data, columns, cut):
    
    """
    Рисует индивидуальные графики
    data - данные
    columns - колонки
    cut - булевый параметр - обрезать ли верхние значения по 99 перцентилю
    """
    
    fig, axes = plt.subplots(len(columns), 1, 
                         figsize = (8, len(columns)*3), dpi = 80, constrained_layout=True)
        
    for i, x in enumerate(columns):
        
        if data[x].dtype == 'object':
            
            label = LabelEncoder().fit(data[x].fillna('MISSING'))
            to_plot = label.transform(data[x].fillna('MISSING'))
            
        else:
            
            minimum = data[x].min() - 1
        
            if cut == True:

                to_plot = data[x]
                to_plot = to_plot.fillna(minimum)
                perc = np.percentile(to_plot, 99)
                means = np.trunc(to_plot.loc[to_plot > perc].mean())
                to_plot.loc[to_plot > perc] = perc

            elif cut == False:
                to_plot = data[x]
                to_plot = to_plot.fillna(minimum)
            
        seab = sns.distplot(to_plot, ax = axes[i], color = 'purple', kde = True, norm_hist = True)


# In[ ]:


def paired_time_hists_by_month(data, columns, time, cut):
    
    """
    Рисует помесячные сравнения фичи с собой.
    data - данные
    columns - переменные
    time - переменная по времени
    cut - обрезать ли данные по 99 перцентилю
    """
    
    unique = np.sort(data[time].unique())
    time_len = len(unique) - 1
    Statistics = []
        
    for x in columns:
        
        fig, axes = plt.subplots(len(unique)-1,
                         1, figsize = (5, 15), dpi = 100, 
                         sharex = True, constrained_layout = True)
        
        for i in range(len(unique)):
            if i == time_len:
                break
            else:
                
                if data[x].dtype == 'object':

                    label = LabelEncoder().fit(data[x].fillna('MISSING'))
                    to_plot = label.transform(data[x].fillna('MISSING'))
                else:  
                    
                    minimum = data[x].min() - 1
                
                    if cut == True:

                        to_plot = data.loc[data[time] == unique[i], x]
                        to_plot = to_plot.fillna(minimum)
                        perc = np.percentile(to_plot, 99)
                        means = to_plot.loc[to_plot > perc].mean()
                        to_plot.loc[to_plot > perc] = perc

                        to_plot1 = data.loc[data[time] == unique[i+1], x]
                        to_plot1 = to_plot1.fillna(minimum)
                        perc1 = np.percentile(to_plot1, 99)
                        means1 = to_plot1.loc[to_plot1 > perc1].mean()
                        to_plot1.loc[to_plot1 > perc1] = perc1

                    elif cut == False:    

                        to_plot = data.loc[data[time] == unique[i], x]
                        to_plot = to_plot.fillna(minimum)
                        to_plot1 = data.loc[data[time] == unique[i+1], x]
                        to_plot1 = to_plot1.fillna(minimum)
                
                seab = sns.distplot(to_plot, ax = axes[i], 
                     color = 'gold', norm_hist = True, kde = True, hist = False,
                                     kde_kws = {'shade': True, 'linewidth': 2},
                                    axlabel = [str(unique[i]), str(unique[i+1])])
                seab1 = sns.distplot(to_plot1, ax = axes[i], 
                                         color = 'purple', norm_hist = True, kde = True, hist = False,
                                     kde_kws = {'shade': True, 'linewidth': 2},
                                     axlabel = [str(unique[i]), str(unique[i+1])])
                seab.set_title(x)
                seab1.set_title(x)


# In[ ]:

def simple_b_score_crm(_tn, _fp, _fn, _tp, m_s , k , c, pos_label=None, sample_weight=None):
    
    """
    Функция в одну строчку, которая считает бизнес метрику по элементам confusion matrix
    
    """
    
    b_score =  m_s*_tp - c*k*(_tp + _fp)
    return b_score
    
    
def indifference_curve(utility,theta):
    vector = [max(0,utility(q,0,theta)-utility) for q in range_q]
    return vector
    
def utility(fpr, tpr, tp_cost, fp_cost, fn_cost, tn_cost, good_rate=0.5):
    return tp_cost*tpr*good_rate+fn_cost*(1-tpr)*good_rate+fp_cost*fpr*(1-good_rate)+tn_cost*(1-fpr)*(1-good_rate)
    

def simple_b_score_risk(_tn, _fp, _fn, _tp, t0, pos_label=None, sample_weight=None):
    
    """
    Функция в одну строчку, которая считает бизнес метрику по элементам confusion matrix
    
    """
    
    b_score =  ((t0*_tn)/(1-t0)) - _fn
    return b_score


# In[ ]:

def b_score_train_and_test(y_true, y_score, y_test, y_test_score, simple_score, business_dictionary, pos_label=None, sample_weight=None):
    """
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    y_test: True TEST targets
    y_test_score: predictions on target
    simple_score - функция бизнес метрики (пример - simple_b_score_risk
    business_dictionary - словарь параметров бизнес метрики
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    
        !!!!ВАЖНО!!!!
        
        В функцию надо подавать как прогнозы и истинные метки трейна, так и прогнозы и истинные метки теста, так как на трейне рассчитывается оптимальный порог, по которому считается бизнес метрика, но итоговые результаты и выводы о бизнес метрике надо делать на ТЕСТЕ!
        
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    tns =  fps[-1] - fps    
    fns =  tps[-1] - tps
    tp_fp = (tps + fps)/y_true.size 
    b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
    best_score_max = b_score.max()
    cut_off_max = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
    #idx_max = max(np.where(tp_fp <= t0)[0])
    #best_score_thr = b_score[idx_max]
    #cut_off_thr = y_score[threshold_idxs][idx_max]
    
    y_best_test_max = pd.Series(np.where(y_test_score >= cut_off_max , 1, 0))
    #y_best_test_thr = pd.Series(np.where(y_test_score >= cut_off_thr  , 1, 0))
                    
    _tn, _fp, _fn, _tp = metrics.confusion_matrix(y_test, y_best_test_max).ravel()
    b_best_max = simple_score(_tn = _tn, _fp = _fp, _fn = _fn, _tp = _tp, **business_dictionary) 
    
    return best_score_max, cut_off_max, b_best_max#, best_score_thr, cut_off_thr # b_score , tns, fns, fps, tps, y_score[threshold_idxs]


# In[ ]:

def hist_bad_rate(x, y, N=10, category=False, reset = True, fillnas = 'median', method = 'quantiles'):
    
    """
   Бьет на бины функцию и считает ее bad rate для каждого бина. Требуется для отрисовки графика зависимости bad rate от значения фичи. 
   Пропуски заполняются:
   'median' - медианой
   'min-1' - минимальным - 1 значением
   None - ничем
   числом, если задать его
   
   """
    
    if reset == True:
        x_name = x.name
        y_name = y.name
        x_new = x.reset_index()[x_name]
        y_new = y.reset_index()[y_name]
        X = pd.DataFrame(x_new).join(pd.DataFrame(y_new))
        to_drop = [i for i in X.columns if i not in [x_name, y_name]]
        X.drop(to_drop, axis = 1, inplace = True)
        
    if category == True or x_new.nunique() == 2 or x_new.dtype == 'object':
        if x_new.nunique()>1:
            new_name = x_name+'_binned'
            X[new_name] = x_new.fillna('MISSING')
        
        else:
            return 'One value!'
    else:
        
        '''Calculate fit metrics for given target and predicted outcomes.'''
    
        min_all = min(x_new)
        max_all = max(x_new)

        if max_all - min_all != 0:
            
            if fillnas == None:
                X[x_name] = X[x_name]
            elif fillnas == 'median':
                X[x_name].fillna(X[x_name].median(), inplace = True)
            elif fillnas == 'min-1':
                X[x_name].fillna(X[x_name].min()-1, inplace = True)
            else:
                X[x_name].fillna(fillnas, inplace = True)
            
            new_name = x_name+'_binned'
            
            if method == 'pieces':
                X[new_name] = pd.cut(X[x_name], N)
            elif method == 'quantiles':
                X[new_name] = pd.qcut(X[x_name], N, duplicates = 'drop')
            
        else:
            return 'One value!'

    grouped = X.groupby(new_name).aggregate({y_name: 'mean', x_name: 'count'})
    new_name2 = x_name+'_mean'
    grouped[new_name2] = X.groupby(new_name).aggregate({ x_name: 'mean'})
        
    return grouped


# In[ ]:

def max_prof_corve(y_true, y_score, simple_score, business_dictionary, pos_label=None, sample_weight=None):
    
    """
    
    Создает векторы для отрисовки кривой max_profit.
    Пример использования:
    
    b_auc, tp_fps_auc, cut_auc, best_auc, cutoff_auc = max_prof_corve(y_2, auc_test_pred, simple_b_score_crm, business_dictionary)
    
    plt.figure(figsize = (10, 10))

    plt.title('Business score test')

    plt.plot(tp_fps_auc, b_auc, color='green',
             lw=lw, label='maxProfit AUC model')
    
    """

    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    tns =  fps[-1] - fps    
    fns =  tps[-1] - tps
    tp_fp = (tps + fps)/y_true.size 
    b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
    best_score = b_score.max()
    cut_off = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
    return b_score, tp_fp, y_score[threshold_idxs], best_score, cut_off

# In[ ]:

def calculate_vif(data):
    
    features = data.fillna(data.median(skipna=True))
    vif = pd.DataFrame()
    vif["Features"] = features.columns
    vifs = []
    k = 0
    for i in range(features.shape[1]):
        if k/10 == int(k/10):
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print(tm, k, features.columns[i])
        vifs.append(variance_inflation_factor(features.values, i))
        k +=1
    
    vif['VIF'] = vifs

    return(vif)

# In[ ]:

def individual_hists(data, columns, time, cut):
        
    for x in columns:
        
        fig, axes = plt.subplots(len(data[time].unique()), 1, 
                         figsize = (15, 4), dpi = 100, sharex = True, constrained_layout = True)
        
        for j, cut_time in enumerate(data[time].unique()):
            
            if data[x].dtype == 'object':
                
                label = LabelEncoder().fit(data[x].fillna('MISSING'))
                to_plot = label.transform(data[x].fillna('MISSING'))
            else:
                
                minimum = data[x].min() - 1
                
                if cut == True:

                    to_plot = data.loc[data[time] == cut_time, x]
                    to_plot = to_plot.fillna(minimum)
                    perc = np.percentile(to_plot, 99.5)
                    means = to_plot.loc[to_plot > perc].mean()
                    to_plot.loc[to_plot > perc] = perc

                elif cut == False:
                    to_plot = data.loc[data[time] == cut_time, x]
                    to_plot = to_plot.fillna(minimum)
            
            seab = sns.distplot(to_plot, ax = axes[j], norm_hist = True, color = 'purple', kde = True, hist = False,
                               kde_kws = {'shade': True, 'linewidth': 2})
            seab.set_title([x, cut_time])
# In[ ]:

def paired_time_hists(data, columns, time, cut):
    for x in columns:
        
        unique = data[time].unique()
        
        fig, axes = plt.subplots(len(unique),
                         len(unique), figsize = (14, 10), dpi = 100, 
                         sharex = True, constrained_layout =True)
        
        for i in range(len(unique)):
            for j  in range(len(unique)):
                
                    if data[x].dtype == 'object':

                        label = LabelEncoder().fit(data[x].fillna('MISSING'))
                        to_plot = label.transform(data[x].fillna('MISSING'))

                    else:

                        minimum = data[x].min() - 1

                        if cut == True:

                            to_plot = data.loc[data[time] == unique[i], x]
                            to_plot = to_plot.fillna(minimum)
                            perc = np.percentile(to_plot, 99.5)
                            means = to_plot.loc[to_plot > perc].mean()
                            to_plot.loc[to_plot > perc] = perc

                            to_plot1 = data.loc[data[time] == unique[j], x]
                            to_plot1 = to_plot1.fillna(minimum)
                            perc1 = np.percentile(to_plot1, 99.5)
                            means1 = to_plot1.loc[to_plot1 > perc1].mean()
                            to_plot1.loc[to_plot1 > perc1] = perc1

                        elif cut == False:    

                            to_plot = data.loc[data[time] == unique[i], x]
                            to_plot = to_plot.fillna(minimum)
                            to_plot1 = data.loc[data[time] == unique[j], x]
                            to_plot1 = to_plot1.fillna(minimum)

                    seab = sns.distplot(to_plot, ax = axes[i, j], 
                         color = 'gold', norm_hist = True, kde = True, hist = False,
                                         kde_kws = {'shade': True, 'linewidth': 2},
                                        axlabel = [str(unique[i]), str(unique[j])])
                    seab1 = sns.distplot(to_plot1, ax = axes[i, j], 
                                             color = 'purple', norm_hist = True, kde = True, hist = False,
                                         kde_kws = {'shade': True, 'linewidth': 2},
                                         axlabel = [str(unique[i]), str(unique[j])])
                    seab.set_title(x)
                    seab1.set_title(x)

# In[ ]:

def find_ouliers_iqr(data, technical_cols, mult = 1.5, check_percentile = 1):
    
    cols = list(data.columns)
    
    for i in technical_cols:
        if i in cols:
            cols.remove(i)
            
    attr_data = []
    
    for i in cols:
        check_25 = np.nanpercentile(data[i], 25)
        check_75 = np.nanpercentile(data[i], 75)
        
        low_perc = check_percentile
        high_perc = 100-check_percentile
        check_1 = np.nanpercentile(data[i], low_perc)
        check_99 = np.nanpercentile(data[i], high_perc)
        
        maximum = data[i].max()
        minimum = data[i].min()
        
        if check_25 != check_75:
            
            q_25 = np.nanpercentile(data[i], 25)
            q_75 = np.nanpercentile(data[i], 75)

            iqr = q_75-q_25
            right_border_test = q_75+iqr*mult
            left_border_test = q_25-iqr*mult
            
        else:
            x = data.loc[data[i] != check_25, i]
            q_25 = np.nanpercentile(x, 25)
            q_75 = np.nanpercentile(x, 75)
            
            iqr = q_75-q_25
            right_border_test = q_75+iqr*mult
            left_border_test = q_25-iqr*mult
            
        if left_border_test < minimum:
            left_border = minimum
        elif check_1 < left_border_test:
            left_border = check_1
        else: 
            left_border = left_border_test
            
        if right_border_test > maximum:
            right_border = maximum
        elif check_99 > right_border_test:
            right_border = check_99
        else: 
            right_border = right_border_test
                
        attr_data.append([i, iqr, right_border, left_border])
            
                   

    attr_columns = ['variable', 'IQR', 'right_border', 'left_border']    
    outlier_data = pd.DataFrame.from_records(attr_data, columns = attr_columns)

    return outlier_data


# In[ ]:

def find_outliers_z_score(data, technical_cols, treshold = 3):
    
    cols = list(data.columns)
    
    for i in technical_cols:
        if i in cols:
            cols.remove(i)
            
    attr_data = []
    
    for i in cols:
        scales_right = treshold*data[i].std()+data[i].mean()
        scales_left = (-1)*treshold*data[i].std()+data[i].mean()
        
        attr_data.append([i, treshold, scales_right, scales_left])
        
    attr_columns = ['variable', 'treshold', 'right_border', 'left_border']    
    outlier_data = pd.DataFrame.from_records(attr_data, columns = attr_columns)

    return outlier_data



def data_preprocessing_cols(X_1, y_1, X_2, y_2, technical_values, 
                           yeo_johnson = None, 
                           attribute_list = None, var_col = None,
                           scale = None, 
                           median = 'median',
                           high_outlier = None, 
                           low_outlier = None,
                           check_percentile = 1,
                           cols_outlier = None,
                            cut_non_out_9999 = True
                          ):
          
    """
    Проводит препроцессинг для train и test выборки.

    Функция data_preprocessing_cols дублирует data_preprocessing. 
    Отличие от data_preprocessing по передаваемым аргументам:
    1. yeo_johnson - это список переменных, к которым применяется преобразование нормализации. Если преобразования не нужны, то передается пустой список.
    2. scale - это список переменных, к которым применяется преобразование масштабирования. Если преобразования не нужны, то передается пустой список.
    3. cols_outlier - это список переменных, к которым применяется преобразование импутации выбросов. Если преобразования не нужны, то передается пустой список.

    X_1, y_1, X_2, y_2 - данные
    technical_values - список технических переменных
    technical_values исключены из анализа и удаляются из выборки. Если технических переменных нет, можно задать пустой список. 
    
    yeo_johnson - проводить ли нормализацию Йео-Джонсона (приведение распределения данных к нормальному виду). Подается список колонок для преобразования
    attribute_list - данные attribute_list. Могут быть None. По дефолту None
    var_col - в каком поле attribute_list находятся названия фичей. Если attribute_list не задан, то в var_col нет нужды.
    По дефолту None.
    scale - проводить ли стандартизацию StandardScaler. Подается список колонок для преобразования
    
    median - импутация пропусков. Возможные принимаемые значения:
        - 'median' - тогда на train куске рассчитываются медианы, и импутация ими и train и test кусков
        - 'min-1' - тогда на train куске рассчитываются минимальные значения - 1, и импутация ими
        - число - если задать число, то пропуски будут заполняться этим числом
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
        - None. Тогда пропуски не импутируются
        По дефолту задано значение 'median'
    high_outlier - импутация выбросов вверх. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 99 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
        - None. Тогда выбросы вверх не импутируются.
    low_outlier - импутация выбросов вниз. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 1 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
        - None. Тогда выбросы вниз не импутируются. 
     cols_outlier - Подается список колонок для импутации выбросов
     
     Возвращает измененные данные, кортеж для преобразования Йео_Джонсон (масштабирование и лямбда) и обученный Scaler
     
    """
    
    xtrain = X_1.copy()
    xtest = X_2.copy()
    ytrain = y_1.copy()
    ytest = y_2.copy()
    
    for i in technical_values:
        if i in xtrain.columns:
            xtrain.drop(i, axis = 1, inplace = True)
        if i in xtest.columns:
            xtest.drop(i, axis = 1, inplace= True)
            
    train_c = xtrain.columns
    test_c = xtrain.columns
    train_ind = xtrain.index
    test_ind = xtest.index
                
    for oo in tqdm(xtrain.columns):
        if median != None:
            if median == 'median':
                medians = xtrain[oo].median(skipna = True)
            elif median == 'min-1':
                medians = xtrain[oo].min(skipna = True)-1
            elif type(attribute_list) != type(None) and median in attribute_list.columns:
                medians = list(attribute_list.loc[attribute_list[var_col] == oo, median])[0]
            else:
                medians = median
                
        if high_outlier != None:
            if oo in cols_outlier:
                if type(attribute_list) != type(None) and high_outlier in attribute_list.columns:
                    to_replace_high = list(attribute_list.loc[attribute_list[var_col] == oo, high_outlier])[0]
                    
                elif high_outlier == 'IQR':
                    check_25 = np.nanpercentile(xtrain[oo], 25)
                    check_75 = np.nanpercentile(xtrain[oo], 75)
                    check_99 = np.nanpercentile(xtrain[oo], 100-check_percentile)
                    maximum = xtrain[oo].max()
                    
                    if check_25 != check_75:
                        q_25 = np.nanpercentile(xtrain[oo], 25)
                        q_75 = np.nanpercentile(xtrain[oo], 75)
                        iqr = q_75-q_25
                        right_border = q_75+iqr*1.5
                    else:
                        x = xtrain.loc[xtrain[oo] != check_25, oo]
                        q_25 = np.nanpercentile(x, 25)
                        q_75 = np.nanpercentile(x, 75)
                        iqr = q_75-q_25
                        right_border = q_75+iqr*1.5
                    
                    if right_border > maximum:
                        to_replace_high = maximum
                    elif check_99 > right_border:
                        to_replace_high = check_99
                    else:
                        to_replace_high = right_border
                        
                elif high_outlier == 'z-score':
                    to_replace_high = 3*xtrain[oo].std()+xtrain[oo].mean()
                        
                else:
                    to_replace_high = np.nanpercentile(xtrain[oo], high_outlier)
                        
        elif high_outlier == None:
            to_replace_high = None

        if low_outlier != None:
            if oo in cols_outlier:
                if type(attribute_list) != type(None) and low_outlier in attribute_list.columns:
                    to_replace_low = list(attribute_list.loc[attribute_list[var_col] == oo, low_outlier])[0]
                    
                elif low_outlier == 'IQR':
                        check_25 = np.nanpercentile(xtrain[oo], 25)
                        check_75 = np.nanpercentile(xtrain[oo], 75)
                        check_1 = np.nanpercentile(xtrain[oo], check_percentile)
                        minimum = xtrain[oo].min()
                        if check_25 != check_75:
                            q_25 = np.nanpercentile(xtrain[oo], 25)
                            q_75 = np.nanpercentile(xtrain[oo], 75)
                            iqr = q_75-q_25
                            left_border = q_25-iqr*1.5
                        else:
                            x = xtrain.loc[xtrain[oo] != check_25, oo]
                            q_25 = np.nanpercentile(x, 25)
                            q_75 = np.nanpercentile(x, 75)
                            iqr = q_75-q_25
                            left_border = q_25-iqr*1.5

                        if left_border < minimum:
                            to_replace_low = minimum
                        elif check_1 < left_border:
                            to_replace_low = check_1
                        else:
                            to_replace_low = left_border
                            
                elif low_outlier == 'z-score':
                    to_replace_low = (-1)*3*xtrain[oo].std()+xtrain[oo].mean()
                else:
                    to_replace_low = np.nanpercentile(xtrain[oo], low_outlier)
                    
        elif low_outlier == None:
            to_replace_low = None
                    
        if median != None:
            xtrain[oo] = xtrain[oo].fillna(medians)
            xtest[oo] = xtest[oo].fillna(medians)
        if high_outlier != None:
            if oo in cols_outlier:
                xtrain.loc[xtrain[oo] > to_replace_high, oo] = to_replace_high
                xtest.loc[xtest[oo] > to_replace_high, oo] = to_replace_high
        if low_outlier != None:
            if oo in cols_outlier:
                xtrain.loc[xtrain[oo] < to_replace_low, oo] = to_replace_low
                xtest.loc[xtest[oo] < to_replace_low, oo] = to_replace_low

    powermodel_maxmin = []
    powermodel_lambda = []
    for ccc in yeo_johnson:
        max_min = xtrain[ccc].max() - xtrain[ccc].min()
        xtrain[ccc] = xtrain[ccc]/max_min
        xtest[ccc] = xtest[ccc]/max_min
        power = PowerTransformer(method = 'yeo-johnson', standardize = False).fit(xtrain[[ccc]])
        xtrain[ccc] = power.transform(xtrain[[ccc]])
        xtest[ccc] = power.transform(xtest[[ccc]])
        powermodel_maxmin.append(max_min)
        powermodel_lambda.append(power.lambda_)
        
    pr = preprocessing.StandardScaler()
    
    if isinstance(scale,list) and len(scale) > 0:
        pr.fit(xtrain[scale])
        xtrain[scale] = pr.transform(xtrain[scale])
        xtest[scale] = pr.transform(xtest[scale])
        
    return xtrain, xtest, ytrain, ytest, (powermodel_maxmin, powermodel_lambda),  pr
# In[ ]:
def check_attribute_list_cases(data, attribute_list, target, task = 'binary'):
    
    result_list = []
        
    if task == 'binary':
        if attribute_list['count_dist'].max() == 1 and attribute_list['count_miss'].min() != 0:
            for i in attribute_list['attribute']:
                data_target_1_sum = data.loc[data[i].isna() == True, target].sum()
                data_target_1_count = data.loc[data[i].isna() == True, target].count()
                data_target_2_sum = data.loc[data[i].isna() == False, target].sum()  
                data_target_2_count = data.loc[data[i].isna() == False, target].count()
                
                data_target_1_br = data_target_1_sum/data_target_1_count
                data_target_2_br = data_target_2_sum/data_target_2_count
                
                Br_divide = data_target_1_br/data_target_2_br
                stat, pval = proportions_ztest([data_target_1_sum, data_target_2_sum], 
                                            [data_target_1_count, data_target_2_count])
                
                result_list.append([i, data_target_1_count, data_target_2_count, 
                                    data_target_1_br, data_target_2_br, Br_divide, pval])
                
        elif attribute_list['count_miss'].min() >= data.shape[0]*0.97 and attribute_list['count_miss'].max() != data.shape[0]:
            for i in attribute_list['attribute']:
                data_target_1_sum = data.loc[data[i].isna() == True, target].sum()
                data_target_1_count = data.loc[data[i].isna() == True, target].count()
                data_target_2_sum = data.loc[data[i].isna() == False, target].sum()  
                data_target_2_count = data.loc[data[i].isna() == False, target].count()
                
                data_target_1_br = data_target_1_sum/data_target_1_count
                data_target_2_br = data_target_2_sum/data_target_2_count
                
                Br_divide = data_target_1_br/data_target_2_br
                stat, pval = proportions_ztest([data_target_1_sum, data_target_2_sum], 
                                            [data_target_1_count, data_target_2_count])
                
                result_list.append([i, data_target_1_count, data_target_2_count, 
                                    data_target_1_br, data_target_2_br, Br_divide, pval])
                
        elif (attribute_list['99%']-attribute_list['1%']).max() == 0:
            for i in attribute_list['attribute']:
                
                Perc_99 = list(attribute_list.loc[attribute_list['attribute'] == i, '99%'])[0]
                
                if data.loc[data[i] > Perc_99, target].count() != 0:
                
                    data_target_1_sum = data.loc[data[i] > Perc_99, target].sum()
                    data_target_1_count = data.loc[data[i] > Perc_99, target].count()
                    data_target_2_sum = data.loc[data[i] <= Perc_99, target].sum()  
                    data_target_2_count = data.loc[data[i] <= Perc_99, target].count()

                    data_target_1_br = data_target_1_sum/data_target_1_count
                    data_target_2_br = data_target_2_sum/data_target_2_count

                    Br_divide = data_target_1_br/data_target_2_br
                    stat, pval = proportions_ztest([data_target_1_sum, data_target_2_sum], 
                                                [data_target_1_count, data_target_2_count])

                    result_list.append([i, data_target_1_count, data_target_2_count, 
                                        data_target_1_br, data_target_2_br, Br_divide, pval])
                
                Perc_1 = list(attribute_list.loc[attribute_list['attribute'] == i, '1%'])[0]
                
                if data.loc[data[i] < Perc_1, target].count() != 0:
                
                    data_target_1_sum = data.loc[data[i] < Perc_1, target].sum()
                    data_target_1_count = data.loc[data[i] < Perc_1, target].count()
                    data_target_2_sum = data.loc[data[i] >= Perc_1, target].sum()  
                    data_target_2_count = data.loc[data[i] >= Perc_1, target].count()

                    data_target_1_br = data_target_1_sum/data_target_1_count
                    data_target_2_br = data_target_2_sum/data_target_2_count

                    Br_divide = data_target_1_br/data_target_2_br
                    stat, pval = proportions_ztest([data_target_1_sum, data_target_2_sum], 
                                                [data_target_1_count, data_target_2_count])

                    result_list.append([i, data_target_1_count, data_target_2_count, 
                                        data_target_1_br, data_target_2_br, Br_divide, pval])

        labels = ['attribute', 'Count_1', 'Count_2', 'Bad_Rate_1', 'Bad_Rate_2', 'Target_divides', 'p_value']
        
        check_data = pd.DataFrame.from_records(result_list, columns = labels)
    
    if task == 'numeric':
        if attribute_list['count_dist'].max() == 1 and attribute_list['count_miss'].min() != 0:
            for i in attribute_list['attribute']:
                
                a = data.loc[data[i].isna() == True, target]
                b = data.loc[data[i].isna() == False, target]
                
                Br_divide = a.mean()/b.mean()
                
                if len(a.unique()) > 8 and len(b.unique()) > 8:
                    x_normal = normaltest(a, nan_policy = 'omit').pvalue
                    y_normal = normaltest(b, nan_policy = 'omit').pvalue
                    x_ks = kstest(a, 'norm').pvalue
                    y_ks = kstest(b, 'norm').pvalue
                    
                    if x_normal < 0.01 or y_normal < 0.01 or x_ks < 0.01 or y_ks < 0.01:
                        result_list.append([i, a.count(), b.count(), 
                                            a.mean(), b.mean(),
                                            Br_divide, mannwhitneyu(a, b).pvalue])
                    else:
                        result_list.append([i, a.count(), b.count(), 
                                            a.mean(), b.mean(), 
                                            Br_divide, ttest_ind(a, b, nan_policy = 'omit').pvalue])
                
        elif attribute_list['count_miss'].min() >= data.shape[0]*0.97 and attribute_list['count_miss'].max() != data.shape[0]:
            for i in attribute_list['attribute']:
                a = data.loc[data[i].isna() == True, target]
                b = data.loc[data[i].isna() == False, target]
                
                Br_divide = a.mean()/b.mean()
                
                if len(a.unique()) > 8 and len(b.unique()) > 8:
                    x_normal = normaltest(a, nan_policy = 'omit').pvalue
                    y_normal = normaltest(b, nan_policy = 'omit').pvalue
                    x_ks = kstest(a, 'norm').pvalue
                    y_ks = kstest(b, 'norm').pvalue
                    
                    if x_normal < 0.01 or y_normal < 0.01 or x_ks < 0.01 or y_ks < 0.01:
                        result_list.append([i, a.count(), b.count(), 
                                            a.mean(), b.mean(),
                                            Br_divide, mannwhitneyu(a, b).pvalue])
                    else:
                        result_list.append([i, a.count(), b.count(), 
                                            a.mean(), b.mean(), 
                                            Br_divide, ttest_ind(a, b, nan_policy = 'omit').pvalue])
                        
                else:
                    result_list.append([i, a.count(), b.count(), 
                                            a.mean(), b.mean(), 
                                            Br_divide, None])
                
        elif (attribute_list['99%']-attribute_list['1%']).max() == 0:
            for i in attribute_list['attribute']:
                
                Perc_99 = list(attribute_list.loc[attribute_list['attribute'] == i, '99%'])[0]
                
                if data.loc[data[i] > Perc_99, target].count() != 0:
                
                    a = data.loc[data[i] > Perc_99, target]
                    b = data.loc[data[i] <= Perc_99, target]
                
                    Br_divide = a.mean()/b.mean()

                    if len(a.unique()) > 8 and len(b.unique()) > 8:
                        x_normal = normaltest(a, nan_policy = 'omit').pvalue
                        y_normal = normaltest(b, nan_policy = 'omit').pvalue
                        x_ks = kstest(a, 'norm').pvalue
                        y_ks = kstest(b, 'norm').pvalue

                        if x_normal < 0.01 or y_normal < 0.01 or x_ks < 0.01 or y_ks < 0.01:
                            result_list.append([i, a.count(), b.count(), 
                                                a.mean(), b.mean(),
                                                Br_divide, mannwhitneyu(a, b).pvalue])
                        else:
                            result_list.append([i, a.count(), b.count(), 
                                                a.mean(), b.mean(), 
                                                Br_divide, ttest_ind(a, b, nan_policy = 'omit').pvalue])
                    else:
                        result_list.append([i, a.count(), b.count(), 
                                                a.mean(), b.mean(), 
                                                Br_divide, None])

                Perc_1 = list(attribute_list.loc[attribute_list['attribute'] == i, '1%'])[0]
                
                if data.loc[data[i] < Perc_1, target].count() != 0:
                    
                    a = data.loc[data[i] < Perc_1, target]
                    b = data.loc[data[i] >= Perc_1, target]

                    Br_divide = a.mean()/b.mean()

                    if len(a.unique()) > 8 and len(b.unique()) > 8:
                        x_normal = normaltest(a, nan_policy = 'omit').pvalue
                        y_normal = normaltest(b, nan_policy = 'omit').pvalue
                        x_ks = kstest(a, 'norm').pvalue
                        y_ks = kstest(b, 'norm').pvalue

                        if x_normal < 0.01 or y_normal < 0.01 or x_ks < 0.01 or y_ks < 0.01:
                            result_list.append([i, a.count(), b.count(), 
                                                a.mean(), b.mean(), 
                                                Br_divide, mannwhitneyu(a, b).pvalue])
                        else:
                            result_list.append([i, a.count(), b.count(), 

                                                a.mean(), b.mean(), 
                                                Br_divide, ttest_ind(a, b, nan_policy = 'omit').pvalue])
                            
                    else:
                        result_list.append([i, a.count(), b.count(), 
                                                a.mean(), b.mean(), 
                                                Br_divide, None])
                
        labels = ['attribute', 'Count_1', 'Count_2', 'Target_mean_1', 'Target_mean_2', 'Target_divides', 'p_value']
        
        check_data = pd.DataFrame.from_records(result_list, columns = labels)
        
    return check_data

# In[ ]:

def br_stat(data, index_month, target, alpha = 0, silent = False):
    """
    Вывод статистики по BR по срезам
    data - датасет
    index_month - переменная, по которой выравниваем BR 
    target - BADFLAG
    alpha - изменяет угол наклона BR
    """
    bad_rate_by_month = data.groupby(index_month).aggregate({index_month: 'count',
                                                             target: ['sum', 'mean']}).sort_index()
    bad_rate_by_month.reset_index(inplace = True)
    bad_rate_by_month.columns = [' '.join(col).strip() for col in bad_rate_by_month.columns.values]
    bad_rate_by_month[target+' mean-1'] = 1-bad_rate_by_month[target+' mean']
    bad_rate_by_month.rename(columns = {index_month+' count': 'all_count', 
                                        target+' sum': 'filling of target', 
                                        target+' mean': 'bad_rate', 
                                        target+' mean-1':'good_rate'}, inplace = True)
    max_br = bad_rate_by_month['bad_rate'].max()
    if silent != True:
        print('Max BR =', round(max_br,3))
    
    bad_rate_by_month['max_br'] = max_br
    bad_rate_by_month['max_br_alpha'] = bad_rate_by_month['max_br']+bad_rate_by_month.index*alpha
    max_br_alpha_delta = (bad_rate_by_month['max_br_alpha'] - bad_rate_by_month['bad_rate']).min()
    bad_rate_by_month['max_br_alpha'] = bad_rate_by_month['max_br_alpha'] - max_br_alpha_delta
    bad_rate_by_month['k_alpha'] = (bad_rate_by_month['max_br_alpha']*bad_rate_by_month['all_count']/bad_rate_by_month['filling of target'] -1 )/(1-bad_rate_by_month['max_br_alpha'])

    return bad_rate_by_month

# In[ ]:

def br_correction(data, index_month, target, correct = False, random_state = 42, alpha = 0, silent=False):
    """
    Выравнивание BR 
    data - датасет
    index_month - переменная, по которой выравниваем BR 
    target - BADFLAG
    alpha - изменяет угол наклона BR
    correct - Позволяет визуально взглянуть на выравнивание BR: 
        - False - Вывод графиков BR без изменения датасета
        - True  - Вывод графиков BR с изменением датасета
    random_state - нужен для воспроизводимости. 
    """
    
    bad_rate_by_month = br_stat(data, index_month, target, alpha, silent=False)
    
    if correct == False:
        plt.plot(bad_rate_by_month[index_month], bad_rate_by_month['bad_rate'], '.-') 
        plt.plot(bad_rate_by_month[index_month], bad_rate_by_month['max_br_alpha'], '.-') 
        return 
    
    X_tmp = data.head(0)
    sd_uniq = data[index_month].unique() 
    if silent != True:
        print('Список срезов для выравнивания:', sd_uniq)
    
    for sd in sd_uniq:
        # print_log('Добавим кратное число строк в', sd)
        data_add = data.loc[(data[target]==1)&(data[index_month]==sd)]
        k=bad_rate_by_month.loc[bad_rate_by_month[index_month]==sd, 'k_alpha'].values[0]
        # print(str(int(k)) + ' ' + str(k) + ' ' + str(data_add.shape))
        
        for i in range(0,int(k)):
            X_tmp = X_tmp.append(data_add, ignore_index=True)
        #      print_log(str(int(k)) + ' ' + str(k)+ ' ' + str(i) + ' ' + str(X_tmp.shape))
            
        # print('Добавим дробное число строк')    
        k_fraction = k - int(k)
        n_samples = int(round(data_add.shape[0]*k_fraction))
        # print(k_fraction, n_samples)
        
        ix = np.random.RandomState(random_state).choice(data_add.shape[0], n_samples)
        data_add_fraction = data_add.iloc[ix]
        X_tmp = X_tmp.append(data_add_fraction, ignore_index=True)
        #  print(str(int(k)) + ' ' + str(k)+ ' ' + ' ' + str(X_tmp.shape))
        
    data = data.append(X_tmp, ignore_index=True)
    if silent != True:
        plt.plot(bad_rate_by_month[index_month], bad_rate_by_month['bad_rate'], '.-')
        bad_rate_by_month = br_stat(data, index_month, target, silent=silent)
        plt.plot(bad_rate_by_month[index_month], bad_rate_by_month['bad_rate'], '.-')
    
    return data

# In[ ]:

def to_zip(file_name):
    file_name_zip = file_name.replace('.csv','.zip') 
    with zipfile.ZipFile(file_name_zip, 'w') as zf:
        zf.write(filename = file_name, arcname = os.path.basename(file_name), compress_type = zipfile.ZIP_DEFLATED)
        os.remove(file_name)

# In[ ]:

def find_meta_params_mem(X, Y, params_dictionary, params_to_model, pass_model, 
                         sort_by_var, list_of_vars_for_strat, n_folds, second_target, 
                         yeo_johnson, 
                         attribute_list, var_col, 
                         need_business = True, 
                         draw = True, 
                         draw_by_approval_rate = False,
                         simple_b_score = None, business_dict = None, business_dict_sec = None,
                         scale = None, 
                         median = 'median',
                         high_outlier = None, 
                         low_outlier = None, 
                         cols_outlier = None,
                         random_state = None, k_logs = 10, file_png = 'All_Max_Profit.png'):
    
    """
    Функция find_meta_params_mem дублирует find_meta_params. 
    Отличие в логике - ускорение работы за счет дополнительного использования памяти, суть - проводится подготовка фолдов единожды.
    Отличие от find_meta_params по передаваемым аргументам:
    1. yeo_johnson - это список переменных, к которым применяется преобразование нормализации. Если преобразования не нужны, то передается пустой список.
    2. scale - это список переменных, к которым применяется преобразование масштабирования. Если преобразования не нужны, то передается пустой список.
    3. cols_outlier - это список переменных, к которым применяется преобразование импутации выбросов. Если преобразования не нужны, то передается пустой список.
    4. file_png - имя файла итоговой картинки

    
    Функция find_meta_params_mem возвращает meta файл, в котором содержится brute-force поиск по сетке. Функция может применяться к различным моделям и для различных параметров, аналогично GridSearchCV. Параметры:

    - X, Y - матрица данных и таргет. Матрица X не должна быть предобработана, так как предобработка делается внутри функции! 

    - params_dictionary - словарь с параметрами, аналогично GridSearchCV. Пример: params_dictionary = {'C': [0.05, 0.1, 0.2], 'weight_0': [0.01, 0.015], 'regularization': 'l2', 'random_state': 241, 'solver': 'liblinear', 'max_iter': 300}. ***!Важно!*** Если подбирается параметр веса класса, ключ имеет значение:

        - Если параметр веса задается в виде списка или np.array веса одного из классов, следует указать какого именно. Например, если производится поиск веса для класса 0, примеры возможных названий: 'weight_0', '0_weight', 'var_0_weight', 'weight_var_0'. В любом случае должны присутстовать '0' и 'weight'. Если ищется класс 1, то должны присутствовать '1' и 'weight'.
        - Если параметр веса с самого начала задается словарем аля [{0: 0.07, 1:1}, {0: 0.14, 1:1}], то ключ словаря обязан иметь 'class_weight'

    - params_to_model - так как ***params_dictionary*** ключи словаря могут отличаться от обозначений в функции, следует задать словарь соответствия. Пример: params_to_model = {'C': 'C', 'weight_0': 'class_weight', 'regularization':'penalty', 'random_state': 'random_state', 'solver': 'solver', 'max_iter': 'max_iter'}

    - pass_model - вызываемая функция. Пример: pass_model = LogisticRegression или pass_model = DecisionTreeClassifier

    - sort_by_var - переменная для деления (пример - id клиента, клиенты должны попасть либо в тест, либо в трейн)
    
    - list_of_vars_for_strat - список переменных для стратификации. Пример: распределение по регионам, распределение по месяцам
    
    - n_folds - количество фолдов

    - second_target - используется ли второй таргет (как в research для модели CRM)

    - yeo_johnson - используется ли преобразование Йео-Джонсона. Подается список колонок для преобразования.
    
    - attribute_list - аттрибут лист для использования его в импутациях пропусков и обрезании выбросов
    
    - var_col - имя поля, в котором в attribute_list находятся названия переменных
    
    - need_business - флажок True/False. Считать ли бизнес метрику. По умолчанию True
    
    - draw - флажок True/False. Рисовать ли картинки. По умолчанию True
    
    - draw_by_approval_rate - флажок True/False. Рисовать ли картинки по Approval_rate вместо treshold. По умолчанию False
    
    - simple_b_score - функция, которая расчитывает бизнес метрику
    
    - business_dict - словарь с параметрами для бизнес метрики. Пример для CRM:business_dictionary = {'t0': 0.1, 'm_s': 19000, 'fund': 1, 'k': 20, 'c': 3}
    
    - business_dic_sec - словарь с параметрами для бизнес метрики для второго таргета, если он используется
    
    - scale - делать ли стандартизацию. Подается список колонок для преобразования.
    
    - median - импутация пропусков. Возможные принимаемые значения:
        - 'median' - тогда на train куске рассчитываются медианы, и импутация ими
        - 'min-1' - тогда на train куске рассчитываются минимальные значения - 1, и импутация ими
        - число - если задать число, то пропуски будут заполняться этим числом
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
        - None. Тогда пропуски не импутируются
        По дефолту задано значение 'median'
    - high_outlier - импутация выбросов вверх. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 99 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
        - None. Тогда выбросы вверх не импутируются.
    - low_outlier - импутация выбросов вниз. Возможные принимаемые значения:
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 1 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
        - None. Тогда выбросы вниз не импутируются. 
    - cols_outlier - Подается список колонок для импутации выбросов
    - k_logs - частота вывода промежуточных результатов. По умолчанию выводится каждый 10 результат 
    - file_png - имя файла для сохранения графиков
    ВАЖНО!!!!!!!!!!!! Для поиска по сетке не нужно подавать данные с импутированными пропусками/выбросами! Более того, не стоит 
    использовать attribute_list, так как на каждом разбиении должна считаться собственная импутация.
    
    random_state - random_state для биения данных на куски. random_state для модели следует подавать в словаре для обучения!!!!

    Возвращает таблицу meta с сеткой и показателями. 


    """
    
    def max_prof_corve(y_true, y_score, simple_score, business_dictionary, pos_label=None, sample_weight=None):
    
        """

        Создает векторы для отрисовки кривой max_profit.
        Пример использования:

        b_auc, tp_fps_auc, cut_auc, best_auc, cutoff_auc = max_prof_corve(y_2, auc_test_pred, simple_b_score_crm, business_dictionary)

        plt.figure(figsize = (10, 10))

        plt.title('Business score test')

        plt.plot(tp_fps_auc, b_auc, color='green',
                 lw=lw, label='maxProfit AUC model')

        """

        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score = b_score.max()
        cut_off = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
        #return  tns, fns, fps, tps, y_score[threshold_idxs]
        return b_score, tp_fp, y_score[threshold_idxs], best_score, cut_off
    
    def b_score_train_and_test(y_true, y_score, y_test, y_test_score, simple_score, business_dictionary, pos_label=None, 
                               sample_weight=None):
        """
        ----------
        y_true : array, shape = [n_samples]
            True targets of binary classification
        y_score : array, shape = [n_samples]
            Estimated probabilities or decision function
        y_test: True TEST targets
        y_test_score: predictions on target
        pos_label : int or str, default=None
            The label of the positive class
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        Returns

            !!!!ВАЖНО!!!!

            В функцию надо подавать как прогнозы и истинные метки трейна, так и прогнозы и истинные метки теста, так как на трейне
            рассчитывается оптимальный порог, по которому считается бизнес метрика, но итоговые результаты и выводы о бизнес метрике надо
            делать на ТЕСТЕ!

        """
        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        #ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score_max = b_score.max()
        cut_off_max = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
        #idx_max = max(np.where(tp_fp <= t0)[0])
        #best_score_thr = b_score[idx_max]
        #cut_off_thr = y_score[threshold_idxs][idx_max]
        #return  tns, fns, fps, tps, y_score[threshold_idxs]

        y_best_test_max = pd.Series(np.where(y_test_score >= cut_off_max , 1, 0))
        #y_best_test_thr = pd.Series(np.where(y_test_score >= cut_off_thr  , 1, 0))

        _tn, _fp, _fn, _tp = metrics.confusion_matrix(y_test, y_best_test_max).ravel()
                    #m_s*fund*tps - c*k*tp_fp*y_true.size 
        b_best_max = simple_score(_tn = _tn, _fp = _fp, _fn = _fn, _tp = _tp, **business_dictionary) 

        return best_score_max, cut_off_max, b_best_max#, best_score_thr, cut_off_thr # b_score , tns, fns, fps, tps, y_score[threshold_idxs]
    
    if not callable(pass_model):
        return 'Error! Model should be callable'

    data = X.join(Y)
   
    target = data.columns[-1]
    
    max_target = data.groupby(sort_by_var).aggregate({target: 'max'})
    max_target = max_target.reset_index()
    
    data = pd.merge(data, max_target, on = sort_by_var, suffixes = ["", "_max"])  
    
    target1 = target+"_max"
    
    list_of_vars_for_strat1 = list_of_vars_for_strat.copy()
    if len(list_of_vars_for_strat1) == 0:
        list_of_vars_for_strat1 = [target1]
    if target in list_of_vars_for_strat1:
        list_of_vars_for_strat1.remove(target)
        list_of_vars_for_strat1.append(target1)
    else:
        list_of_vars_for_strat1.append(target1)
        
    for i in list_of_vars_for_strat1:
        if i == list_of_vars_for_strat1[0]:
            data['For_stratify'] = data[i].astype('str')
        else:
            data['For_stratify'] += data[i].astype('str')

    data_nodup = data[[sort_by_var, 'For_stratify', target1]].drop_duplicates(subset = sort_by_var)
    
    cross_val = StratifiedKFold(n_splits=n_folds, shuffle = True, random_state = random_state)

    fold_dic = {}
    f_n = 0
    for idx_train, idx_test in cross_val.split(data_nodup[sort_by_var], data_nodup['For_stratify']):

        xtrain_id, xtest_id = data_nodup.iloc[idx_train][sort_by_var], data_nodup.iloc[idx_test][sort_by_var]
        xtrain = data[data[sort_by_var].isin(xtrain_id)].copy()
        train_index = xtrain.index
        ytrain = data.iloc[train_index][target].copy()

        xtest = data[data[sort_by_var].isin(xtest_id)].copy()
        test_index = xtest.index
        ytest = data.iloc[test_index][target].copy()

        print('1', datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"),  sum(xtrain.memory_usage(deep=True))/1024**2, 
              sum(xtest.memory_usage(deep=True))/1024**2)

        xtrain.drop(list_of_vars_for_strat1, axis = 1, inplace = True)
        xtrain.drop(sort_by_var, axis = 1, inplace = True)
        xtrain.drop(target, axis = 1, inplace = True)
        if target1 in xtrain.columns:
            xtrain.drop(target1, axis = 1, inplace = True)
        xtrain.drop('For_stratify', axis = 1, inplace = True)

        xtest.drop(list_of_vars_for_strat1, axis = 1, inplace = True)
        xtest.drop(sort_by_var, axis = 1, inplace = True)
        xtest.drop(target, axis = 1, inplace = True)
        if target1 in xtest.columns:
            xtest.drop(target1, axis = 1, inplace = True)
        xtest.drop('For_stratify', axis = 1, inplace = True)

        if second_target != None:
            y_train_2 = xtrain[second_target]
            y_test_2 = xtest[second_target]
            xtrain.drop(second_target, axis = 1, inplace = True)
            xtest.drop(second_target, axis = 1, inplace = True)

        train_с = xtrain.columns
        train_ind = xtrain.index
        test_с = xtest.columns
        test_ind = xtest.index

        for oo in xtrain.columns:
            if median != None:
                if median == 'median':
                    medians = xtrain[oo].median(skipna = True)
                elif median == 'min-1':
                    medians = xtrain[oo].min(skipna = True)-1
                elif type(attribute_list) != type(None) and median in attribute_list.columns:
                    medians = list(attribute_list.loc[attribute_list[var_col] == oo, median])[0]
                else:
                    medians = median

            if high_outlier != None:
                if oo in cols_outlier:
                    if type(attribute_list) != type(None) and high_outlier in attribute_list.columns:
                        to_replace_high = list(attribute_list.loc[attribute_list[var_col] == oo, high_outlier])[0]
                    else:
                        to_replace_high = np.nanpercentile(xtrain[oo], high_outlier)

            if low_outlier != None:
                if oo in cols_outlier:
                    if type(attribute_list) != type(None) and low_outlier in attribute_list.columns:
                        to_replace_low = list(attribute_list.loc[attribute_list[var_col] == oo, low_outlier])[0]
                    else:
                        to_replace_low = np.nanpercentile(xtrain[oo], low_outlier)

            if median != None:
                xtrain[oo] = xtrain[oo].fillna(medians)
                xtest[oo] = xtest[oo].fillna(medians)
            if high_outlier != None:
                if oo in cols_outlier:
                    xtrain.loc[xtrain[oo] > to_replace_high, oo] = to_replace_high
                    xtest.loc[xtest[oo] > to_replace_high, oo] = to_replace_high
            if low_outlier != None:
                if oo in cols_outlier:
                    xtrain.loc[xtrain[oo] < to_replace_low, oo] = to_replace_low
                    xtest.loc[xtest[oo] < to_replace_low, oo] = to_replace_low

            # if oo == 'TOTAL_INCOME_CURR':
            #     print('max_', xtrain['TOTAL_INCOME_CURR'].max(), to_replace_high, to_replace_low)
            
        print('2', datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"), 
              sum(xtrain.memory_usage(deep=True))/1024**2, 
              sum(xtest.memory_usage(deep=True))/1024**2
             )
        
        for ccc in yeo_johnson:
            #   print(datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"), ccc)
            #   if ccc == 'TOTAL_INCOME_CURR':
            #       print(xtrain[[ccc]].describe())
            max_min = xtrain[ccc].max() - xtrain[ccc].min()
            xtrain[ccc] = xtrain[ccc]/max_min
            xtest[ccc] = xtest[ccc]/max_min
            power = PowerTransformer(method = 'yeo-johnson', standardize = False).fit(xtrain[[ccc]])
            xtrain[ccc] = power.transform(xtrain[[ccc]])
            xtest[ccc] = power.transform(xtest[[ccc]])

            # if ccc == 'TOTAL_INCOME_CURR':
            #     print('lambda', power.lambdas_[0])
            #     print(xtrain[[ccc]].describe())
                
        print('3', datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"),  
              sum(xtrain.memory_usage(deep=True))/1024**2, 
              sum(xtest.memory_usage(deep=True))/1024**2)    
        
        if isinstance(scale,list) and len(scale) > 0:
            pr = preprocessing.StandardScaler()
            pr.fit(xtrain[scale])
            xtrain[scale] = pr.transform(xtrain[scale])
            xtest[scale] = pr.transform(xtest[scale])
            
            print('4', datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"),
                  sum(xtrain.memory_usage(deep=True))/1024**2, 
                  sum(xtest.memory_usage(deep=True))/1024**2)
               
            #  power = PowerTransformer(method = 'yeo-johnson', standardize = False).fit(xtrain[numeric_cols])
            #  xtrain[numeric_cols] = power.transform(xtrain[numeric_cols])
            #  xtest[numeric_cols] = power.transform(xtest[numeric_cols])

            #  pr = preprocessing.StandardScaler()
            #  pr.fit(xtrain[numeric_cols])
            #  xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
            #  xtest[numeric_cols] = pr.transform(xtest[numeric_cols])

        
        f_n = f_n + 1
        fold_dic[f_n] = (xtrain, xtest, ytrain, ytest)
    
    
    
    meta_container = pd.DataFrame()
    
    for key in params_dictionary.keys():
        if type(params_dictionary[key]) != np.ndarray and type(params_dictionary[key]) != list:
            check = []
            check.append(params_dictionary[key])
            params_dictionary[key] = check
    
    print(params_dictionary)
    
    vals = params_dictionary.values()
    
    import itertools

    combs = list(itertools.product(*vals))
    
    k = 0
          
    if draw == True:
        fig, axs = plt.subplots(len(combs), 1, figsize=(2.2, len(combs)), sharey='all', sharex='all', constrained_layout=True)
        fig.suptitle('Graphs of Max profit', fontsize=3)
        plt.close(fig)
    
    for combination in combs:
        outputs = list(combination)
        dicts = {}
        for position in range(len(combination)):
            dicts[list(params_dictionary.keys())[position]] = combination[position]
        
        parameters = {}
        
        for key in dicts.keys():
            if len(re.findall('weight', key)) == 0:
                parameters[params_to_model[key]] = dicts[key]
            elif len(re.findall('class_weight', key)) > 0:
                parameters[params_to_model[key]] = dicts[key]
            else:
                if len(re.findall('0', key)) > 0:
                    parameters[params_to_model[key]] = {0: dicts[key], 1:1}
                if len(re.findall('1', key)) > 0:
                    parameters[params_to_model[key]] = {0:1, 1:dicts[key]}

        model = pass_model(**parameters)
            
        if second_target != None:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],    
                'AUC': [],
                'b_best' : [] ,
                'cutoff' : [] ,
                'Bad_Rate': [],
                'ScoreF1_second_target': [],
                'Acc_second_target': [],
                'Pre_second_target': [],
                'Rec_second_target': [] ,
                'APS_second_target': [],
                'Brier_score_second_target': [],
                'AUC_second_target': [],
                'b_best_second_target' : [] ,
                'cutoff_second_target' : [] ,
                #'b_best_thr_second_target' :[] ,
                #'cutoff_thr_second_target' : [],
                'b_best_second_target': [],
                'b_best_thr_second_target': [],
                'Bad_Rate_second_target': []
                #, 'AR' : [] ,
                #'def_6' : []
                #'b_best_norm' : []
                
                }
    
        else:
                
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],
                'AUC': [],
                'b_best' : [] ,
                'cutoff' : [],
                'Bad_Rate': []
                #'b_best_thr' :[] ,
                #'cutoff_thr' : []
                #, 'AR' : [] ,
                #'def_6' : []
                #'b_best_norm' : []
                
            }
                
        if draw == True:
            color=iter(cm.rainbow(np.linspace(0, 1, n_folds)))
            fig_each, ax_each = plt.subplots(1, 1, figsize=(10, 5))    
                
        for f_n in fold_dic:
            
            (xtrain, xtest, ytrain, ytest) = fold_dic[f_n]
#             print(f_n, id(xtrain))
            
            model.fit(xtrain, ytrain)
            yhat_test = model.predict(xtest)
                
            yhat_test_proba = model.predict_proba(xtest)[:,1]
            
            yhat_train_proba = model.predict_proba(xtrain)[:,1]
                              
            scores['ScoreF1'].append(metrics.f1_score(ytest, yhat_test))
            scores['Acc'].append(metrics.accuracy_score(ytest, yhat_test))
            scores['Pre'].append(metrics.precision_score(ytest, yhat_test))
            scores['Rec'].append(metrics.recall_score(ytest, yhat_test)) 
            scores['APS'].append(metrics.average_precision_score(ytest, yhat_test_proba))
            scores['Brier_score'].append(metrics.brier_score_loss(ytest, yhat_test_proba))
            scores['AUC'].append(metrics.roc_auc_score(ytest, yhat_test_proba))
            scores['Bad_Rate'].append(ytest.value_counts()[1]/len(ytest))
            # находим лучший cut-off по трейн и применяем его для тест!!
            #best_score_max, cut_off_max, best_score_thr, cut_off_thr
                
                
            if need_business == True:
                b_best_train_max, cutoff_train_max, b_best_max = b_score_train_and_test(ytrain,
                                                                    yhat_train_proba, ytest, 
                                                                yhat_test_proba, simple_b_score, business_dict)
                if draw == True:
                    b_score_array, approval_rate, cutoff, best_sc, best_cutoff = max_prof_corve(ytest, 
                                                                                                yhat_test_proba, 
                                                                                                simple_b_score,
                                                                                                business_dict)
                    b_score_array_train, approval_rate_train, cutoff_train, best_sc_train, best_cutoff_train = max_prof_corve(ytrain, 
                                                                                                            yhat_train_proba, 
                                                                                                            simple_b_score,
                                                                                                            business_dict)
                    if draw_by_approval_rate == False:
                        x_plot = cutoff
                        y_plot = b_score_array #/len(y_test)
                        c = next(color) 
                        x_plot_train = cutoff_train
                        y_plot_train = b_score_array_train #/len(y_test)
                        
                        if k/k_logs == int(k/k_logs) or k == 1:
                            
                            ax_each.scatter(x_plot, y_plot, s = 0.01, color=c, alpha=0.1)
                            ax_each.scatter(x_plot_train, y_plot_train, s = 0.01, color=c, alpha=0.1)
                            ax_each.plot([best_cutoff_train, best_cutoff_train], [0, best_sc_train], '--', color=c, alpha=0.8)
                            ax_each.plot([best_cutoff, best_cutoff], [0, best_sc], '--', color=c, alpha=0.8)

                        axs[k].scatter(x_plot, y_plot, s = 0.01, color=c, alpha=0.1, linewidth=0.2)
                        axs[k].scatter(x_plot_train, y_plot_train, s = 0.01, color=c, alpha=0.1, linewidth=0.2)
                        axs[k].plot([best_cutoff_train, best_cutoff_train], [0, best_sc_train], '--', linewidth=0.2, color=c, alpha=0.8)
                        axs[k].plot([best_cutoff, best_cutoff], [0, best_sc], '--', linewidth=0.2, color=c, alpha=0.8)
                        axs[k].tick_params(labelsize=2, which='both', labelbottom=True, labelleft=True, width = 0.2)
                       
                    else:
                        
                        x_plot = approval_rate
                        y_plot = b_score_array
                        c = next(color) 
                        x_plot_train = approval_rate_train
                        y_plot_train = b_score_array_train #/len(y_test)
                        
                        if k/k_logs == int(k/k_logs) or k == 1:
                            ax_each.scatter(x_plot, y_plot, s = 0.01, color=c, alpha=0.1)
                            ax_each.scatter(x_plot_train, y_plot_train, s = 0.01, color=c, alpha=0.1)

                        axs[k].scatter(x_plot, y_plot, s = 0.01, color=c, alpha=0.1, linewidth=0.2)
                        axs[k].scatter(x_plot_train, y_plot_train, s = 0.01, color=c, alpha=0.1, linewidth=0.2)
                        axs[k].tick_params(labelsize=2, which='both', labelbottom=True, labelleft=True, width = 0.2)
                                        
                scores['b_best'].append(b_best_max)
                scores['cutoff'].append(cutoff_train_max)
                                    
            if second_target != None:
                    
                scores['ScoreF1_second_target'].append(metrics.f1_score(y_test_2, yhat_test))
                scores['Acc_second_target'].append(metrics.accuracy_score(y_test_2, yhat_test))
                scores['Pre_second_target'].append(metrics.precision_score(y_test_2, yhat_test))
                scores['Rec_second_target'].append(metrics.recall_score(y_test_2, yhat_test)) 
                scores['APS_second_target'].append(metrics.average_precision_score(y_test_2, yhat_test_proba))
                scores['Brier_score_second_target'].append(metrics.brier_score_loss(y_test_2, yhat_test_proba))
                scores['AUC_second_target'].append(metrics.roc_auc_score(y_test_2, yhat_test_proba))
                scores['Bad_Rate_second_target'].append(y_test_2.value_counts()[1]/len(y_test_2))
                # находим лучший cut-off по трейн и применяем его для тест!!
                #best_score_max, cut_off_max, best_score_thr, cut_off_thr
                    
                if need_business == True:
                    b_best_train_max, cutoff_train_max, b_best_max  = b_score_train_and_test(y_train_2, 
                                                            yhat_train_proba, y_test_2, yhat_test_proba, simple_b_score,
                                                                                     business_dict_sec)           

                    scores['b_best_second_target'].append(b_best_max)
                    scores['cutoff_second_target'].append(cutoff_train_max)
        
        if need_business == True:
            if draw == True:
                if draw_by_approval_rate == False:
                    if k/k_logs == int(k/k_logs) or k == 1:
                        ax_each.set_xlabel('Treshold')
                        ax_each.set_ylabel('Profit')
                        ax_each.set_title(parameters)
                        plt.show()

                    axs[k].set_xlabel('Treshold', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_ylabel('Profit', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_title(parameters, fontdict = {'fontsize': 2, 'fontweight' : 2})

                else:
                    if k/k_logs == int(k/k_logs) or k == 1:
                        ax_each.set_xlabel('Approval Rate')
                        ax_each.set_ylabel('Profit')
                        ax_each.set_title(parameters)
                        plt.show()

                    axs[k].set_xlabel('Approval Rate', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_ylabel('Profit', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_title(parameters, fontdict = {'fontsize': 2, 'fontweight' : 2})
        
        plt.close()
        scores = pd.DataFrame(scores)
        outputs.extend(scores.mean().tolist())
        outputs.extend(scores.std().tolist())
        score_cols = scores.columns.tolist()
        score_cols_std = [c+'_std' for c in score_cols]
        cols = list(params_dictionary.keys()) + score_cols + score_cols_std
        outputs = pd.DataFrame([outputs], columns=cols)
        meta_container = meta_container.append(outputs)

        if k/k_logs == int(k/k_logs) or k == 1:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
                
            print(20*'-',tm, k, 20*'-', '\n',
                  'Параметры:', parameters, '\n', 
                  'Среднее значение бизнес метрики =', scores['b_best'].mean(), '\n', 
                  'Среднее значение AUC =',            scores['AUC'].mean()
                 ) 
        k += 1
                      
    meta_container.reset_index(drop=True, inplace=True)
    fig.savefig(file_png, dpi = 300)
    return meta_container
    
    
def linear_calibration(y_true, y_pred, ct, n_bins, strategy, mpv_strategy, plt_show=False):
  
    """
    Функция для калибровки с помощью линейной регрессии логарифма шансов. 
    Выводит коэффициенты a и b, медианное значение pd/скорингового балла (на выборке, на которой калибруем модель), 
    а также рисует график odds-pd и выводит обученную модель.
    
    Методика - G:/New Risk Management/Decision Science/Knowledge Base/Calibration/FW Шкалирование и калибровка.msg
    Пример использования - G:/New Risk Management/Decision Science/kgrushin/PD Models/Calibration/Калибровка BL19_BEELINE v2
    
    y_true - истинные метки выборки, на которую калибруем модель
    y_pred - предсказанные калибруемой моделью PD на той же выборке
    ct - значение центральной тенденции. Например DR на выборке oot.
    
    n_bins - количество бинов, на которое разбиваем выборку, на которую калибруем модель
    strategy - принимает значение 'uniform' и 'quantile'. 
               uniform - бьем на бины равной ширины, quantile - бьем на бины с равным количеством наблюдений
   mpv_strategy - поправка PD. 'median' or 'average'
    plt_show - если True - рисуем график отношения шансов к PD (Score), по умолчанию = False
    
    """
    
    def calibration_curve_v2(y_true, y_prob, normalize=False, n_bins=5,
                      strategy='uniform'):
        """Compute true and predicted probabilities for a calibration curve.
        The method assumes the inputs come from a binary classifier.
        Calibration curves may also be referred to as reliability diagrams.
        Read more in the :ref:`User Guide <calibration>`.
        Parameters
        ----------
        y_true : array, shape (n_samples,)
            True targets.
        y_prob : array, shape (n_samples,)
            Probabilities of the positive class.
        normalize : bool, optional, default=False
            Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
            a proper probability. If True, the smallest value in y_prob is mapped
            onto 0 and the largest one onto 1.
        n_bins : int
            Number of bins. A bigger number requires more data. Bins with no data
            points (i.e. without corresponding values in y_prob) will not be
            returned, thus there may be fewer than n_bins in the return value.
        strategy : {'uniform', 'quantile'}, (default='uniform')
            Strategy used to define the widths of the bins.
            uniform
                All bins have identical widths.
            quantile
                All bins have the same number of points.
        Returns
        -------
        prob_true : array, shape (n_bins,) or smaller
            The true probability in each bin (fraction of positives).
        prob_pred : array, shape (n_bins,) or smaller
            The mean predicted probability in each bin.
        References
        ----------
        Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
        Probabilities With Supervised Learning, in Proceedings of the 22nd
        International Conference on Machine Learning (ICML).
        See section 4 (Qualitative Analysis of Predictions).
        """
        y_true = column_or_1d(y_true)
        y_prob = column_or_1d(y_prob)
        check_consistent_length(y_true, y_prob)

        if normalize:  # Normalize predicted values into interval [0, 1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        elif y_prob.min() < 0 or y_prob.max() > 1:
            raise ValueError("y_prob has values outside [0, 1] and normalize is "
                             "set to False.")

        labels = np.unique(y_true)
        if len(labels) > 2:
            raise ValueError("Only binary classification is supported. "
                             "Provided labels %s." % labels)
        y_true = label_binarize(y_true, labels)[:, 0]

        if strategy == 'quantile':  # Determine bin edges by distribution of data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.percentile(y_prob, quantiles * 100)
            bins[-1] = bins[-1] + 1e-8
        elif strategy == 'uniform':
            bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        else:
            raise ValueError("Invalid entry to 'strategy' input. Strategy "
                             "must be either 'quantile' or 'uniform'.")

        binids = np.digitize(y_prob, bins) - 1

        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        prob_true = (bin_true[nonzero] / bin_total[nonzero])
        prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

        return prob_true, prob_pred
    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    
    fop, mpv = calibration_curve_v2(y_true, y_pred, normalize=False, n_bins=n_bins, strategy=strategy)
    if mpv_strategy=='median':
        mpv_med = np.median(mpv)
    elif mpv_strategy=='average':
        mpv_med = np.average(mpv)
    sdr = y_true.sum()/y_true.shape[0] 

    fop_sc_lst = []
    for i in range(len(fop)):
        dr_i = fop[i]
        fop_sc = (dr_i*ct/sdr) / ((1-dr_i)*((1-ct)/(1-sdr))+dr_i*ct/sdr)
        fop_sc_lst.append(fop_sc)

    odds_lst = []
    score_lst = []
    for i in range(len(fop_sc_lst)):
        dr_i_sc = fop_sc_lst[i]
        if dr_i_sc==1:
            raise ValueError(f"Error. DRi = 1. В бине {i} все наблюдения принадлежат к одному классу. Попробуйте уменьшить количество бинов.")
            # odds = np.log((1-dr_i_sc+0.0001)/dr_i_sc)
        if dr_i_sc==0:
            raise ValueError(f"Error. DRi = 0. В бине {i} все наблюдения принадлежат к одному классу. Попробуйте уменьшить количество бинов.")
            # odds = np.log((1-dr_i_sc-0.0001)/dr_i_sc+0.0001)
        else:
            odds = np.log((1-dr_i_sc)/dr_i_sc)
        score = mpv[i] - mpv_med
        odds_lst.append(odds)
        score_lst.append(score)
    odds_lst = pd.DataFrame(odds_lst) 
    score_lst = pd.DataFrame(score_lst) 

    lin_reg = LinearRegression()
    lin_reg.fit(score_lst, odds_lst)

    a = lin_reg.intercept_[0]
    b = lin_reg.coef_[0][0]
    
    if plt_show:
        plt.figure(figsize=[6,4])
        plt.plot(score_lst.loc[:,0], odds_lst.loc[:,0], "s-", label='', alpha=1)
        plt.tight_layout()
        plt.grid(True, alpha=0.65)
        plt.xlabel('PD', fontsize=10)
        plt.ylabel('Odds', fontsize=10)
        plt.title('График Odds-PD', fontsize=10)
        #plt.savefig(savefig+'/SIGM_CALIB_PLOT_'+seg_name+'_'+score_partn, bbox_inches ='tight', pad_inches = 0.1)
        plt.show()

    return a, b, mpv_med, lin_reg

# In[ ]:

def calibration_curve_v2(y_true, y_prob, normalize=False, n_bins=5,
                      strategy='uniform'):
    """Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier.
    Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data. Bins with no data
        points (i.e. without corresponding values in y_prob) will not be
        returned, thus there may be fewer than n_bins in the return value.
    strategy : {'uniform', 'quantile'}, (default='uniform')
        Strategy used to define the widths of the bins.
        uniform
            All bins have identical widths.
        quantile
            All bins have the same number of points.
    Returns
    -------
    prob_true : array, shape (n_bins,) or smaller
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins,) or smaller
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)
    y_true = label_binarize(y_true, labels)[:, 0]

    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return prob_true, prob_pred
# In[ ]:
def linear_calib_use_model_data(train, test, oot, target, model,
                                our_outputs, strategy, n_bins, sample_to_calib, mpv_strategy, 
                                text_file = 'Calibration_SAS.txt',
                                preprocessed = True,
                                selected_features = None, necessary_fields_upper = None, categorials_data_fin = None, 
                                attribute_list_model = None,
                                preproc = None, grade_lst = None, 
                                n_split = 10, 
                                mediana = 'val_mediana', right_border = 'right_border', left_border = 'left_border',
                                var_col = 'attribute', inverse = True, base_points = 700, 
                                score_points_to_change = 50, odds_change = 2
                                ):
    ''' 
    Функция для использования калибровки и вывода результатов для бленд-моделей 2019 с Билайном
    
    train, test, oot - сырые датасеты без обработки с таргетом 
    target - название целевой переменной
    selected_features - датафрейм с отобранными переменными. Необходимым является поле new variable
    necessary_fields_upper - необходимые переменные (список) 
    categorials_data_fin - датафрейм с категориальными переменными. Ключевым является поле feature
    attribute_list_model - датафрейм со статистиками 
    preproc - обученный файл с препроцессингом (dat файл)
    model - обученный файл с моделью (dat файл)
    our_outputs - путь, куда сохранять картинки
    strategy - калибровка по квантилям ('quantile'), либо по бинам ('uniform')
    n_bins - количество бинов для калибровки 
    sample_to_calib - выборка, используемая для калибровки 'train/test/oot'
    mpv_strategy - 'median' или 'average' корректировка для калибровки 
    grade_lst - названия для грейдов от лучшего к худшему
    n_split - количество делений (= количество грейдов)
    preprocessed - предобработаны ли данные
    mediana - название поля в датафрейме attribute_list_model для импутации медианами
    right_border - название поля в датафрейме attribute_list_model для обрезания справа
    left_border - название поля в датафрейме attribute_list_model для обрезания слева
    var_col - название поля в датафрейме attribute_list_model для нахождения переменной
    inverse - ведет ли больший скор к меньшему PD? по умолчанию True
    base_points - базовый скор (по умолчанию и на период 14.01.2020). 
    score_points_to_change - сколько нужно скор баллов для изменения (удвоения/утроения и пр) шансов. По умолчанию 50
    odds_change - изменение шансов (2 - удвоение, 3 - утроение и тд). По умолчанию 2
    
    Возвращает значения коэффициентов a и b, медианную/среднюю поправку pd, и датафрейм со сводными результатами 
    грейдирования по равнозаполненным сплитам (средний dr, min/max/avg pd, границы)
    
    '''
    from sklearn.calibration import calibration_curve 
    def write_and_print(file_name, content, n = True, mode = 'a'):

        if n:
            content = content + '\n'

        with open(file_name, mode = mode, encoding='utf8') as file:
            file.write(content)

        return content
    
    
    def linear_calibration(y_true, y_pred, ct, n_bins, strategy, mpv_strategy, plt_show=False):
  
        """
        Функция для калибровки с помощью линейной регрессии логарифма шансов. 
        Выводит коэффициенты a и b, медианное значение pd/скорингового балла (на выборке, на которой калибруем модель), 
        а также рисует график odds-pd и выводит обученную модель.

        Методика - G:/New Risk Management/Decision Science/Knowledge Base/Calibration/FW Шкалирование и калибровка.msg
        Пример использования - G:/New Risk Management/Decision Science/kgrushin/PD Models/Calibration/Калибровка BL19_BEELINE v2

        y_true - истинные метки выборки, на которую калибруем модель
        y_pred - предсказанные калибруемой моделью PD на той же выборке
        ct - значение центральной тенденции. Например DR на выборке oot.

        n_bins - количество бинов, на которое разбиваем выборку, на которую калибруем модель
        strategy - принимает значение 'uniform' и 'quantile'. 
                   uniform - бьем на бины равной ширины, quantile - бьем на бины с равным количеством наблюдений
       mpv_strategy - поправка PD. 'median' or 'average'
        plt_show - если True - рисуем график отношения шансов к PD (Score), по умолчанию = False

        """

        def calibration_curve_v2(y_true, y_prob, normalize=False, n_bins=5,
                          strategy='uniform'):
            """Compute true and predicted probabilities for a calibration curve.
            The method assumes the inputs come from a binary classifier.
            Calibration curves may also be referred to as reliability diagrams.
            Read more in the :ref:`User Guide <calibration>`.
            Parameters
            ----------
            y_true : array, shape (n_samples,)
                True targets.
            y_prob : array, shape (n_samples,)
                Probabilities of the positive class.
            normalize : bool, optional, default=False
                Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
                a proper probability. If True, the smallest value in y_prob is mapped
                onto 0 and the largest one onto 1.
            n_bins : int
                Number of bins. A bigger number requires more data. Bins with no data
                points (i.e. without corresponding values in y_prob) will not be
                returned, thus there may be fewer than n_bins in the return value.
            strategy : {'uniform', 'quantile'}, (default='uniform')
                Strategy used to define the widths of the bins.
                uniform
                    All bins have identical widths.
                quantile
                    All bins have the same number of points.
            Returns
            -------
            prob_true : array, shape (n_bins,) or smaller
                The true probability in each bin (fraction of positives).
            prob_pred : array, shape (n_bins,) or smaller
                The mean predicted probability in each bin.
            References
            ----------
            Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
            Probabilities With Supervised Learning, in Proceedings of the 22nd
            International Conference on Machine Learning (ICML).
            See section 4 (Qualitative Analysis of Predictions).
            """
            y_true = column_or_1d(y_true)
            y_prob = column_or_1d(y_prob)
            check_consistent_length(y_true, y_prob)

            if normalize:  # Normalize predicted values into interval [0, 1]
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            elif y_prob.min() < 0 or y_prob.max() > 1:
                raise ValueError("y_prob has values outside [0, 1] and normalize is "
                                 "set to False.")

            labels = np.unique(y_true)
            if len(labels) > 2:
                raise ValueError("Only binary classification is supported. "
                                 "Provided labels %s." % labels)
            y_true = label_binarize(y_true, labels)[:, 0]

            if strategy == 'quantile':  # Determine bin edges by distribution of data
                quantiles = np.linspace(0, 1, n_bins + 1)
                bins = np.percentile(y_prob, quantiles * 100)
                bins[-1] = bins[-1] + 1e-8
            elif strategy == 'uniform':
                bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
            else:
                raise ValueError("Invalid entry to 'strategy' input. Strategy "
                                 "must be either 'quantile' or 'uniform'.")

            binids = np.digitize(y_prob, bins) - 1

            bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
            bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
            bin_total = np.bincount(binids, minlength=len(bins))

            nonzero = bin_total != 0
            prob_true = (bin_true[nonzero] / bin_total[nonzero])
            prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

            return prob_true, prob_pred

        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression

        fop, mpv = calibration_curve_v2(y_true, y_pred, normalize=False, n_bins=n_bins, strategy=strategy)
        if mpv_strategy=='median':
            mpv_med = np.median(mpv)
        elif mpv_strategy=='average':
            mpv_med = np.average(mpv)
        sdr = y_true.sum()/y_true.shape[0] 

        fop_sc_lst = []
        for i in range(len(fop)):
            dr_i = fop[i]
            fop_sc = (dr_i*ct/sdr) / ((1-dr_i)*((1-ct)/(1-sdr))+dr_i*ct/sdr)
            fop_sc_lst.append(fop_sc)

        odds_lst = []
        score_lst = []
        for i in range(len(fop_sc_lst)):
            dr_i_sc = fop_sc_lst[i]
            if dr_i_sc==1:
                raise ValueError(f"Error. DRi = 1. В бине {i} все наблюдения принадлежат к одному классу. Попробуйте уменьшить количество бинов.")
                # odds = np.log((1-dr_i_sc+0.0001)/dr_i_sc)
            if dr_i_sc==0:
                raise ValueError(f"Error. DRi = 0. В бине {i} все наблюдения принадлежат к одному классу. Попробуйте уменьшить количество бинов.")
                # odds = np.log((1-dr_i_sc-0.0001)/dr_i_sc)
            else:
                odds = np.log((1-dr_i_sc)/dr_i_sc)
            score = mpv[i] - mpv_med
            odds_lst.append(odds)
            score_lst.append(score)
        
        odds_lst = pd.DataFrame(odds_lst)
        score_lst = pd.DataFrame(score_lst)

        lin_reg = LinearRegression()
        lin_reg.fit(score_lst, odds_lst)

        a = lin_reg.intercept_[0]
        b = lin_reg.coef_[0][0]

        if plt_show:
            plt.figure(figsize=[6,4])
            plt.plot(score_lst.loc[:,0], odds_lst.loc[:,0], "s-", label='', alpha=1)

            plt.tight_layout()
            plt.grid(True, alpha=0.65)
            plt.xlabel('PD', fontsize=10)
            plt.ylabel('Odds', fontsize=10)
            plt.title('График Odds-PD', fontsize=10)
            #plt.savefig(savefig+'/SIGM_CALIB_PLOT_'+seg_name+'_'+score_partn, bbox_inches ='tight', pad_inches = 0.1)
            plt.show()

        return a, b, mpv_med, lin_reg
    
    def score_calculation(y_pred, base_points = 700, score_points_to_change = 50, 
                          odds_change = 2, inverse = True):
        '''
        y_pred - вектор с оценками PD
        base_points - базовый скор (по умолчанию и на период 14.01.2020)
        score_points_to_change - сколько нужно скор баллов для изменения (удвоения/утроения и пр) шансов. По умолчанию 50.
        odds_change - изменение шансов (2 - удвоение, 3 - утроение и тд). По умолчанию 2
        inverse - ведет ли больший скор к меньшему PD? по умолчанию True
        '''
        if inverse == True:
            y_pred_score = 700+ 50*(np.log((1/y_pred - 1)/10)/np.log(2))
        else:
            y_pred_score = 700+ 50*(np.log((y_pred/(1-y_pred))/10)/np.log(2))
        return y_pred_score
    
    def score_calculation_to_pd(y_score, base_points = 700, score_points_to_change = 50, 
                               odds_change = 2, inverse = True):
        '''
        y_score - вектор с оценками score
        base_points - базовый скор (по умолчанию и на период 14.01.2020)
        score_points_to_change - сколько нужно скор баллов для изменения (удвоения/утроения и пр) шансов. По умолчанию 50.
        odds_change - изменение шансов (2 - удвоение, 3 - утроение и тд). По умолчанию 2
        inverse - ведет ли больший скор к меньшему PD? по умолчанию True
        '''
        if inverse == True:
            y_pd = 1/(10*2**((y_score-700)/50)+1)    
        else:
            y_pd = 1-1/(10*2**((y_score-700)/50)+1)   
        return y_pd
    
    
    if preprocessed == False:
        target = target.upper()

        train.columns = [col.upper() for col in train.columns]
        test.columns = [col.upper() for col in test.columns]
        oot.columns = [col.upper() for col in oot.columns]
        

        train = turn_variables_with_values(train, selected_features)
        test = turn_variables_with_values(test, selected_features)
        oot = turn_variables_with_values(oot, selected_features)


        y_true_train = train[target]
        y_true_test = test[target]
        y_true_oot = oot[target]
        y_true = pd.concat([y_true_train, y_true_test, y_true_oot], axis=0, ignore_index=True)

        train.drop(target, axis = 1, inplace = True)
        test.drop(target, axis = 1, inplace = True)
        oot.drop(target, axis = 1, inplace = True)

        list_of_columns = selected_features['new variable'].to_list()
        list_of_columns_no_sys = list_of_columns.copy()
        for i in necessary_fields_upper:
            if i in list_of_columns_no_sys:
                list_of_columns_no_sys.remove(i)

        list_of_categories_sel = attribute_list_model.loc[(attribute_list_model['count_dist'] == 2), 'attribute'].to_list() 

        for i in list_of_columns_no_sys:
            if i in categorials_data_fin['feature'].to_list():
                list_of_categories_sel.append(i)

        non_outliers = attribute_list_model.loc[attribute_list_model['99%'] == attribute_list_model['1%'], 
                                                    'attribute'].to_list()
        cols_outliers = list_of_columns_no_sys.copy()

        for o in non_outliers:
            if o in cols_outliers:
                cols_outliers.remove(o)

        X_1_2 = data_preprocessing_test(train, y_true_train, necessary_fields_upper, categorial_list=list_of_categories_sel,
                                        drop_technical=True,
                                        attribute_list=attribute_list_model, var_col=var_col,
                                        median=mediana,
                                        high_outlier=right_border, low_outlier=left_border, scale=preproc,
                                        yeo_johnson=None, cols_outlier=cols_outliers)

        X_2_2 = data_preprocessing_test(test, y_true_test, necessary_fields_upper, categorial_list=list_of_categories_sel,
                                        drop_technical=True,
                                        attribute_list=attribute_list_model, var_col=var_col,
                                        median=mediana,
                                        high_outlier=right_border, low_outlier=left_border, scale=preproc,
                                        yeo_johnson=None, cols_outlier=cols_outliers)

        X_3_2 = data_preprocessing_test(oot, y_true_oot, necessary_fields_upper, categorial_list=list_of_categories_sel,
                                        drop_technical=True,
                                        attribute_list=attribute_list_model, var_col=var_col,
                                        median=mediana,
                                        high_outlier=right_border, low_outlier=left_border, scale=preproc,
                                        yeo_johnson=None, cols_outlier=cols_outliers)
        
        
    
    if preprocessed == True:
        target = target.upper()

        train.columns = [col.upper() for col in train.columns]
        test.columns = [col.upper() for col in test.columns]
        oot.columns = [col.upper() for col in oot.columns]
        
        y_true_train = train[target]
        y_true_test = test[target]
        y_true_oot = oot[target]
        y_true = pd.concat([y_true_train, y_true_test, y_true_oot], axis=0, ignore_index=True)

        train.drop(target, axis = 1, inplace = True)
        test.drop(target, axis = 1, inplace = True)
        oot.drop(target, axis = 1, inplace = True)
        
        X_1_2 = train
        X_2_2 = test
        X_3_2 = oot
    
    y_pred_train = model.predict_proba(X_1_2)[:, 1]
    y_pred_test = model.predict_proba(X_2_2)[:, 1]
    y_pred_oot = model.predict_proba(X_3_2)[:, 1]
    y_pred = np.concatenate((y_pred_train, y_pred_test, y_pred_oot), axis=0)
    
    print("Размер данных TRAIN: " + str(X_1_2.shape))
    print("Размер данных TEST: " + str(X_2_2.shape))
    print("Размер данных OOT: " + str(X_3_2.shape))
    
    y_pred_score = score_calculation(y_pred, base_points, score_points_to_change, odds_change, inverse)
    y_pred_train_score = score_calculation(y_pred_train, base_points, score_points_to_change, odds_change, inverse)
    y_pred_test_score = score_calculation(y_pred_test, base_points, score_points_to_change, odds_change, inverse)
    y_pred_oot_score = score_calculation(y_pred_oot, base_points, score_points_to_change, odds_change, inverse)
    
    print("Минимальный PD: ",np.min(y_pred))
    print("Максимальный PD: ",np.max(y_pred))

    print("Минимальный SCORE: ",np.min(y_pred_score))
    print("Максимальный SCORE: ",np.max(y_pred_score))

    # Калибруем
    ct = y_true_oot.sum()/y_true_oot.shape[0] # DR на выборке OOT
    
    if sample_to_calib=='test':
        a, b, mpv_med, lin_reg = linear_calibration(y_true_test, y_pred_test, ct, n_bins=n_bins, 
                                                    strategy=strategy, mpv_strategy=mpv_strategy, plt_show=True)
    elif sample_to_calib=='oot':
        a, b, mpv_med, lin_reg = linear_calibration(y_true_oot, y_pred_oot, ct, n_bins=n_bins, 
                                                    strategy=strategy, mpv_strategy=mpv_strategy, plt_show=True)
    elif sample_to_calib=='train':
        a, b, mpv_med, lin_reg = linear_calibration(y_true_train, y_pred_train, ct, n_bins=n_bins, 
                                                    strategy=strategy, mpv_strategy=mpv_strategy, plt_show=True)

    print("a (Intercept) = ", a)
    print("b = ", b)
    print("median/average = ", mpv_med)
    
    content = 'PD_calib = 1/(1+exp(' + str(a) + '+' + str(b)+'*(prediction_proba -' + str(mpv_med) + ')))'
    print(content)
    
    txt_file = our_outputs+'/'+ text_file
    
    cont = write_and_print(txt_file, content, n = False, mode = 'w')

    y_pred_calib = 1/(1+np.exp(a + b * (y_pred-mpv_med)))
    y_pred_train_calib = 1/(1+np.exp(a + b * (y_pred_train-mpv_med)))
    y_pred_test_calib = 1/(1+np.exp(a + b * (y_pred_test-mpv_med)))
    y_pred_oot_calib = 1/(1+np.exp(a + b * (y_pred_oot-mpv_med)))
        
    # Применяем FICO-преобразование. Для PD=1 корректировку сдвигаем.
    y_pred_calib_w = y_pred_calib.copy()
    y_pred_train_calib_w = y_pred_train_calib.copy()
    y_pred_test_calib_w = y_pred_test_calib.copy()
    y_pred_oot_calib_w = y_pred_oot_calib.copy()
        
    y_pred_calib_w[y_pred_calib_w==1] = y_pred_calib_w[y_pred_calib_w==1]-0.00001
    y_pred_train_calib_w[y_pred_train_calib_w==1] = y_pred_train_calib_w[y_pred_train_calib_w==1]-0.00001
    y_pred_test_calib_w[y_pred_test_calib_w==1] = y_pred_test_calib_w[y_pred_test_calib_w==1]-0.00001
    y_pred_oot_calib_w[y_pred_oot_calib_w==1] = y_pred_oot_calib_w[y_pred_oot_calib_w==1]-0.00001

    
    y_pred_calib_score = score_calculation(y_pred_calib_w, base_points, score_points_to_change, odds_change, inverse)
    y_pred_train_calib_score = score_calculation(y_pred_train_calib_w, base_points, 
                                                 score_points_to_change, odds_change, inverse)
    y_pred_test_calib_score = score_calculation(y_pred_test_calib_w, base_points, 
                                                 score_points_to_change, odds_change, inverse)
    y_pred_oot_calib_score = score_calculation(y_pred_oot_calib_w, base_points, 
                                                 score_points_to_change, odds_change, inverse)
    
    #     y_pred_calib = 1/(1+np.exp(a + b * y_pred))
    #     y_pred_train_calib = 1/(1+np.exp(a + b * y_pred_train))
    #     y_pred_test_calib = 1/(1+np.exp(a + b * y_pred_test))
    #     y_pred_oot_calib = 1/(1+np.exp(a + b * y_pred_oot))

    print("Минимальный калиброванный бленд PD: ",np.min(y_pred_calib))
    print("Максимальный калиброванный бленд PD: ",np.max(y_pred_calib))

    print("Минимальный калиброванный бленд SCORE: ",np.min(y_pred_calib_score))
    print("Максимальный калиброванный бленд SCORE: ",np.max(y_pred_calib_score))

    # График калибровочных кривых до калибровки
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, normalize=False, n_bins=10)
    fop_train, mpv_train = calibration_curve(y_true_train, y_pred_train, normalize=False, n_bins=10)
    fop_test, mpv_test = calibration_curve(y_true_test, y_pred_test, normalize=False, n_bins=10)
    fop_oot, mpv_oot = calibration_curve(y_true_oot, y_pred_oot, normalize=False, n_bins=10)

    plt.figure(figsize=[12,8])
    plt.plot([0, 1], [0, 1], "k:", label="Идеальная калибровка")
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Train+Test+OOT до калибровки', alpha=0.5)
    plt.plot(mpv_train, fop_train, "s-", label='Train до калибровки', alpha=1)
    plt.plot(mpv_test, fop_test, "s-", label='Test до калибровки', alpha=1)
    plt.plot(mpv_oot, fop_oot, "s-", label='Oot до калибровки', alpha=1)
    plt.tight_layout()
    plt.grid(True, alpha=0.65)
    plt.xlabel('PD', fontsize=10)
    plt.ylabel('Bad Rate', fontsize=10)
    plt.legend(loc='upper left')
    plt.title('График PD до калибровки', fontsize=10)
    plt.savefig(our_outputs+'/'+'graph_before_calibration', bbox_inches ='tight', pad_inches = 0.1)
    plt.show()

    # График калибровочных кривых после калибровки
    fraction_of_positives_c, mean_predicted_value_c = calibration_curve(y_true, y_pred_calib, normalize=False, n_bins=10)
    fop_train_c, mpv_train_c = calibration_curve(y_true_train, y_pred_train_calib, normalize=False, n_bins=10)
    fop_test_c, mpv_test_c = calibration_curve(y_true_test, y_pred_test_calib, normalize=False, n_bins=10)
    fop_oot_c, mpv_oot_c = calibration_curve(y_true_oot, y_pred_oot_calib, normalize=False, n_bins=10)

    plt.figure(figsize=[12,8])
    plt.plot([0, 1], [0, 1], "k:", label="Идеальная калибровка")
    plt.plot(mean_predicted_value_c, fraction_of_positives_c, "s-", label='Train+Test+OOT калиброванный', alpha=0.5)
    plt.plot(mpv_train_c, fop_train_c, "s-", label='Train калиброванный', alpha=1)
    plt.plot(mpv_test_c, fop_test_c, "s-", label='Test калиброванный', alpha=1)
    plt.plot(mpv_oot_c, fop_oot_c, "s-", label='Oot калиброванный', alpha=1)
    plt.tight_layout()
    plt.grid(True, alpha=0.65)
    plt.xlabel('PD', fontsize=10)
    plt.ylabel('Bad Rate', fontsize=10)
    plt.legend(loc='upper left')
    plt.title('График PD после калибровки', fontsize=10)
    plt.savefig(our_outputs+'/'+'graph_after_calibration', bbox_inches ='tight', pad_inches = 0.1)
    plt.show()

    # График распределения PD до калибровки
    x_hst = y_pred
    print("Всего скоров: ", len(x_hst))
    plt.figure(figsize=[15,10])
    n, bins, patches = plt.hist(x=x_hst, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
    plt.grid(True, alpha=0.75)
    plt.xlabel('PD', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xlim(0,1)
    plt.title('Распределение PD до калибровки', fontsize=10)
    plt.savefig(our_outputs+'/'+'hist_PD_before_calibration', bbox_inches ='tight', pad_inches = 0.1)
    plt.show()

    # График распределения PD после калибровки
    x_hst = y_pred_calib
    print("Всего скоров: ", len(x_hst))
    plt.figure(figsize=[15,10])
    n, bins, patches = plt.hist(x=x_hst, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
    plt.grid(True, alpha=0.75)
    plt.xlabel('PD', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xlim(0,1)
    plt.title('Распределение калиброванного PD', fontsize=10)
    plt.savefig(our_outputs+'/'+'hist_PD_after_calibration', bbox_inches ='tight', pad_inches = 0.1)
    plt.show()

    # График распределения скора до калибровки
    x_hst = y_pred_score
    print("Всего скоров: ", len(x_hst))
    plt.figure(figsize=[15,10])
    n, bins, patches = plt.hist(x=x_hst, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
    plt.grid(True, alpha=0.75)
    plt.xlabel('SCORE', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xlim(0,1000)
    plt.title('Распределение SCORE до калибровки', fontsize=10)
    plt.savefig(our_outputs+'/'+'hist_SCORE_before_calibration', bbox_inches ='tight', pad_inches = 0.1)
    plt.show()

    # График распределения скора после калибровки
    x_hst = y_pred_calib_score
    print("Всего скоров: ", len(x_hst))
    plt.figure(figsize=[15,10])
    n, bins, patches = plt.hist(x=x_hst, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
    plt.grid(True, alpha=0.75)
    plt.xlabel('SCORE', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xlim(0,1000)
    plt.title('Распределение SCORE после калибровки', fontsize=10)
    plt.savefig(our_outputs+'/'+'hist_SCORE_after_calibration', bbox_inches ='tight', pad_inches = 0.1)
    plt.show()

    def mean_in_bins(bins, scores):
        '''Среднее расстояние между бинами'''
        lst_delt = []
        for i in range(len(bins)):
            if i < len(bins)-1:
                if (bins[i]>scores.min()) & (bins[i]<scores.max()):
                    if (bins[i+1]<scores.max()):
                        delt = bins[i+1] - bins[i]
                        lst_delt.append(delt)
        mean = np.mean(lst_delt)
        return mean

    # Получаем границы неоткалиброванных скоров по n сплитам
    y_pred_score_rnd = y_pred_score.round(0) # округляем калиброванные скоры, для расчета Грейдов
    
    perc_lst = [round((i+1)*100/n_split, 4) for i in range(n_split)]
    perc_lst.append(0)
    perc_lst = sorted(perc_lst)
    
    
    bins_grade_lst = []
    # lst_bins = grade_dict[seg_name].copy() # - грейды, уже подготовленные ПМ
    lst_bins = list(np.percentile(y_pred_score_rnd, perc_lst)) # Грейды по 6 сплитам

    def bins_grade(lst_bins):
        ''' Функция, преобразовывает cut points для np.histogram() '''
        mvl = 0.00000000001
        bins_grade = [lst_bins[i]+mvl for i in range(len(lst_bins))]
        return bins_grade
    bins_grade_lst = bins_grade(lst_bins)

    #bins_grade_lst.append(y_pred_score.min())
    #bins_grade_lst.append(y_pred_score.max()+0.00000000001)
    #bins_grade_lst = sorted(bins_grade_lst)

    print("\n Грейды неоткалиброванных скоров: ")
    hist, bins = np.histogram(y_pred_score, bins=bins_grade_lst)
    print("Cut Points грейдов: ", bins)
    mean_bin = mean_in_bins(bins, y_pred_score)
    print("\n Среднее расстояние между бинами ", mean_bin)
    print("\n")
    
    if inverse == True:
        if grade_lst == None:
            grade_lst = sorted(list(range(n_split)), reverse = True)
    else: 
        if grade_lst == None:
            grade_lst = sorted(list(range(n_split))) 
            
    if len(hist)!=len(grade_lst):
        print(len(hist), len(grade_lst))
        raise ValueError('len(hist) != len(grade_lst)')
    
    print(hist, grade_lst)
    
    for cnt, grade in zip(hist, grade_lst):
        print(str(grade) + ' : ' + str(cnt) + ' (' + str(round(cnt/hist.sum()*100,2)) + '%)')

    # Получаем границы калиброванных скоров по n сплитам
    y_pred_calib_score_rnd = y_pred_calib_score.round(0) # округляем калиброванные скоры, для расчета Грейдов
    perc_lst = [round((i+1)*100/n_split, 4) for i in range(n_split)]
    perc_lst.append(0)
    perc_lst = sorted(perc_lst)

    bins_grade_lst = []
    # lst_bins = grade_dict[seg_name].copy() # - грейды, уже подготовленные ПМ
    lst_bins = list(np.percentile(y_pred_calib_score_rnd, perc_lst)) 
    bins_grade_lst = bins_grade(lst_bins)

    #bins_grade_lst.append(y_pred_calib_score.min())
    #bins_grade_lst.append(y_pred_calib_score.max()+0.00000000001)
    #bins_grade_lst = sorted(bins_grade_lst)

    y_pred_calib_score_ = pd.DataFrame(y_pred_calib_score, columns=['calib_score'])
    y_pred_calib_ = pd.DataFrame(y_pred_calib, columns=['pd'])
    # y_true_ = pd.DataFrame(y_true.asarray(), columns=['badflag'])
    y_pred_calib_score_ = pd.concat([y_pred_calib_score_, y_pred_calib_, y_true], axis=1)

    print("\n Грейды откалиброванных скоров: ")
    hist, bins = np.histogram(y_pred_calib_score_['calib_score'], bins=bins_grade_lst)
    print("Cut Points грейдов: ", bins)
    mean_bin = mean_in_bins(bins, y_pred_calib_score_['calib_score'])
    print("\n Среднее расстояние между бинами ", mean_bin)
    print("\n")
    
    grade_lst_inverse = [grade_lst[len(grade_lst)-1-i] for i in range(len(grade_lst))]
    
    if inverse == True:
        for i, grade in zip(range(len(bins)), grade_lst_inverse):
            if i+2<len(bins):
                print(bins[i], bins[i+1], grade)
                if i == 0:
                    
                    y_pred_calib_score_.loc[y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001, 
                                            'grade'] = grade
                    
                    y_pred_calib_score_.loc[y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001, 
                     'grade_edges'] = '(-inf; '+str(bins[i+1]-0.00000000001)+')'
                    
                    pd_edge = round(score_calculation_to_pd(bins[i+1]-0.00000000001, base_points, 
                                                      score_points_to_change, odds_change, inverse), 4)
                    
                    y_pred_calib_score_.loc[y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001, 
                     'grade_edges_pd'] = '('+str(pd_edge)+'; 1)'
                    
                else:
                    y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001) & 
                                        (y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001), 'grade'] = grade
                    
                    y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001) & 
                                            (y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001), 
                     'grade_edges'] = '['+str(bins[i]-0.00000000001)+'; '+str(bins[i+1]-0.00000000001)+')'
                    
                    pd_edge = round(score_calculation_to_pd(bins[i]-0.00000000001, base_points, 
                                                      score_points_to_change, odds_change, inverse), 4)
                    pd_edge1 = round(score_calculation_to_pd(bins[i+1]-0.00000000001, base_points, 
                                                      score_points_to_change, odds_change, inverse), 4)
                    
                    y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001) & 
                                            (y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001), 
                     'grade_edges_pd'] = '('+str(pd_edge1)+'; '+str(pd_edge)+']'
                    
            elif i+1 == len(bins)-1:
                
                print(bins[i], bins[i+1], grade)
                y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001), 'grade'] = grade
                
                y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001), 
                     'grade_edges'] = '[' +str(bins[i]-0.00000000001) + '; +inf)'
                
                pd_edge = round(score_calculation_to_pd(bins[i]-0.00000000001, base_points, 
                                                      score_points_to_change, odds_change, inverse), 4)
                
                y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001), 
                     'grade_edges_pd'] = '(0; '+str(pd_edge)+']'
                    
    else: 
        for i, grade in zip(range(len(bins)), grade_lst):
            if i+2<len(bins):
                print(bins[i], bins[i+1], grade)
                if i == 0:
                    y_pred_calib_score_.loc[y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001, 
                                            'grade'] = grade
                    
                    y_pred_calib_score_.loc[y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001, 
                     'grade_edges'] = '(-inf; '+str(bins[i+1]-0.00000000001)+')'
                    
                    pd_edge = round(score_calculation_to_pd(bins[i+1]-0.00000000001, base_points, 
                                                      score_points_to_change, odds_change, inverse), 4)
                    
                    y_pred_calib_score_.loc[y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001, 
                     'grade_edges_pd'] = '(0; ' + str(pd_edge) + ')'
                    
                else:
                    y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001) & 
                                        (y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001), 
                                            'grade'] = grade
                    
                    y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001) & 
                                            (y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001), 
                     'grade_edges'] = '['+str(bins[i]-0.00000000001) +'; '+str(bins[i+1]-0.00000000001)+')'
                    
                    pd_edge = round(score_calculation_to_pd(bins[i]-0.00000000001, base_points, 
                                                      score_points_to_change, odds_change, inverse), 4)
                    pd_edge1 = round(score_calculation_to_pd(bins[i+1]-0.00000000001, base_points, 
                                                      score_points_to_change, odds_change, inverse), 4)
                    
                    y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001) & 
                                            (y_pred_calib_score_['calib_score'] < bins[i+1]-0.00000000001), 
                     'grade_edges_pd'] = '[' + str(pd_edge)+ '; ' + str(pd_edge1) + ')'
                
            elif i+1 == len(bins)-1:
                print(bins[i], bins[i+1], grade)
                y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001), 'grade'] = grade
                y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001), 
                     'grade_edges'] = '[' +str(bins[i]-0.00000000001) + '; +inf)'
                
                pd_edge = round(score_calculation_to_pd(bins[i]-0.00000000001, base_points, 
                                                      score_points_to_change, odds_change, inverse), 4)
                
                y_pred_calib_score_.loc[(y_pred_calib_score_['calib_score'] >= bins[i]-0.00000000001), 
                     'grade_edges_pd'] = '[' + str(pd_edge) + '; 1)'
    
    
    pivot_calib_score = pd.pivot_table(y_pred_calib_score_, ['pd', 'calib_score', target, 'grade_edges', 
                                                             'grade_edges_pd'], index='grade',
                                       aggfunc={'pd':['min', 'max', 'mean'], 'calib_score':['min', 'max', 'mean'], 
                                                target:['mean'], 'grade_edges': 'max'})
    
    grades_ = y_pred_calib_score_[['grade', 'grade_edges_pd']].drop_duplicates()
    print(grades_)
    
    pivot_calib_score.loc[(pivot_calib_score['pd']['min']<=pivot_calib_score[target]['mean']) & 
                          (pivot_calib_score[target]['mean']<=pivot_calib_score['pd']['max']), 
                          'fl_in_grade'] = 1 # флаг того, что badrate в грейде лежит между PD грейда
    pivot_calib_score = pivot_calib_score.reset_index()
    
    to_replace= {}
    for i, v in enumerate(grade_lst):
        to_replace[v] = i
    
    pivot_calib_score['sort_col'] = pivot_calib_score['grade'].replace(to_replace, inplace = True)
    pivot_calib_score = pivot_calib_score.sort_values(by='sort_col')
    pivot_calib_score = pivot_calib_score.set_index('grade')
    pivot_calib_score = pivot_calib_score.drop(['sort_col'], axis=1)
    print(pivot_calib_score)
    columns_piv = pivot_calib_score.columns.to_list()
    
    pivot_calib_score = pd.merge(pivot_calib_score, grades_, on = 'grade')
            
    if len(hist)!=len(grade_lst):
        print(len(hist), len(grade_lst))
        raise ValueError('len(hist) != len(grade_lst)')
    for cnt, grade in zip(hist, grade_lst):
        print(str(grade) + ' : ' + str(cnt) + ' (' + str(round(cnt/hist.sum()*100,2)) + '%)')

    from tabulate import tabulate
    print(pivot_calib_score)
    print(tabulate(pivot_calib_score, headers='keys', tablefmt= 'psql'))
        
    return a, b, mpv_med, pivot_calib_score
# In[ ]:
def score_calculation(y_pred, base_points = 700, score_points_to_change = 50, 
                          odds_change = 2, inverse = True):
    '''
    y_pred - вектор с оценками PD
    base_points - базовый скор (по умолчанию и на период 14.01.2020)
    score_points_to_change - сколько нужно скор баллов для изменения (удвоения/утроения и пр) шансов. По умолчанию 50.
    odds_change - изменение шансов (2 - удвоение, 3 - утроение и тд). По умолчанию 2
    inverse - ведет ли больший скор к меньшему PD? по умолчанию True
    '''
    if inverse == True:
        y_pred_score = 700+ 50*(np.log((1/y_pred - 1)/10)/np.log(2))
    else:
        y_pred_score = 700+ 50*(np.log((y_pred/(1-y_pred))/10)/np.log(2))
    return y_pred_score
    
def score_calculation_to_pd(y_score, base_points = 700, score_points_to_change = 50, 
                               odds_change = 2, inverse = True):
    '''
    y_score - вектор с оценками score
    base_points - базовый скор (по умолчанию и на период 14.01.2020)
    score_points_to_change - сколько нужно скор баллов для изменения (удвоения/утроения и пр) шансов. По умолчанию 50.
    odds_change - изменение шансов (2 - удвоение, 3 - утроение и тд). По умолчанию 2
    inverse - ведет ли больший скор к меньшему PD? по умолчанию True
    '''
    if inverse == True:
        y_pd = 1/(10*2**((y_score-700)/50)+1)    
    else:
        y_pd = 1-1/(10*2**((y_score-700)/50)+1)   
    return y_pd

# In[ ]:

def lr_test_one_simple(model, data, y, feature_restr, params, class_weight = None, 
                           features = None, feature_empty = False, 
                       intercept = False):
    
    """
    model - обученная модель
    data - данные
    y - таргет
    feature_restr - набор переменных с ограничениями
    params - параметры обучения модели. Можно получить из обученной модели с помощью model.get_params()
    class_weight - class_weight
    features - список переменных "длинной" модели
    features_empty - если True, то проверяется значимость всех переменных одновременно
    intercept - если True, то проверяется значимость свободного члена
    
    """
        
    def _check_sample_weight(sample_weight, X, dtype=None):
        """Validate sample weights.
        Note that passing sample_weight=None will output an array of ones.
        Therefore, in some cases, you may want to protect the call with:            
        if sample_weight is not None:
        sample_weight = _check_sample_weight(...)
        Parameters
        ----------
        sample_weight : {ndarray, Number or None}, shape (n_samples,)
           Input sample weights.
           X : nd-array, list or sparse matrix
        Input data.
        dtype: dtype
        dtype of the validated `sample_weight`.
           If None, and the input `sample_weight` is an array, the dtype of the
           input is preserved; otherwise an array with the default numpy dtype
           is be allocated.  If `dtype` is not one of `float32`, `float64`,
           `None`, the output will be of dtype `float64`.
        Returns
        -------
        sample_weight : ndarray, shape (n_samples,)
            Validated sample weight. It is guaranteed to be "C" contiguous.
        """
        from sklearn.utils import check_array
            
        n_samples = _num_samples(X)

        if dtype is not None and dtype not in [np.float32, np.float64]:
            dtype = np.float64
        if sample_weight is None or isinstance(sample_weight, numbers.Number): # Исправлено
            if sample_weight is None: 
                sample_weight = np.ones(n_samples, dtype=dtype)
            else:
                sample_weight = np.full(n_samples, sample_weight,
                                        dtype=dtype)
        else:
            if dtype is None:
                dtype = [np.float64, np.float32]
            sample_weight = check_array(
                    sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
                    order="C") # Исправлено
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight.shape == {}, expected {}!"
                                     .format(sample_weight.shape, (n_samples,)))
        return sample_weight
        
    def _num_samples(x):
        """Return number of samples in array-like x."""
        import numbers
        message = 'Expected sequence or array-like, got %s' % type(x)
        if hasattr(x, 'fit') and callable(x.fit):
            # Don't get num_samples from an ensembles length!
            raise TypeError(message)

        if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
            if hasattr(x, '__array__'):
                x = np.asarray(x)
            else:
                raise TypeError(message)

        if hasattr(x, 'shape') and x.shape is not None:
            if len(x.shape) == 0:
                raise TypeError("Singleton array %r cannot be considered"
                                " a valid collection." % x)
            # Check that shape is returning an integer or default to len
            # Dask dataframes may not return numeric shape[0] value
            if isinstance(x.shape[0], numbers.Integral):
                return x.shape[0]
        
    from scipy.stats import chi2
    if features is None:
        features = list(data.columns)
            
    model_alt = model(**params).fit(data[features], y)

    if feature_empty == True:

        features_empty = pd.DataFrame(np.ones(data.shape[0])).rename(columns = {0: 'ones'}) # Исправлено
        null_pred = sum(y)/len(y)*np.ones(len(y))
        df = len(features)

    elif intercept == True:

        model2 = model(**params)
        model2 = model2.set_params(**{'fit_intercept': False})
        model1 = model2.fit(data[features], y)
        null_pred = model1.predict_proba(data[features])
        df = 1

    else:
        if len(feature_restr) < 1:
            raise ValueError('At least one feature is needed for H0!')
        model2 = model(**params)
        model1 = model2.fit(data[feature_restr], y)
        null_pred = model1.predict_proba(data[feature_restr])
        df = len(features) - len(feature_restr)
        
    if class_weight is not None:
        sample_weight = None
        sample_weight = _check_sample_weight(sample_weight, data)
        classes = np.unique(y)
        le = LabelEncoder()
        if (y.nunique() > 2):
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]
        # for compute_class_weight

        class_weight_ = compute_class_weight(class_weight, classes, y)
        sample_weight *= class_weight_[le.fit_transform(y)]
    else:
        sample_weight = None
            
        
    alt_pred = model_alt.predict_proba(data[features])
    null_log_likelihood = -log_loss(y, null_pred, sample_weight = sample_weight, normalize = False)
    alternative_log_likelihood = -log_loss(y, alt_pred, sample_weight = sample_weight, normalize = False)
    G = 2*(alternative_log_likelihood - null_log_likelihood)
    p_value = chi2.sf(G, df)
    
    return p_value

def lr_test_all_features(model, model_method, data, y, class_weight = None, whole_model = True, one_feature_naive=False):
    
    """
    model - обученная модель
    model_method - метод обучения линейной модели LogisticRegression, например
    data - данные
    y - таргет
    class_weight - class_weight. Если не подавать, то каждому наблюдению присваивается вес 1
    whole_model - если True, то дополнительно будет проверена значимость всей модели 
    
    """
    
    def lr_test_one_simple(model, data, y, feature_restr, params, class_weight = None, 
                           features = None, feature_empty = False, 
                       intercept = False):
        
        """
        model - метод обучения линейной модели LogisticRegression, например
        data - данные
        y - таргет
        feature_restr - набор переменных с ограничениями
        params - параметры обучения модели. 
        class_weight - class_weight
        features - список переменных "длинной" модели
        features_empty - если True, то проверяется значимость всех переменных одновременно
        intercept - если True, то проверяется значимость свободного члена

        """
        
        def _check_sample_weight(sample_weight, X, dtype=None):
            """Validate sample weights.
            Note that passing sample_weight=None will output an array of ones.
            Therefore, in some cases, you may want to protect the call with:
            if sample_weight is not None:
                sample_weight = _check_sample_weight(...)
            Parameters
            ----------
            sample_weight : {ndarray, Number or None}, shape (n_samples,)
               Input sample weights.
            X : nd-array, list or sparse matrix
                Input data.
            dtype: dtype
               dtype of the validated `sample_weight`.
               If None, and the input `sample_weight` is an array, the dtype of the
               input is preserved; otherwise an array with the default numpy dtype
               is be allocated.  If `dtype` is not one of `float32`, `float64`,
               `None`, the output will be of dtype `float64`.
            Returns
            -------
            sample_weight : ndarray, shape (n_samples,)
               Validated sample weight. It is guaranteed to be "C" contiguous.
            """
            import numbers

            from sklearn.utils import check_array
            
            n_samples = _num_samples(X)

            if dtype is not None and dtype not in [np.float32, np.float64]:
                dtype = np.float64

            if sample_weight is None or isinstance(sample_weight, numbers.Number): # Исправлено
                if sample_weight is None:
                    sample_weight = np.ones(n_samples, dtype=dtype)
                else:
                    sample_weight = np.full(n_samples, sample_weight,
                                            dtype=dtype)
            else:
                if dtype is None:
                    dtype = [np.float64, np.float32]
                sample_weight = check_array(
                    sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
                    order="C"
                )           # Исправлено
                if sample_weight.ndim != 1:
                    raise ValueError("Sample weights must be 1D array or scalar")

                if sample_weight.shape != (n_samples,):
                    raise ValueError("sample_weight.shape == {}, expected {}!"
                                     .format(sample_weight.shape, (n_samples,)))
            return sample_weight
        
        def _num_samples(x):
            """Return number of samples in array-like x."""
            import numbers
            message = 'Expected sequence or array-like, got %s' % type(x)
            if hasattr(x, 'fit') and callable(x.fit):
                # Don't get num_samples from an ensembles length!
                raise TypeError(message)

            if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
                if hasattr(x, '__array__'):
                    x = np.asarray(x)
                else:
                    raise TypeError(message)

            if hasattr(x, 'shape') and x.shape is not None:
                if len(x.shape) == 0:
                    raise TypeError("Singleton array %r cannot be considered"
                                    " a valid collection." % x)
                # Check that shape is returning an integer or default to len
                # Dask dataframes may not return numeric shape[0] value
                if isinstance(x.shape[0], numbers.Integral):
                    return x.shape[0]
        
        from scipy.stats import chi2
        if features is None:
            features = list(data.columns)
            
        model_alt = model(**params).fit(data[features], y)

        if feature_empty == True:

            features_empty = pd.DataFrame(np.ones(data.shape[0])).rename(columns = {0: 'ones'})

            null_pred = sum(y)/len(y)*np.ones(len(y))

            df = len(features)

        elif intercept == True:

            model2 = model(**params)
            model2 = model2.set_params(**{'fit_intercept': False})
            model1 = model2.fit(data[features], y)
            null_pred = model1.predict_proba(data[features])
            df = 1

        else:
            if len(feature_restr) < 1:
                raise ValueError('At least one feature is needed for H0!')
            model2 = model(**params)
            model1 = model2.fit(data[feature_restr], y)
            null_pred = model1.predict_proba(data[feature_restr])

            df = len(features) - len(feature_restr)

        alt_pred = model_alt.predict_proba(data[features])
        
        if class_weight is not None:
            sample_weight = None
            sample_weight = _check_sample_weight(sample_weight, data)
            classes = np.unique(y)
            le = LabelEncoder()
            if (y.nunique() > 2):
                raise ValueError('To fit OvR, use the pos_class argument')
            # np.unique(y) gives labels in sorted order.
            pos_class = classes[1]
            # for compute_class_weight

            class_weight_ = compute_class_weight(class_weight, classes,
                                                     y)
            sample_weight *= class_weight_[le.fit_transform(y)]
        else:
            sample_weight = None
            
        
        null_log_likelihood = -log_loss(y, null_pred, sample_weight = sample_weight, normalize = False)

        alternative_log_likelihood = -log_loss(y, alt_pred, sample_weight = sample_weight, normalize = False)

        G = 2*(alternative_log_likelihood - null_log_likelihood)

        p_value = chi2.sf(G, df)

        return p_value
    
    betas = []
    if(one_feature_naive==False):
        k = 0
        for i, v in enumerate(data.columns):
            if k % 5 ==0:
                tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
                print ('Number of finished repetitions:', k , '| time: ' , tm)
            columns_restr = list(data.columns)
            columns_restr.remove(v)
            betas.append([v, model.coef_[0][i], lr_test_one_simple(model_method,
                                                                   data, y, columns_restr, model.get_params(),
                                                                  class_weight = class_weight)])
            k = k+1
    else:
        columns_restr=None
#         whole_model=True

    betas.append(['Intercept', model.intercept_[0], 
                  lr_test_one_simple(model_method, data, y, columns_restr, model.get_params(), 
                                     class_weight = class_weight, 
                                     intercept = True)])

    if (whole_model == True or one_feature_naive==True):
        
        betas.append(['Whole model', 0, 
                  lr_test_one_simple(model_method, data, y, columns_restr, model.get_params(), 
                                     class_weight = class_weight, 
                                     feature_empty = True)])
        
    betas = pd.DataFrame.from_records(betas, columns = ['Variable', 'Coef', 'p_value'])
    
    return betas


# In[ ]:
def coef_test_bootstrap(X_1, y_1, X_2,  y_2, model, pass_model, 
              nreps, out, preprocessing_hyps, confidence = 0.95, params_to_fit = None,
                        preprocessing = True, 
                       task = 'binary'):
    
    """
    X_1, y_1, X_2, y_2 - данные
    model - обученная модель 
    pass_model - метод обучения (LogisticRegression или RandomForestClassifier)
    nreps - количество повторений
    preprocessing_hyps - параметры для препроцессинга
    confidence - уровень доверия для построения доверительного интервала
    params_to_fit - параметры для обучения (можно поставить None - тогда будут использованы параметры model)
    preprocessing - индикатор использования препроцессинга
    task - 'binary'
    
    """
    
    
    from scipy.stats import sem, t, ttest_1samp
    
    if params_to_fit is None:
        params_to_fit = model.get_params()
    
    train_scores1, test_scores1, betas_box1= bootstrap(X_1, y_1, X_2, y_2, pass_model, 
              nreps, out, preprocessing_hyps, params_to_fit, None, None,
              None, None, None, preprocessing, 
                       task, False,
                      None, None)
    ttest = []
    
    for i, v in enumerate(list(betas_box1.columns)):
        if v != 'Intercept':
            a = 1.0 * np.array(betas_box1[v])
            n = len(np.array(betas_box1[v]))
            m, se = np.mean(np.array(betas_box1[v])), sem(np.array(betas_box1[v]))
            h = se * t.ppf((1 + confidence) / 2., n-1)
            left_border = m-h
            right_border = m+h
            alpha=confidence
            p_lower=((1.0-alpha)/2.0)*100
            p_upper=((1.0+alpha)/2.0)*100
            left_border_percentile = np.percentile(betas_box1[v], p_lower)
            right_border_percentile = np.percentile(betas_box1[v], p_upper)
            mean_gini = np.mean(np.array(train_scores1['GINI']))
            left_border_gini = np.percentile(train_scores1['GINI'], p_lower)
            right_border_gini = np.percentile(train_scores1['GINI'], p_upper)
            ttest.append([v, model.coef_[0][i], round(ttest_1samp(np.array(betas_box1[v]), 0).pvalue, 5), m, left_border, 
                         right_border, left_border_percentile, right_border_percentile, mean_gini, left_border_gini, right_border_gini])
        else:
            a = 1.0 * np.array(betas_box1[v])
            n = len(np.array(betas_box1[v]))
            m, se = np.mean(np.array(betas_box1[v])), sem(np.array(betas_box1[v]))
            h = se * t.ppf((1 + confidence) / 2., n-1)
            left_border = m-h
            right_border = m+h
            alpha=confidence
            p_lower=((1.0-alpha)/2.0)*100
            p_upper=((1.0+alpha)/2.0)*100
            left_border_percentile = np.percentile(betas_box1[v], p_lower)
            right_border_percentile = np.percentile(betas_box1[v], p_upper)
            mean_gini = np.mean(np.array(train_scores1['GINI']))
            left_border_gini = np.percentile(train_scores1['GINI'], p_lower)
            right_border_gini = np.percentile(train_scores1['GINI'], p_upper)
            ttest.append([v, model.intercept_[0], round(ttest_1samp(np.array(betas_box1[v]), 0).pvalue, 5), m, left_border, 
                         right_border, left_border_percentile, right_border_percentile, mean_gini, left_border_gini, right_border_gini])
    
    ttest1 = pd.DataFrame.from_records(ttest, columns = ['Variable', 'coef', 'p_value', 'mean_betas', 'left_border_conf',
                                                        'right_border_conf', 'left_border_perc', 'right_border_perc', 'mean_gini', 
                                                        'left_border_gini', 'right_border_gini'])

    return ttest1, betas_box1, train_scores1, test_scores1

# In[ ]:

def coef_test_bootstrap_results(model, betas_box1, train_scores1, confidence):
    
    """
    model - обученная модель 
    betas_box1 - файл с векторами оценок для всех коэффициентов из бустрэпа
    confidence - уровень доверия для построения доверительного интервала
    
    """
    
    
    from scipy.stats import sem, t, ttest_1samp
    
    ttest = []
    
    for i, v in enumerate(list(betas_box1.columns)):
        if v != 'Intercept':
            a = 1.0 * np.array(betas_box1[v])
            n = len(np.array(betas_box1[v]))
            m, se = np.mean(np.array(betas_box1[v])), sem(np.array(betas_box1[v]))
            h = se * t.ppf((1 + confidence) / 2., n-1)
            left_border = m-h
            right_border = m+h
            alpha=confidence
            p_lower=((1.0-alpha)/2.0)*100
            p_upper=((1.0+alpha)/2.0)*100
            left_border_percentile = np.percentile(betas_box1[v], p_lower)
            right_border_percentile = np.percentile(betas_box1[v], p_upper)
            mean_gini = np.mean(np.array(train_scores1['GINI']))
            left_border_gini = np.percentile(train_scores1['GINI'], p_lower) # Исправлено
            right_border_gini = np.percentile(train_scores1['GINI'], p_upper)
            ttest.append([v, model.coef_[0][i], round(ttest_1samp(np.array(betas_box1[v]), 0).pvalue, 5), m, left_border, 
                         right_border, left_border_percentile, right_border_percentile, mean_gini, left_border_gini, right_border_gini])
        else:
            a = 1.0 * np.array(betas_box1[v])
            n = len(np.array(betas_box1[v]))
            m, se = np.mean(np.array(betas_box1[v])), sem(np.array(betas_box1[v]))
            h = se * t.ppf((1 + confidence) / 2., n-1)
            left_border = m-h
            right_border = m+h
            alpha=confidence
            p_lower=((1.0-alpha)/2.0)*100
            p_upper=((1.0+alpha)/2.0)*100
            left_border_percentile = np.percentile(betas_box1[v], p_lower)
            right_border_percentile = np.percentile(betas_box1[v], p_upper)
            mean_gini = np.mean(np.array(train_scores1['GINI']))
            left_border_gini = np.percentile(train_scores1['GINI'], p_lower)
            right_border_gini = np.percentile(train_scores1['GINI'], p_upper)
            ttest.append([v, model.intercept_[0], round(ttest_1samp(np.array(betas_box1[v]), 0).pvalue, 5), m, left_border, 
                         right_border, left_border_percentile, right_border_percentile, mean_gini, left_border_gini, right_border_gini])
    
    ttest1 = pd.DataFrame.from_records(ttest, columns = ['Variable', 'coef', 'p_value', 'mean_betas', 'left_border_conf',
                                                        'right_border_conf', 'left_border_perc', 'right_border_perc', 'mean_gini', 
                                                        'left_border_gini', 'right_border_gini'])

    return ttest1
# In[ ]:
def gini_feature_calc_final(X_1_2, y_1_2, X_3_2, y_3, model, pass_model, 
              params_to_fit = None, task = None):
    
    """
    Игорь, выставляю твою функцию
    
    X_1, y_1, X_2, y_2 - данные
    model - обученная модель 
    pass_model - метод обучения (LogisticRegression или DecisionTreeClassifier)
    params_to_fit - параметры для обучения (можно поставить None - тогда будут использованы параметры model)
    task - 'Tree'
    
    """
    
    
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    
    if task is None:
        task = model.get_params()
    
    #Расчет Gini по каждой переменной
    feature_gini = []
    
    for var in X_1_2.columns:
        clf = LogisticRegression(**task)
        clf.fit(X_1_2[var].values.reshape(-1, 1), y_1_2) # обучили модель
        y_pred_train = clf.predict_proba(X_1_2[var].values.reshape(-1, 1))[:, 1]
        y_pred_oot = clf.predict_proba(X_3_2[var].values.reshape(-1, 1))[:, 1]
        fpr_train, tpr_train, thresholds = metrics.roc_curve(y_1_2, y_pred_train)
        fpr_oot, tpr_oot, _ = metrics.roc_curve(y_3, y_pred_oot)
        logit_gini_train = 2 * metrics.auc(fpr_train,tpr_train) - 1
        logit_gini_oot = 2 * metrics.auc(fpr_oot,tpr_oot) - 1
        tree=DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=2, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=241)
        tree.fit(X_1_2[var].values.reshape(-1, 1), y_1_2)
        y_pred_train_tree = tree.predict_proba(X_1_2[var].values.reshape(-1, 1))[:, 1]
        y_pred_oot_tree = tree.predict_proba(X_3_2[var].values.reshape(-1, 1))[:, 1]
        fpr_train_tree, tpr_train_tree, thresholds_tree = metrics.roc_curve(y_1_2, y_pred_train_tree)
        fpr_oot_tree, tpr_oot_tree, _ = metrics.roc_curve(y_3, y_pred_oot_tree)
        tree_gini_train = 2 * metrics.auc(fpr_train_tree,tpr_train_tree) - 1
        tree_gini_oot = 2 * metrics.auc(fpr_oot_tree,tpr_oot_tree) - 1
        feature_gini.append([var, logit_gini_train, logit_gini_oot, tree_gini_train, tree_gini_oot])
        
    feature_gini1 = pd.DataFrame.from_records(feature_gini, columns = ['variable', 'logit_gini_train', 'logit_gini_oot', 'tree_gini_train', 'tree_gini_oot'])        

    return feature_gini1
# In[ ]:
def calculate_metrics_for_one_var(train, y_train, test, y_test, column, model, model_params, 
                                  use_metrics = [metrics.roc_auc_score, metrics.average_precision_score], 
                                  names = ['roc_auc_score', 'average_precision_score'],     
                                  integral_metrics = [metrics.roc_auc_score, metrics.average_precision_score],
                                  task = 'binary'):     
    
    """
    
    train, y_train, test, y_test - данные
    column - колонка 
    model - метод обучения (LogisticRegression или DecisionTreeClassifier)
    model_params - параметры для обучения (можно поставить None - тогда будут использованы параметры model)
    use_metrics - какие метрики используются. Метрика Gini считается автоматически, если есть метрика roc_auc_score
    names - названия для метрик
    
    """
    
    def varname(p):
        import inspect
        import re
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\)', line)
            if m:
                return m.group(1)


    from sklearn import metrics
    def gini_score(y_true, y_preds):
        from sklearn import metrics

        return metrics.roc_auc_score(y_true, y_preds)*2-1
    
    integral_metrics.append(gini_score)
    
    nums = re.compile('[0-9.]')

    model_to_fit = model(**model_params)
    model_fitted = model_to_fit.fit(pd.DataFrame(train[column]), y_train)
        
    metrics_result = []

    if hasattr(model_fitted, 'coef_'):
        columns_metrics = ['Feature', 'metric', 'score', 'coef']
    else:
        columns_metrics = ['Feature', 'metric', 'score']

    if metrics.roc_auc_score in use_metrics:
        use_metrics.append(gini_score)
        names.append('Gini_score')


    if task == 'binary':
        for metric, name in zip(use_metrics, names):
            y_train_pred = model_fitted.predict_proba(pd.DataFrame(train[column]))[:, 1]
            y_test_pred = model_fitted.predict_proba(pd.DataFrame(test[column]))[:, 1]

            if metric in integral_metrics:
                if hasattr(model_fitted, 'coef_'):
                    to_append = [column, name, metric(y_test, y_test_pred), model_fitted.coef_[0][0]]
                else:                    
                    to_append = [column, name, metric(y_test, y_test_pred)]
            else:
                train_thresholds = np.unique(y_train_pred)

                train_metrics = []
                for thre in train_thresholds:
                    y_train_predictions = np.where(y_train_pred < thre, 0, 1)

                    string = name
                    subs = re.sub('N', '', re.sub('N.N', 'N', re.sub('NN', 'N', re.sub(nums, 'N', string))))
                    if subs == 'f_score' and name != 'f_score' and metric != metrics.f1_score:
                        train_metrics.append(metrics.fbeta_score(y_train, y_train_predictions, 
                                                beta = eval(''.join(re.findall(nums, string)))))
                    else:
                        train_metrics.append(metric(y_train, y_train_predictions))

                train_metrics = np.array(train_metrics)
                threshold = train_thresholds[np.argmax(train_metrics)]
                y_test_predictions = np.where(y_test_pred < threshold, 0, 1)

                string = name
                subs = re.sub('N', '', re.sub('N.N', 'N', re.sub('NN', 'N', re.sub(nums, 'N', string))))
                if subs == 'f_score' and name != 'f_score' and metric != metrics.f1_score:
                    if hasattr(model_fitted, 'coef_'):
                        to_append = [column, name, metrics.fbeta_score(y_test, y_test_predictions, 
                                            beta = eval(''.join(re.findall(nums, string)))), 
                                        model_fitted.coef_[0][0]]
                    else:                    
                        to_append = [column, name, metrics.fbeta_score(y_test, y_test_predictions,   # Исправлено
                                                   beta = eval(''.join(re.findall(nums, string))))]

                else:
                    if hasattr(model_fitted, 'coef_'):
                        to_append = [column, name, metric(y_test, y_test_predictions), model_fitted.coef_[0][0]]
                    else:                    
                        to_append = [column, name, metric(y_test, y_test_predictions)]

            metrics_result.append(to_append)

    elif task == 'numeric':
        for metric, name in zip(use_metrics, names):
            y_train_pred = model_fitted.predict(pd.DataFrame(train[column]))
            y_test_pred = model_fitted.predict(pd.DataFrame(test[column]))

            if hasattr(model_fitted, 'coef_'):
                print(model_fitted.coef_)
                to_append = [column, name, metric(y_test, y_test_pred), model_fitted.coef_]
            else:
                to_append = [column, name, metric(y_test, y_test_pred)]

            metrics_result.append(to_append)

    data_final = pd.DataFrame.from_records(metrics_result, columns = columns_metrics)

    return data_final
# In[ ]:
def calculate_metrics_for_several_vars(train, y_train, test, y_test, columns, model, model_params, 
                                  use_metrics = [metrics.roc_auc_score, metrics.average_precision_score], 
                                  names = ['roc_auc_score', 'average_precision_score'], task = 'binary', 
                                  integral_metrics = [metrics.roc_auc_score, metrics.average_precision_score],
                                  n_jobs = 3):
    
    """
    
    train, y_train, test, y_test - данные
    column - колонка 
    model - метод обучения (LogisticRegression или DecisionTreeClassifier)
    model_params - параметры для обучения (можно поставить None - тогда будут использованы параметры model)
    use_metrics - какие метрики используются. Метрика Gini считается автоматически, если есть метрика roc_auc_score
    names - названия для метрик
    task - бинарная или нумерическая
    n_jobs - на сколько бить потоков для распараллеливания
    
    """
    
    def calculate_metrics_for_one_var(train, y_train, test, y_test, column, model, model_params, 
                                  use_metrics = [metrics.roc_auc_score, metrics.average_precision_score], 
                                  names = ['roc_auc_score', 'average_precision_score'], task = 'binary',
                                  integral_metrics = [metrics.roc_auc_score, metrics.average_precision_score]):     
        def varname(p):
            import inspect
            import re
            for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
                m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\)', line)
                if m:
                    return m.group(1)


        from sklearn import metrics
        def gini_score(y_true, y_preds):
            from sklearn import metrics

            return metrics.roc_auc_score(y_true, y_preds)*2-1

        from sklearn import metrics
        integral_metrics.append(gini_score)
        nums = re.compile('[0-9.]')

        model_to_fit = model(**model_params)
        model_fitted = model_to_fit.fit(pd.DataFrame(train[column]), y_train)

        metrics_result = []

        if hasattr(model_fitted, 'coef_'):
            columns_metrics = ['Feature', 'metric', 'score', 'coef']
        else:
            columns_metrics = ['Feature', 'metric', 'score']

        if metrics.roc_auc_score in use_metrics:
            use_metrics.append(gini_score)
            names.append('Gini_score')


        if task == 'binary':
            for metric, name in zip(use_metrics, names):
                y_train_pred = model_fitted.predict_proba(pd.DataFrame(train[column]))[:, 1]
                y_test_pred = model_fitted.predict_proba(pd.DataFrame(test[column]))[:, 1]

                if metric in integral_metrics:
                    if hasattr(model_fitted, 'coef_'):
                        to_append = [column, name, metric(y_test, y_test_pred), model_fitted.coef_[0][0]]
                    else:                    
                        to_append = [column, name, metric(y_test, y_test_pred)]
                else:
                    train_thresholds = np.unique(y_train_pred)

                    train_metrics = []
                    for thre in train_thresholds:
                        y_train_predictions = np.where(y_train_pred < thre, 0, 1)

                        string = name
                        subs = re.sub('N', '', re.sub('N.N', 'N', re.sub('NN', 'N', re.sub(nums, 'N', string))))
                        if subs == 'f_score' and name != 'f_score' and metric != metrics.f1_score:
                            train_metrics.append(metrics.fbeta_score(y_train, y_train_predictions, 
                                                   beta = eval(''.join(re.findall(nums, string)))))
                        else:
                            train_metrics.append(metric(y_train, y_train_predictions))

                    train_metrics = np.array(train_metrics)
                    threshold = train_thresholds[np.argmax(train_metrics)]
                    y_test_predictions = np.where(y_test_pred < threshold, 0, 1)

                    string = name
                    subs = re.sub('N', '', re.sub('N.N', 'N', re.sub('NN', 'N', re.sub(nums, 'N', string))))
                    if subs == 'f_score' and name != 'f_score' and metric != metrics.f1_score:
                        if hasattr(model_fitted, 'coef_'):
                            to_append = [column, name, metrics.fbeta_score(y_test, y_test_predictions, 
                                                   beta = eval(''.join(re.findall(nums, string)))), 
                                         model_fitted.coef_[0][0]]
                        else:                    
                            to_append = [column, name, metrics.fbeta_score(y_test, y_test_predictions, 
                                                   beta = eval(''.join(re.findall(nums, string))))]

                    else:
                        if hasattr(model_fitted, 'coef_'):
                            to_append = [column, name, metric(y_test, y_test_predictions), model_fitted.coef_[0][0]]
                        else:                    
                            to_append = [column, name, metric(y_test, y_test_predictions)]

                metrics_result.append(to_append)

        elif task == 'numeric':
            for metric, name in zip(use_metrics, names):
                y_train_pred = model_fitted.predict(pd.DataFrame(train[column]))
                y_test_pred = model_fitted.predict(pd.DataFrame(test[column]))

                if hasattr(model_fitted, 'coef_'):
                    print(model_fitted.coef_)
                    to_append = [column, name, metric(y_test, y_test_pred), model_fitted.coef_]
                else:
                    to_append = [column, name, metric(y_test, y_test_pred)]

                metrics_result.append(to_append)


        data_final = pd.DataFrame.from_records(metrics_result, columns = columns_metrics)

        return data_final
    
    from joblib import Parallel, delayed
    
    parallel = Parallel(n_jobs=n_jobs)
    
    with parallel:
        par_res = parallel((delayed(calculate_metrics_for_one_var)(train, y_train, test, y_test, column,
                                                                   model, model_params, 
                                                  use_metrics, names, task, integral_metrics) for column in columns))
    
    fin_df = pd.concat(par_res)
    
    return fin_df

# In[ ]:

def adjusted_r2(y_true, y_pred, k):
    R2 = metrics.r2_score(y_true, y_pred)
    n = len(y_true)
    return 1-(1-R2)*(n-1)/(n-k-1)

def by_month_r2(time_period, X, y, k, prediction_name, target = None):
    
    """
    Бьет выборку по месяцам и считает помесячные значение Gini.
    model - модель
    time_period - поле, в котором находятся временные периоды
    X, y - данные (полностью подготовленные) и таргет
    good_bad_dict - словарь вида {'good': 1, 'bad': 0}, нужен для определения, на основании чего считать good_rate и bad_rate статистики
    
    """
    
    def adjusted_r2(y_true, y_pred, k):
        R2 = metrics.r2_score(y_true, y_pred)
        n = len(y_true)
        return 1-(1-R2)*(n-1)/(n-k-1)
    
    X.reset_index(inplace = True)
    X.drop('index', axis = 1, inplace = True)
    if target == None:
        target = y.name
    y_new = y.reset_index()[target]
    
    time_periods = sorted(X[time_period].unique())
    
    scores = []
       
    
    for i in time_periods:
        X_month = X.loc[X[time_period] == i].copy()
        X_index = X_month.index
        y_month = y.iloc[X_index]
        y_mean = y_month.mean()
        y_std = y_month.std()
        
        
        prediction = X_month[prediction_name].copy()        
        pred_mean = prediction.mean()
        pred_std = prediction.std()
        
        if len(y_month)>0:
            r2 = metrics.r2_score(y_month, prediction)
            r2_adj = adjusted_r2(y_month, prediction, k)
            mae = metrics.mean_absolute_error(y_month, prediction)
        else:
            r2 = np.nan
            r2_adj = np.nan
            mae = np.nan
        
        scores.append([i, len(y_month), y_mean, y_std, pred_mean, pred_std, r2, r2_adj, mae])
    
    col_names = ['month_call', 'number', 'y_mean', 'y_std', 'pred_mean', 'pred_std', 'r2', 'r2_adj', 'mae']
    scores = pd.DataFrame.from_records(scores, columns = col_names)
    return scores

# In[ ]:

def lr_test_all_features_numeric(model, model_method, data, y, sample_weight = None, whole_model = True, one_feature_naive=False):
    
    """
    model - обученная модель
    model_method - метод обучения линейной модели LogisticRegression, например
    data - данные
    y - таргет
    class_weight - class_weight. Если не подавать, то каждому наблюдению присваивается вес 1
    whole_model - если True, то дополнительно будет проверена значимость всей модели 
    
    """
    
    def lr_test_one_simple_numeric(model, data, y, feature_restr, params, sample_weight = None,
                           features = None, feature_empty = False, 
                       intercept = False):
    
        """
        model - обученная модель
        data - данные
        y - таргет
        feature_restr - набор переменных с ограничениями
        params - параметры обучения модели. Можно получить из обученной модели с помощью model.get_params()
        class_weight - class_weight
        features - список переменных "длинной" модели
        features_empty - если True, то проверяется значимость всех переменных одновременно
        intercept - если True, то проверяется значимость свободного члена

        """

        from scipy.stats import f
        from sklearn.metrics import mean_squared_error

        if features is None:
            features = list(data.columns)

        model_alt = model(**params).fit(data[features], y)

        if feature_empty == True:

            features_empty = pd.DataFrame(np.ones(data.shape[0])).rename(columns = {0: 'ones'})
            null_pred = sum(y)/len(y)*np.ones(len(y))
            df = len(features)

        elif intercept == True:

            model2 = model(**params)
            model2 = model2.set_params(**{'fit_intercept': False})
            model1 = model2.fit(data[features], y)
            null_pred = model1.predict(data[features])
            df = 1

        else:
            if len(feature_restr) < 1:
                raise ValueError('At least one feature is needed for H0!')
            model2 = model(**params)
            model1 = model2.fit(pd.DataFrame(data[feature_restr]), y)
            null_pred = model1.predict(pd.DataFrame(data[feature_restr]))
            df = len(features) - len(feature_restr)
            
        n = data.shape[0]
        m = data.shape[1]
        alt_pred = model_alt.predict(data[features])
        null_log_likelihood = mean_squared_error(y, null_pred, sample_weight = sample_weight)
        alternative_log_likelihood = mean_squared_error(y, alt_pred, sample_weight = sample_weight)
        F = (null_log_likelihood - alternative_log_likelihood)*(n-m)/(alternative_log_likelihood*df)
        p_value = f.sf(F, df, n-m)

        return p_value

    
    betas = []
    if (one_feature_naive==False):
        k = 0
        for i, v in enumerate(data.columns):
            if k % 5 ==0:
                tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
                print ('Number of finished repetitions:', k , '| time: ' , tm)
            columns_restr = list(data.columns)
            columns_restr.remove(v)
            betas.append([v, model.coef_[i], lr_test_one_simple_numeric(model_method,
                                                                   data, y, columns_restr, model.get_params(),
                                                                  sample_weight = sample_weight)])
            k = k+1
    else:
        columns_restr=None
        
    betas.append(['Intercept', model.intercept_, 
                  lr_test_one_simple_numeric(model_method, data, y, columns_restr, model.get_params(), 
                                     sample_weight = sample_weight, 
                                     intercept = True)])

    if (whole_model == True or one_feature_naive==True):
        betas.append(['Whole model', 0, 
                  lr_test_one_simple_numeric(model_method, data, y, columns_restr, model.get_params(), 
                                     sample_weight = sample_weight, 
                                     feature_empty = True)])
        
    betas = pd.DataFrame.from_records(betas, columns = ['Variable', 'Coef', 'p_value'])
    
    return betas
# In[ ]:
def linear_calib_numeric_all_piece(all_pr_calib, time_period, prediction_name, target, k, 
                                   high_threshold = None,
                                   low_threshold = None, log = True, once_in = 3):
    
    """
    all_pr_calib - данные
    time_period - название столбца с месяцами
    prediction_name - название столбца в прогнозами
    target - название столбца с таргетом
    log - логарифмировать или нет
    once_in - пересчитывать калибровку раз в N месяцев
    
    """
    def adjusted_r2(y_true, y_pred, k):
        R2 = metrics.r2_score(y_true, y_pred)
        n = len(y_true)
        return 1-(1-R2)*(n-1)/(n-k-1)
    
    from sklearn.linear_model import LinearRegression
    unique_months = sorted(all_pr_calib[time_period].unique())
    
    splits = np.arange(0, len(unique_months), once_in)
    
    distinct_periods = []

    for i, v in enumerate(splits):
        if i <= len(splits)-2:
            period = unique_months[splits[i]:splits[i+1]]
        elif i == len(splits)-1:
            period = unique_months[splits[i]:]
        distinct_periods.append(period)    
        
    period_results = []

    for i, period in enumerate(distinct_periods):
        if i <= len(distinct_periods)-2:

            train_calib_month = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                                                     prediction_name])
            y_train_calib_month = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), target]
            
            test_calib_month = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), 
                                                                    prediction_name])
            y_test_calib_month = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), target]
            
            if log == True:
                train_calib_month = pd.DataFrame(np.log(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                                                     prediction_name]))
                test_calib_month = pd.DataFrame(np.log(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), 
                                                                    prediction_name]))
                y_log_train_calib_month = np.log(y_train_calib_month)
                y_log_test_calib_month = np.log(y_test_calib_month)

            lr_month = LinearRegression()
            if log == True:
                lin_reg_month = lr_month.fit(train_calib_month, y_log_train_calib_month)
            else:
                lin_reg_month = lr_month.fit(train_calib_month, y_train_calib_month)

            y_train_predict_month = lin_reg_month.predict(train_calib_month)
            y_test_predict_month = lin_reg_month.predict(test_calib_month)

            if log == True:
                y_train_money_month = np.exp(y_train_predict_month)
                y_test_money_month = np.exp(y_test_predict_month)

                if high_threshold != None:
                    y_train_money_month = np.where(y_train_money_month <= high_threshold, 
                                                   y_train_money_month, high_threshold)
                    y_test_money_month = np.where(y_test_money_month <= high_threshold, 
                                                  y_test_money_month, high_threshold)

                if low_threshold != None:
                    y_train_money_month = np.where(y_train_money_month >= low_threshold, 
                                                   y_train_money_month, low_threshold)

                    y_test_money_month = np.where(y_test_money_month >= low_threshold, y_test_money_month, 
                                                  low_threshold)
                    
            else:
                if high_threshold != None:
                    y_train_predict_month = np.where(y_train_predict_month <= high_threshold, 
                                                   y_train_predict_month, high_threshold)
                    y_test_predict_month = np.where(y_test_predict_month <= high_threshold, 
                                                  y_test_predict_month, high_threshold)

                if low_threshold != None:
                    y_train_predict_month = np.where(y_train_predict_month >= low_threshold, 
                                                   y_train_predict_month, low_threshold)

                    y_test_predict_month = np.where(y_test_predict_month >= low_threshold, y_test_predict_month, 
                                                  low_threshold)

            
            if log == False:
                r2_train_money = metrics.r2_score(y_train_calib_month, y_train_predict_month)
                r2_test_money = metrics.r2_score(y_test_calib_month, y_test_predict_month)
                
                r2_train_money_adj = adjusted_r2(y_train_calib_month, y_train_predict_month, k)
                r2_test_money_adj = adjusted_r2(y_test_calib_month, y_test_predict_month, k)

                mae_train_money = metrics.mean_absolute_error(y_train_calib_month, y_train_predict_month)
                mae_test_money = metrics.mean_absolute_error(y_test_calib_month, y_test_predict_month)
                
                period_results.append([distinct_periods[i], distinct_periods[i+1], r2_train, r2_test, 
                                       mae_train_money, mae_test_money, lin_reg_month.intercept_,
                                      lin_reg_month.coef_])
            
            if log == True:
                r2_train = metrics.r2_score(y_log_train_calib_month, y_train_predict_month)
                r2_test = metrics.r2_score(y_log_test_calib_month, y_test_predict_month)
                
                r2_train_adj = adjusted_r2(y_log_train_calib_month, y_train_predict_month, k)
                r2_test_adj = adjusted_r2(y_log_test_calib_month, y_test_predict_month, k)

                r2_train_money = metrics.r2_score(y_train_calib_month, y_train_money_month)
                r2_test_money = metrics.r2_score(y_test_calib_month, y_test_money_month)
                
                r2_train_money_adj = adjusted_r2(y_train_calib_month, y_train_money_month, k)
                r2_test_money_adj = adjusted_r2(y_test_calib_month, y_test_money_month, k)

                mae_train_money = metrics.mean_absolute_error(y_train_calib_month, y_train_money_month)
                mae_test_money = metrics.mean_absolute_error(y_test_calib_month, y_test_money_month)

                period_results.append([distinct_periods[i], distinct_periods[i+1], 
                                       r2_train, r2_test, 
                                       r2_train_adj, r2_test_adj,
                                       r2_train_money, r2_test_money, 
                                       r2_train_money_adj, r2_test_money_adj,
                                       mae_train_money, mae_test_money, lin_reg_month.intercept_,
                                      lin_reg_month.coef_])

    if log == True:
        per_res = pd.DataFrame(period_results, columns= ['train_period', 'test_period', 
                                                         'r2_log_train', 'r2_log_test', 
                                                         'r2_log_train_adj', 'r2_log_test_adj', 
                                                         'r2_train_money', 'r2_test_money', 
                                                         'r2_train_money_adj', 'r2_test_money_adj', 
                                                         'mae_train_money', 'mae_test_money', 'intercept',
                                                        'coefs']) 
    if log == False:
        per_res = pd.DataFrame(period_results, columns= ['train_period', 'test_period', 
                                                         'r2_train', 'r2_test', 
                                                         'r2_train_adj', 'r2_test_adj', 
                                                         'mae_train', 'mae_test', 'intercept',
                                                        'coefs']) 
    
    return per_res

# In[ ]:
def linear_calib_numeric_all_piece_binned(all_pr_calib, time_period, prediction_name, target, k, ct, n_bins = 100, 
                                          strategy = 'quantile', mpv_strategy='median', normalize=False,
                                   high_threshold = None,
                                   low_threshold = None, once_in = 3, by_tar = False, plt_show = False):
    
    """
    all_pr_calib - данные
    time_period - название столбца с месяцами
    prediction_name - название столбца в прогнозами
    target - название столбца с таргетом
    log - логарифмировать или нет
    once_in - пересчитывать калибровку раз в N месяцев
    
    """
    from sklearn.linear_model import LinearRegression
    
    def adjusted_r2(y_true, y_pred, k):
        R2 = metrics.r2_score(y_true, y_pred)
        n = len(y_true)
        return 1-(1-R2)*(n-1)/(n-k-1)
    
    unique_months = sorted(all_pr_calib[time_period].unique())
    
    splits = np.arange(0, len(unique_months), once_in)
    
    distinct_periods = []

    for i, v in enumerate(splits):
        if i <= len(splits)-2:
            period = unique_months[splits[i]:splits[i+1]]
        elif i == len(splits)-1:
            period = unique_months[splits[i]:]
        distinct_periods.append(period)    
        
    period_results = []

    for i, period in enumerate(distinct_periods):
        if i <= len(distinct_periods)-2:
            
            train_calib_month_all = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                                                     [target, prediction_name]]
            test_calib_month_all = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), 
                                                                    [target, prediction_name]])

            train_calib_month = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                                                     prediction_name])
            y_train_calib_month = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), target]
            
            
            
            test_calib_month = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), 
                                                                    prediction_name])
            y_test_calib_month = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), target]
            
            if ct == None:
                ct1 = y_train_calib_month.mean()
            else:
                ct1 = ct
                    
            a, b, mpv_med, lin_reg_month = linear_calibration_numeric(train_calib_month_all, target, prediction_name, ct1, 
                                                    n_bins = n_bins, strategy = strategy, mpv_strategy=mpv_strategy, 
                                                    normalize=normalize, by_tar = by_tar, plt_show = plt_show)

            y_train_predict_month = lin_reg_month.predict(train_calib_month)
            y_test_predict_month = lin_reg_month.predict(test_calib_month)
            
            all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                          'train_preds_calib'] = y_train_predict_month
            all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), 
                                          'test_preds_calib'] = y_test_predict_month

            if high_threshold != None:
                y_train_predict_month = np.where(y_train_predict_month <= high_threshold, y_train_predict_month, 
                                                 high_threshold)
                y_test_predict_month = np.where(y_test_predict_month <= high_threshold, y_test_predict_month, high_threshold)

            if low_threshold != None:
                y_train_predict_month = np.where(y_train_predict_month >= low_threshold, y_train_predict_month, low_threshold)
                y_test_predict_month = np.where(y_test_predict_month >= low_threshold, y_test_predict_month, low_threshold)

            r2_train = metrics.r2_score(y_train_calib_month, y_train_predict_month)
            r2_test = metrics.r2_score(y_test_calib_month, y_test_predict_month)

            r2_train_adj = adjusted_r2(y_train_calib_month, y_train_predict_month, k)
            r2_test_adj = adjusted_r2(y_test_calib_month, y_test_predict_month, k)

            
            mae_train_money = metrics.mean_absolute_error(y_train_calib_month, y_train_predict_month)
            mae_test_money = metrics.mean_absolute_error(y_test_calib_month, y_test_predict_month)
                
            period_results.append([distinct_periods[i], distinct_periods[i+1], 
                                   r2_train, r2_test, 
                                   r2_train_adj, r2_test_adj,
                                       mae_train_money, mae_test_money, a,
                                      b])

    per_res = pd.DataFrame(period_results, columns= ['train_period', 'test_period', 
                                                     'r2_train', 'r2_test', 
                                                     'r2_train_adj', 'r2_test_adj',
                                                     'mae_train', 'mae_test', 
                                                     'intercept', 'coefs']) 
    
    return per_res, all_pr_calib


def linear_calib_numeric_test_binned(all_pr_calib, test_pr_calib, 
                                               time_period, prediction_name, target, k, ct, n_bins = 100, 
                                          strategy = 'quantile', mpv_strategy='median', normalize=False,
                                   high_threshold = None,
                                   low_threshold = None, once_in = 3, by_tar = False, plt_show = False):
    
    """
    all_pr_calib - данные
    time_period - название столбца с месяцами
    prediction_name - название столбца в прогнозами
    target - название столбца с таргетом
    log - логарифмировать или нет
    once_in - пересчитывать калибровку раз в N месяцев
    
    """
    from sklearn.linear_model import LinearRegression
    
    def adjusted_r2(y_true, y_pred, k):
        R2 = metrics.r2_score(y_true, y_pred)
        n = len(y_true)
        return 1-(1-R2)*(n-1)/(n-k-1)
    
    unique_months = sorted(all_pr_calib[time_period].unique())
    unique_months_test = sorted(test_pr_calib[time_period].unique())
    
    splits = np.arange(0, len(unique_months), once_in)
    
    distinct_periods = []

    for i, v in enumerate(splits):
        if i <= len(splits)-2:
            period = unique_months[splits[i]:splits[i+1]]
        elif i == len(splits)-1:
            period = unique_months[splits[i]:]
        distinct_periods.append(period)    
        
    distinct_periods_test = []

    for period_num, period in enumerate(distinct_periods):
        if period_num <= len(distinct_periods)-2:
            period_test = []
            for per_num, per in enumerate(unique_months_test):
                if per > max(distinct_periods[period_num]) and per <= max(distinct_periods[period_num+1]):
                    period_test.append(per)

        elif period_num == len(distinct_periods)-1:
            period_test = []
            for per_num, per in enumerate(unique_months_test):
                if per > max(distinct_periods[period_num]):
                    period_test.append(per)
        distinct_periods_test.append(period_test)    
        
    period_results = []

    for i, period in enumerate(distinct_periods):
        if i <= len(distinct_periods)-2:
            
            train_calib_month_all = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                                                     [target, prediction_name]]
            test_calib_month_all = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), 
                                                                    [target, prediction_name]])
            

            train_calib_month = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                                                     prediction_name])
            y_train_calib_month = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), target]
            
            
            
            test_calib_month = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), 
                                                                    prediction_name])
            y_test_calib_month = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), target]
            
        
            if ct == None:
                ct1 = y_train_calib_month.mean()
            else:
                ct1 = ct
                    
            a, b, mpv_med, lin_reg_month = linear_calibration_numeric(train_calib_month_all, target, prediction_name, ct1, 
                                                    n_bins = n_bins, strategy = strategy, mpv_strategy=mpv_strategy, 
                                                    normalize=normalize, by_tar = by_tar, plt_show = plt_show)
                

            
            y_train_predict_month = lin_reg_month.predict(train_calib_month)
            y_test_predict_month = lin_reg_month.predict(test_calib_month)

            all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                              'train_preds_calib'] = y_train_predict_month
            all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i+1]), 
                                              'test_preds_calib'] = y_test_predict_month

            if high_threshold != None:
                y_train_predict_month = np.where(y_train_predict_month <= high_threshold, y_train_predict_month, 
                                                 high_threshold)
                y_test_predict_month = np.where(y_test_predict_month <= high_threshold, y_test_predict_month, high_threshold)

            if low_threshold != None:
                y_train_predict_month = np.where(y_train_predict_month >= low_threshold, y_train_predict_month, 
                                                 low_threshold)
                y_test_predict_month = np.where(y_test_predict_month >= low_threshold, y_test_predict_month, low_threshold)


            r2_train = metrics.r2_score(y_train_calib_month, y_train_predict_month)
            r2_test = metrics.r2_score(y_test_calib_month, y_test_predict_month)
            
            r2_train_adj = adjusted_r2(y_train_calib_month, y_train_predict_month, k)
            r2_test_adj = adjusted_r2(y_test_calib_month, y_test_predict_month, k)

            mae_train_money = metrics.mean_absolute_error(y_train_calib_month, y_train_predict_month)
            mae_test_money = metrics.mean_absolute_error(y_test_calib_month, y_test_predict_month)

            if len(distinct_periods_test[i]) == 0:
                period_results.append([distinct_periods[i], distinct_periods[i+1], np.nan, 
                                       r2_train, r2_test, np.nan, 
                                       mae_train_money, mae_test_money, np.nan,
                                       a, b])
            else:
                test_calib_sec_month_all = test_pr_calib.loc[test_pr_calib[time_period].isin(distinct_periods_test[i]), 
                                                                     [target, prediction_name]]
                test_calib_sec_month = pd.DataFrame(test_pr_calib.loc[test_pr_calib[time_period].isin(distinct_periods_test[i]), 
                                                                     prediction_name])
                y_test_calib_sec_month = test_pr_calib.loc[test_pr_calib[time_period].isin(distinct_periods_test[i]), target]
                
                y_test_predict_sec_month = lin_reg_month.predict(test_calib_sec_month)
                
                test_pr_calib.loc[test_pr_calib[time_period].isin(distinct_periods_test[i]), 
                                              'train_preds_calib'] = y_test_predict_sec_month
                
                if high_threshold != None:
                    y_test_predict_sec_month = np.where(y_test_predict_sec_month <= high_threshold, y_test_predict_sec_month, 
                                              high_threshold)

                if low_threshold != None:
                    y_test_predict_sec_month = np.where(y_test_predict_sec_month >= low_threshold, y_test_predict_sec_month, 
                                               low_threshold)
                    
                r2_test_sec = metrics.r2_score(y_test_calib_sec_month, y_test_predict_sec_month)
                r2_test_sec_adj = adjusted_r2(y_test_calib_sec_month, y_test_predict_sec_month, k)
                mae_test_sec_money = metrics.mean_absolute_error(y_test_calib_sec_month, y_test_predict_sec_month)
                
                period_results.append([distinct_periods[i], distinct_periods[i+1], distinct_periods_test[i], 
                                       r2_train, r2_test, r2_test_sec, 
                                       r2_train_adj, r2_test_adj, r2_test_sec_adj,
                                       mae_train_money, mae_test_money, mae_test_sec_money,
                                       a, b])
                
    if len(distinct_periods_test[len(distinct_periods)-1]) > 0:
        i = len(distinct_periods)-1
        train_calib_month_all = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                                                     [target, prediction_name]]
            

        train_calib_month = pd.DataFrame(all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                                                     prediction_name])
        y_train_calib_month = all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), target]
            
        
        if ct == None:
            ct1 = y_train_calib_month.mean()
        else:
            ct1 = ct
                    
        a, b, mpv_med, lin_reg_month = linear_calibration_numeric(train_calib_month_all, target, prediction_name, ct1, 
                                                    n_bins = n_bins, strategy = strategy, mpv_strategy=mpv_strategy, 
                                                    normalize=normalize, by_tar = by_tar, plt_show = plt_show)
        
        test_calib_sec_month_all = test_pr_calib.loc[test_pr_calib[time_period].isin(distinct_periods_test[i]), 
                                                                     [target, prediction_name]]
        test_calib_sec_month = pd.DataFrame(test_pr_calib.loc[test_pr_calib[time_period].isin(distinct_periods_test[i]), 
                                                                     prediction_name])
        y_test_calib_sec_month = test_pr_calib.loc[test_pr_calib[time_period].isin(distinct_periods_test[i]), target]
                
        y_train_predict_month = lin_reg_month.predict(train_calib_month)
        y_test_predict_sec_month = lin_reg_month.predict(test_calib_sec_month)
                
        all_pr_calib.loc[all_pr_calib[time_period].isin(distinct_periods[i]), 
                                              'train_preds_calib'] = y_train_predict_month
        test_pr_calib.loc[test_pr_calib[time_period].isin(distinct_periods_test[i]), 
                                              'train_preds_calib'] = y_test_predict_sec_month
                
        if high_threshold != None:
            y_train_predict_month = np.where(y_train_predict_month <= high_threshold, y_train_predict_month, 
                                                 high_threshold)
            y_test_predict_sec_month = np.where(y_test_predict_sec_month <= high_threshold, y_test_predict_sec_month, 
                                              high_threshold)

        if low_threshold != None:
            y_train_predict_month = np.where(y_train_predict_month >= low_threshold, y_train_predict_month, 
                                                 low_threshold)
            y_test_predict_sec_month = np.where(y_test_predict_sec_month >= low_threshold, y_test_predict_sec_month, 
                                               low_threshold)
                    
        r2_train = metrics.r2_score(y_train_calib_month, y_train_predict_month)
        r2_train_adj = adjusted_r2(y_train_calib_month, y_train_predict_month, k)
        mae_train_money = metrics.mean_absolute_error(y_train_calib_month, y_train_predict_month)
        
        r2_test_sec = metrics.r2_score(y_test_calib_sec_month, y_test_predict_sec_month)
        r2_test_sec_adj = adjusted_r2(y_test_calib_sec_month, y_test_predict_sec_month, k)
        mae_test_sec_money = metrics.mean_absolute_error(y_test_calib_sec_month, y_test_predict_sec_month)
                
        period_results.append([distinct_periods[i], np.nan, distinct_periods_test[i], 
                                       r2_train, np.nan, r2_test_sec, 
                                       r2_train_adj, np.nan, r2_test_sec_adj, 
                                       mae_train_money, np.nan, mae_test_sec_money,
                                       a, b])
    
                
    per_res = pd.DataFrame(period_results, columns= ['train_period', 'train_next_period', 'test_period', 
                                                     'r2_train', 'r2_train_next_period', 'r2_test',
                                                     'r2_train_adj', 'r2_train_next_period_adj', 'r2_test_adj',
                                                     'mae_train', 'mae_train_next_period', 'mae_test',
                                                     'intercept', 'coefs']) 
    
    return per_res, all_pr_calib, test_pr_calib


def calibration_curve_numeric(calib_data, target, prediction_name, normalize=False, n_bins=5,
                          strategy='uniform', by_tar = False):
    """Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier.
    Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
         True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whethey_prob needs to be normalized into the bin [0, 1], i.e. is not
                a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data. Bins with no data
        points (i.e. without corresponding values in y_prob) will not be
        returned, thus there may be fewer than n_bins in the return value.
    strategy : {'uniform', 'quantile'}, (default='uniform')
        Strategy used to define the widths of the bins.
        uniform
            All bins have identical widths.
            quantile
            All bins have the same number of points.
    Returns
    -------
    prob_true : array, shape (n_bins,) or smaller
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins,) or smaller
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    from sklearn.linear_model import LinearRegression
    
    y_true = calib_data[target]
    y_prob = calib_data[prediction_name]
            
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    if normalize:  # MinMaxScaler
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())


    if by_tar == False:
        if strategy == 'quantile':  # Determine bin edges by distribution of data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.percentile(y_prob, quantiles * 100)
            bins[-1] = bins[-1] + 1e-8
        elif strategy == 'uniform':
            bins = np.linspace(0., max(y_prob), n_bins + 1)
        else:
            raise ValueError("Invalid entry to 'strategy' input. Strategy "
                                             "must be either 'quantile' or 'uniform'.")

        for i, grade in enumerate(bins):
            if i+2<len(bins):
                if i == 0:
                    calib_data.loc[calib_data[prediction_name] < bins[i+1],'grade'] = i

                else:
                    calib_data.loc[(calib_data[prediction_name] >= bins[i]) & 
                                                   (calib_data[prediction_name] < bins[i+1]), 'grade'] = i

            elif i+1 == len(bins)-1:
                calib_data.loc[(calib_data[prediction_name] >= bins[i]), 'grade'] = i
                        
    else:
        if strategy == 'quantile':  # Determine bin edges by distribution of data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.percentile(y_true, quantiles * 100)
            bins[-1] = bins[-1] + 1e-8
        elif strategy == 'uniform':
            bins = np.linspace(0., max(y_true), n_bins + 1)
        else:
            raise ValueError("Invalid entry to 'strategy' input. Strategy "
                                             "must be either 'quantile' or 'uniform'.")

        for i, grade in enumerate(bins):
            if i+2<len(bins):
                if i == 0:
                    calib_data.loc[calib_data[target] < bins[i+1],'grade'] = i

                else:
                    calib_data.loc[(calib_data[target] >= bins[i]) & 
                                                   (calib_data[target] < bins[i+1]), 'grade'] = i

            elif i+1 == len(bins)-1:
                calib_data.loc[(calib_data[target] >= bins[i]), 'grade'] = i

    groupped = calib_data.groupby('grade')[[target, prediction_name]].mean().reset_index()
            
    trues = groupped[target]
    preds = groupped[prediction_name]

    return trues, preds, groupped


def linear_calibration_numeric(calib_data, target, prediction_name, ct, n_bins, strategy, mpv_strategy, normalize = False,
                               by_tar = False, plt_show=False):
  
        """
        Функция для калибровки с помощью линейной регрессии логарифма шансов. 
        Выводит коэффициенты a и b, медианное значение pd/скорингового балла (на выборке, на которой калибруем модель), 
        а также рисует график odds-pd и выводит обученную модель.

        Методика - G:/New Risk Management/Decision Science/Knowledge Base/Calibration/FW Шкалирование и калибровка.msg
        Пример использования - G:/New Risk Management/Decision Science/kgrushin/PD Models/Calibration/Калибровка BL19_BEELINE v2

        y_true - истинные метки выборки, на которую калибруем модель
        y_pred - предсказанные калибруемой моделью PD на той же выборке
        ct - значение центральной тенденции. Например DR на выборке oot.

        n_bins - количество бинов, на которое разбиваем выборку, на которую калибруем модель
        strategy - принимает значение 'uniform' и 'quantile'. 
                   uniform - бьем на бины равной ширины, quantile - бьем на бины с равным количеством наблюдений
       mpv_strategy - поправка PD. 'median' or 'average'
        plt_show - если True - рисуем график отношения шансов к PD (Score), по умолчанию = False

        """
        def calibration_curve_numeric(calib_data, target, prediction_name, normalize=False, n_bins=5,
                          strategy='uniform', by_tar = False):
            """Compute true and predicted probabilities for a calibration curve.
            The method assumes the inputs come from a binary classifier.
            Calibration curves may also be referred to as reliability diagrams.
            Read more in the :ref:`User Guide <calibration>`.
            Parameters
            ----------
            y_true : array, shape (n_samples,)
                 True targets.
            y_prob : array, shape (n_samples,)
                Probabilities of the positive class.
            normalize : bool, optional, default=False
                Whethey_prob needs to be normalized into the bin [0, 1], i.e. is not
                        a proper probability. If True, the smallest value in y_prob is mapped
                onto 0 and the largest one onto 1.
            n_bins : int
                Number of bins. A bigger number requires more data. Bins with no data
                points (i.e. without corresponding values in y_prob) will not be
                returned, thus there may be fewer than n_bins in the return value.
            strategy : {'uniform', 'quantile'}, (default='uniform')
                Strategy used to define the widths of the bins.
                uniform
                    All bins have identical widths.
                    quantile
                    All bins have the same number of points.
            Returns
            -------
            prob_true : array, shape (n_bins,) or smaller
                The true probability in each bin (fraction of positives).
            prob_pred : array, shape (n_bins,) or smaller
                The mean predicted probability in each bin.
            References
            ----------
            Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
            Probabilities With Supervised Learning, in Proceedings of the 22nd
            International Conference on Machine Learning (ICML).
            See section 4 (Qualitative Analysis of Predictions).
            """
            
            from sklearn.linear_model import LinearRegression
            y_true = calib_data[target]
            y_prob = calib_data[prediction_name]

            y_true = column_or_1d(y_true)
            y_prob = column_or_1d(y_prob)
            check_consistent_length(y_true, y_prob)

            if normalize:  # MinMaxScaler
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

            if by_tar == False:
                if strategy == 'quantile':  # Determine bin edges by distribution of data
                    quantiles = np.linspace(0, 1, n_bins + 1)
                    bins = np.percentile(y_prob, quantiles * 100)
                    bins[-1] = bins[-1] + 1e-8
                elif strategy == 'uniform':
                    bins = np.linspace(0., max(y_prob), n_bins + 1)
                else:
                    raise ValueError("Invalid entry to 'strategy' input. Strategy "
                                             "must be either 'quantile' or 'uniform'.")

                for i, grade in enumerate(bins):
                    if i+2<len(bins):
                        if i == 0:
                            calib_data.loc[calib_data[prediction_name] < bins[i+1],'grade'] = i

                        else:
                            calib_data.loc[(calib_data[prediction_name] >= bins[i]) & 
                                                   (calib_data[prediction_name] < bins[i+1]), 'grade'] = i

                    elif i+1 == len(bins)-1:
                        calib_data.loc[(calib_data[prediction_name] >= bins[i]), 'grade'] = i
                        
            else:
                if strategy == 'quantile':  # Determine bin edges by distribution of data
                    quantiles = np.linspace(0, 1, n_bins + 1)
                    bins = np.percentile(y_true, quantiles * 100)
                    bins[-1] = bins[-1] + 1e-8
                elif strategy == 'uniform':
                    bins = np.linspace(0., max(y_true), n_bins + 1)
                else:
                    raise ValueError("Invalid entry to 'strategy' input. Strategy "
                                             "must be either 'quantile' or 'uniform'.")

                for i, grade in enumerate(bins):
                    if i+2<len(bins):
                        if i == 0:
                            calib_data.loc[calib_data[target] < bins[i+1],'grade'] = i

                        else:
                            calib_data.loc[(calib_data[target] >= bins[i]) & 
                                                   (calib_data[target] < bins[i+1]), 'grade'] = i

                    elif i+1 == len(bins)-1:
                        calib_data.loc[(calib_data[target] >= bins[i]), 'grade'] = i

            groupped = calib_data.groupby('grade')[[target, prediction_name]].mean().reset_index()

            trues = groupped[target]
            preds = groupped[prediction_name]

            return trues, preds, groupped
        

        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression

        fop, mpv, groupped = calibration_curve_numeric(calib_data, target, prediction_name, 
                                                       normalize=normalize, n_bins=n_bins, strategy=strategy, 
                                                      by_tar = by_tar)
        
        y_true = calib_data[target]
        y_preds = calib_data[prediction_name]
        
        if mpv_strategy=='median':
            mpv_med = np.median(mpv)
        elif mpv_strategy=='average':
            mpv_med = np.average(mpv)
        sdr = y_true.sum()/y_true.shape[0]  #среднее значение на куске для калибровки

        fop_sc_lst = []
        for i in range(len(fop)):
            dr_i = fop[i]
            fop_sc = (dr_i*ct/sdr) #/ ((1-dr_i)*((1-ct)/(1-sdr))+dr_i*ct/sdr)
            fop_sc_lst.append(fop_sc)

        odds_lst = []
        score_lst = []
        for i in range(len(fop_sc_lst)):
            odds = fop_sc_lst[i]
            #if dr_i_sc==1:
                #raise("Error. DRi = 1")
                #odds = np.log((1-dr_i_sc+0.0001)/dr_i_sc)
            #if dr_i_sc==0:
                #raise("Error. DRi = 0")
                #odds = np.log((1-dr_i_sc-0.0001)/dr_i_sc)
            #else:
                #odds = np.log((1-dr_i_sc)/dr_i_sc)
            #score = mpv[i] - mpv_med
            #odds_lst.append(odds)
            #score_lst.append(score)
            
        odds_lst = pd.DataFrame(fop_sc_lst) 
        score_lst = pd.DataFrame(mpv) 

        lin_reg = LinearRegression()
        lin_reg.fit(score_lst, odds_lst)

        a = lin_reg.intercept_[0]
        b = lin_reg.coef_[0][0]

        if plt_show:
            plt.figure(figsize=[6,4])
            plt.plot(score_lst.loc[:,0], odds_lst.loc[:,0], "s-", label='', alpha=1)
            plt.tight_layout()
            plt.grid(True, alpha=0.65)
            plt.xlabel('PD', fontsize=10)
            plt.ylabel('Odds', fontsize=10)
            plt.title('График Odds-PD', fontsize=10)
            #plt.savefig(savefig+'/SIGM_CALIB_PLOT_'+seg_name+'_'+score_partn, bbox_inches ='tight', pad_inches = 0.1)
            plt.show()

        return a, b, mpv_med, lin_reg
# In[ ]:
def lr_test_one_simple_numeric(model, data, y, feature_restr, params, sample_weight = None,
                           features = None, feature_empty = False, 
                       intercept = False):
    
    """
    model - обученная модель
    data - данные
    y - таргет
    feature_restr - набор переменных с ограничениями
    params - параметры обучения модели. Можно получить из обученной модели с помощью model.get_params()
    class_weight - class_weight
    features - список переменных "длинной" модели
    features_empty - если True, то проверяется значимость всех переменных одновременно
    intercept - если True, то проверяется значимость свободного члена
    
    """
            
    from scipy.stats import f
    from sklearn.metrics import mean_squared_error
    
    if features is None:
        features = list(data.columns)
        
    model_alt = model(**params).fit(data[features], y)

    if feature_empty == True:

        features_empty = pd.DataFrame(np.ones(X_1_2.shape[0])).rename(columns = {0: 'ones'})
        null_pred = sum(y)/len(y)*np.ones(len(y))
        df = len(features)

    elif intercept == True:

        model2 = model(**params)
        model2 = model2.set_params(**{'fit_intercept': False})
        model1 = model2.fit(data[features], y)
        null_pred = model1.predict(data[features])
        df = 1

    else:
        if len(feature_restr) < 1:
            raise ValueError('At least one feature is needed for H0!')
        model2 = model(**params)
        model1 = model2.fit(pd.DataFrame(data[feature_restr]), y)
        null_pred = model1.predict(pd.DataFrame(data[feature_restr]))
        df = len(features) - len(feature_restr)
    
    n = data.shape[0]
    m = data.shape[1]
    alt_pred = model_alt.predict(data[features])
    null_log_likelihood = mean_squared_error(y, null_pred, sample_weight = sample_weight)
    alternative_log_likelihood = mean_squared_error(y, alt_pred, sample_weight = sample_weight)
    F = (null_log_likelihood - alternative_log_likelihood)*(n-m)/(alternative_log_likelihood*df)
    p_value = f.sf(F, df, n-m)
    
    return p_value


# In[ ]:

# define Python user-defined exceptions
class CorrException(Exception):
    """Raised when the correlation between spline"""
    pass

def for_splines(X_1, X_2, y_train, categorical_cols, max_d =2, random_state = 241, task = 'binary', silent=False, strategy = 'recursive'):
    
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from copy import deepcopy
    
    X_1_spline = X_1.copy()
    X_2_spline = X_2.copy()
    X_1_spline.reset_index(inplace=True, drop=True)
    X_2_spline.reset_index(inplace=True, drop=True)
    
    max_d_init = max_d
    
    columns = list(X_1.columns)
    from patsy import bs
    
    split_points = []
    for i in columns:
        result_tab_num=pd.DataFrame(["For recursive"])
        max_d = max_d_init
        while result_tab_num.shape[0] != 0: 

            if i not in categorical_cols:
                columns_spline = [i+'_spline_' +str(depth) for depth in np.arange(2**max_d)]
                if task == 'numeric':
                    dec_tree_var = DecisionTreeRegressor(max_depth=max_d, random_state = random_state)
                    dec_tree_var_trained = dec_tree_var.fit(pd.DataFrame(X_1[i]), y_train)
                elif task == 'binary':
                    dec_tree_var = DecisionTreeClassifier(max_depth=max_d, random_state = random_state)
                    dec_tree_var_trained = dec_tree_var.fit(pd.DataFrame(X_1[i]), y_train)
                
                tre = dec_tree_var.tree_.threshold
                bins1 = sorted(tre[np.where(tre != -2)[0]])
                bins2 = deepcopy(bins1)

                train_train = np.array(X_1[i])
                array_bs_train1 = bs(train_train, degree = 0, knots = bins1, include_intercept = True)
                columns_spline = columns_spline[:array_bs_train1.shape[1]]
                array_bs_train = pd.DataFrame(array_bs_train1, 
                                              columns = columns_spline[:array_bs_train1.shape[1]])
                X_1_spline = X_1_spline.join(array_bs_train)
                columns_spline_test = columns_spline.copy()

                #### test
                train_test = np.array(X_2[i])
                
                min_all = np.min(train_test)
                max_all = np.max(train_test)

                zero_columns_left = []
                zero_columns_right = []

                while np.min(bins2) < min_all and len(bins2) > 1:
                    bins2.pop(0)
                    zero_columns_left.append(columns_spline_test[0])
                    columns_spline_test.pop(0)

                while np.max(bins2) > max_all and len(bins2) > 1:
                    if silent is not True:
                        print('max', len(bins2), np.max(bins2), max_all)
                    bins2.pop(-1)
                    zero_columns_right.append(columns_spline_test[-1])
                    columns_spline_test.pop(-1)


                if np.max(bins2) <= max_all and np.min(bins2) >= min_all:
                    array_bs_test1 = bs(train_test, degree = 0, knots = bins2, include_intercept = True)
                    array_bs_test = pd.DataFrame(array_bs_test1, 
                                             columns = columns_spline_test[:array_bs_test1.shape[1]])
                    X_2_spline = X_2_spline.join(array_bs_test)
                
                elif len(bins2) == 1 and np.min(bins2) < min_all:
                    X_2_spline[columns_spline_test[0]] = 0
                    X_2_spline[columns_spline_test[1]] = 1

                elif len(bins2) == 1 and np.max(bins2) > max_all:
                    X_2_spline[columns_spline_test[0]] = 1
                    X_2_spline[columns_spline_test[1]] = 0 

                if len(zero_columns_right) > 0:
                    for z_col in zero_columns_right:
                        X_2_spline[z_col] = 0
                
                if len(zero_columns_left) > 0:
                    for z_col in zero_columns_left:
                        X_2_spline[z_col] = 0

                for v in columns_spline[:array_bs_train1.shape[1]]:
                    X_1_spline[v] = X_1_spline[v]*X_1_spline[i]
                    X_2_spline[v] = X_2_spline[v]*X_2_spline[i]

                correlation = X_1_spline[columns_spline[:array_bs_train1.shape[1]]].corr()
                
                result_tab_num, non_doubles_num, col_doubles_num, double_dic_num = find_doubles_corr(X_1_spline, 
                                                                                    columns_spline[:array_bs_train1.shape[1]], 
                                                                                    correlation, 
                                                                definition = None, lvl = 0.7, light_unstable = None, silent = silent)
                #if i in ["ADRTIMEINRSDNT_HOME_AP","POSTOALL_2020_12","C_COUNTOFGP","IB_CREDIT_DO_6M"]:
                    #display(correlation, result_tab_num, X_1_spline[columns_spline])
                    #print(X_1_spline[result_tab_num.loc[0, 'double']].hist())
                    #plt.show()
                    #, X_1_spline[result_tab_num.loc[0, 'firts']].hist())
                    #print(X_1_spline[result_tab_num.loc[0, 'firts']].hist())
                    #plt.show()
                if result_tab_num.shape[0] == 0:
                    X_1_spline.drop(i, axis = 1, inplace = True)
                    X_2_spline.drop(i, axis = 1, inplace = True)
                    split_points.append([i, bins1])


                    
                elif result_tab_num.shape[0] > 0:
                    all_dups = sorted(list(set(result_tab_num['double'].to_list() + result_tab_num['firts'].to_list())))
                    if len(all_dups) == len(columns_spline[:array_bs_train1.shape[1]]):

                        for col_splines in columns_spline[:array_bs_train1.shape[1]]:

                            X_1_spline.drop(col_splines, axis = 1, inplace = True)
                            X_2_spline.drop(col_splines, axis = 1, inplace = True)
                        if silent is not True:
                            print(f'{columns_spline[0].split("_spline")[0]} не разбита на сплайны')
                            
                        break

                    else:
                        
                        try:
                            while result_tab_num.shape[0] != 0:
                                double = result_tab_num.loc[0, 'double']
                                first = result_tab_num.loc[0, 'firts']
                                nums = sorted([columns_spline.index(double), columns_spline.index(first)])
                                if abs(nums[0]-nums[1]) == 1:
                                    X_1_spline[columns_spline[nums[0]]] += X_1_spline[columns_spline[nums[1]]]
                                    X_2_spline[columns_spline[nums[0]]] += X_2_spline[columns_spline[nums[1]]]
                                    
                                    X_1_spline.drop(columns_spline[nums[1]], axis = 1, inplace = True)
                                    X_2_spline.drop(columns_spline[nums[1]], axis = 1, inplace = True)
                                    bins1.pop(nums[0])
                                    for col_num, col_spline in enumerate(columns_spline):
                                        if col_num > nums[1]:
                                            X_1_spline.rename(columns = {col_spline : columns_spline[col_num-1]}, inplace = True)
                                            X_2_spline.rename(columns = {col_spline : columns_spline[col_num-1]}, inplace = True)
                                            
                                    columns_spline.pop(-1)
                                    correlation = X_1_spline[columns_spline[:array_bs_train1.shape[1]-1]].corr()
                                    result_tab_num, non_doubles_num, col_doubles_num, double_dic_num = find_doubles_corr(X_1_spline, 
                                                                                        columns_spline[:array_bs_train1.shape[1]-1], 
                                                                                        correlation, definition = None, 
                                                                                        lvl = 0.7, light_unstable = None, 
                                                                                        silent = silent)
                                else:
                                    raise CorrException

                                    
                            X_1_spline.drop(i, axis = 1, inplace = True)
                            X_2_spline.drop(i, axis = 1, inplace = True)
                
                            split_points.append([i, bins1])
                        except CorrException:
                            if (strategy=="break"):
                                print("Use break")
                                X_1_spline.drop(i, axis = 1, inplace = True)
                                X_2_spline.drop(i, axis = 1, inplace = True)
                                
                                split_points.append([i, bins1])
                                break
                            else:
                                print(f'Уменьшаем глубину {columns_spline[0].split("_spline")[0]} на 1, новая глубина {max_d-1}')
                                max_d = max_d-1
                                X_1_spline.drop(columns_spline, axis = 1, inplace = True)
                                X_2_spline.drop(columns_spline, axis = 1, inplace = True)
            else:
                break

    split_points_data = pd.DataFrame.from_records(split_points, columns = ['attribute', 'split_points'])
    
    return X_1_spline, X_2_spline, split_points_data




def for_splines_train(X_1, y_train, categorical_cols, max_d =2, random_state = 241, task = 'binary'):
    
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from patsy import bs
    X_1_spline = X_1.copy()
    X_1_spline.reset_index(inplace=True, drop=True)

    columns = list(X_1.columns)
    correlations = []
    split_points = []
    for i in columns:
        if i not in categorical_cols:
            columns_spline = [i+'_spline_' +str(depth) for depth in np.arange(2**max_d)]
            print(columns_spline)
            if task == 'binary':
                dec_tree_var = DecisionTreeClassifier(max_depth=max_d, random_state = random_state)
                dec_tree_var_trained = dec_tree_var.fit(pd.DataFrame(X_1[i]), y_train)
            elif task == 'numeric':
                dec_tree_var = DecisionTreeRegressor(max_depth=max_d, random_state = random_state)
                dec_tree_var_trained = dec_tree_var.fit(pd.DataFrame(X_1[i]), y_train)
            
            tre = dec_tree_var.tree_.threshold
            bins1 = sorted(tre[np.where(tre != -2)[0]])

            train_train = np.array(X_1[i])
            
            array_bs_train1 = bs(train_train, degree = 0, knots = bins1, include_intercept = True)
            array_bs_train = pd.DataFrame(array_bs_train1, 
                                          columns = columns_spline[:array_bs_train1.shape[1]])
            X_1_spline = X_1_spline.join(array_bs_train)

            for v in columns_spline[:array_bs_train1.shape[1]]:
                X_1_spline[v] = X_1_spline[v]*X_1_spline[i]
                
            correlation = X_1_spline[columns_spline[:array_bs_train1.shape[1]]].corr()
            
            result_tab_num, non_doubles_num, col_doubles_num, double_dic_num = find_doubles_corr(X_1_spline, 
                                                                                columns_spline[:array_bs_train1.shape[1]], 
                                                                                correlation, 
                                                            definition = None, lvl = 0.70, light_unstable = None)
            if result_tab_num.shape[0] == 0:
                X_1_spline.drop(i, axis = 1, inplace = True)
                split_points.append([i, bins1])
                
            elif result_tab_num.shape[0] > 0:
                all_dups = sorted(list(set(result_tab_num['double'].to_list() + result_tab_num['firts'].to_list())))

                if len(all_dups) == len(columns_spline[:array_bs_train1.shape[1]]):
                    
                    for col_splines in columns_spline[:array_bs_train1.shape[1]]:
                        X_1_spline.drop(col_splines, axis = 1, inplace = True)
                else:
                    while result_tab_num.shape[0] != 0:
                        print('result_tab_num in a loop', result_tab_num.shape[0])
                        double = result_tab_num.loc[0, 'double']
                        first = result_tab_num.loc[0, 'firts']
                        nums = sorted([columns_spline.index(double), columns_spline.index(first)])

                        if abs(nums[0]-nums[1]) == 1:

                            X_1_spline[columns_spline[nums[0]]] += X_1_spline[columns_spline[nums[1]]]
                            X_1_spline.drop(columns_spline[nums[1]], axis = 1, inplace = True)
                            bins1.pop(nums[0])
                            for col_num, col_spline in enumerate(columns_spline):
                                if col_num > nums[1]:
                                    X_1_spline.rename(columns = {col_spline : columns_spline[col_num-1]}, inplace = True)
                                    
                            columns_spline.pop(-1)
                            correlation = X_1_spline[columns_spline[:array_bs_train1.shape[1]-1]].corr()
                            result_tab_num, non_doubles_num, col_doubles_num, double_dic_num = find_doubles_corr(X_1_spline, 
                                                                                columns_spline[:array_bs_train1.shape[1]-1], 
                                                                                correlation, 
                                                            definition = None, lvl = 0.70, light_unstable = None)
                        else:
                            print("Уменьшаем глубину на 1")
                            X_1_spline, split_points_data = for_splines_train(X_1, y_train, categorical_cols, max_d = max_d-1, random_state = 241, task = 'binary')
                            return X_1_spline, split_points_data
                            break
                            
                    X_1_spline.drop(i, axis = 1, inplace = True)
                            
#             correlations.append(result_tab_num)
                    split_points.append([i, bins1])
            
#             X_1_spline.drop(i, axis = 1, inplace = True)

    split_points_data = pd.DataFrame.from_records(split_points, columns = ['attribute', 'split_points'])
    
    return X_1_spline, split_points_data #, correlations

    
def for_splines_test(X_1, split_points_data):
    from patsy import bs
    
    X_1_spline = X_1.copy()
    X_1_spline.reset_index(inplace=True, drop=True)

    for i in split_points_data['attribute'].to_list():
        if isinstance(split_points_data.loc[split_points_data['attribute'] == i, 'split_points'].to_list()[0], list):
            bins1 = split_points_data.loc[split_points_data['attribute'] == i, 'split_points'].to_list()[0]
        else:
            bins1 = eval(split_points_data.loc[split_points_data['attribute'] == i, 'split_points'].to_list()[0])

        columns_spline = [i+'_spline_' +str(depth) for depth in np.arange(len(bins1)+1)]
        
        train_train = np.array(X_1[i])
        min_all = np.min(train_train)
        max_all = np.max(train_train)
        
        zero_columns_left = []
        zero_columns_right = []
        
        while np.min(bins1) < min_all and len(bins1) > 1:
            bins1.pop(0)
            zero_columns_left.append(columns_spline[0])
            columns_spline.pop(0)
            
        while np.max(bins1) > max_all and len(bins1) > 1:

            bins1.pop(-1)
            zero_columns_right.append(columns_spline[-1])
            columns_spline.pop(-1)
            
        if len(zero_columns_left) > 0:
            for z_col in zero_columns_left:
                X_1_spline[z_col] = 0
                
        if np.max(bins1) <= max_all and np.min(bins1) >= min_all:
            array_bs_train1 = bs(train_train, degree = 0, knots = bins1, include_intercept = True)
            array_bs_train = pd.DataFrame(array_bs_train1, 
                                              columns = columns_spline[:array_bs_train1.shape[1]])
            X_1_spline = X_1_spline.join(array_bs_train)
            
        elif len(bins1) == 1 and np.min(bins1) < min_all:
            X_1_spline[columns_spline[0]] = 0
            X_1_spline[columns_spline[1]] = 1
            
        elif len(bins1) == 1 and np.max(bins1) > max_all:
            X_1_spline[columns_spline[0]] = 1
            X_1_spline[columns_spline[1]] = 0 
        
        if len(zero_columns_right) > 0:
            for z_col in zero_columns_right:
                X_1_spline[z_col] = 0

        for v in columns_spline[:array_bs_train1.shape[1]]:
            X_1_spline[v] = X_1_spline[v]*X_1_spline[i]

        X_1_spline.drop(i, axis = 1, inplace = True)
        
    return X_1_spline
# In[ ]:

def parameter_optimization(model_name, X_train, y_train, path, prediction="predict_proba", optimize_metric="roc_auc_score", X_test=None, y_test=None, X_oot=None, y_oot=None,
                           conf=None, direction='minimize', n_trials=1):
    """
    model_name - Название модели (class.__name__)
    X_train, y_train - обучающие данные
    prediction - вид предсказания ("predict_proba" или "predict")
    optimize_metric - метрика выбора лучшей модели на тесте ("roc_auc_score")
        direction - правило выбора лучшей модели ('minimize'-минимум значение метрики, 'maximize'- максимум значение метрики)
    n_trials - кол-во шагов подбора
    conf - параметры модели в форматах ({param_name : [int(a), int(b)] - перебор целых чисел от a до b
                                         param_name : [float(a), int(b)] - перебор в диапазоне от a до b
                                         param_name : [а, b, с] - выбор одного из списка
                                         param_name : а - присвоение занчения а
                                        })
    direction - правило выбора лучшей модели ('minimize'-минимум значение метрики, 'maximize'- максимум значение метрики)
    n_trials - кол-во шагов подбора
    """
    BEST_LOSS = float(0) if direction=='maximize' else float('Inf')
    # EMBEDDING_DIM = X_train.shape[1]
    LOSS_FILE = path+'/loss.txt'
    PARAM_FILE = path+'/param.txt'
    CNT=0 
    def objective(trial):
        nonlocal BEST_LOSS
        nonlocal CNT
    
        try:
            f = open(LOSS_FILE)
            BEST_LOSS = float(f.read())
            print(BEST_LOSS)
            f.close()
        except IOError:
            print ("Отсутствует история обучения")
        
        config={}
        for i in conf.keys():
            if type(conf[i]) is not list or len(conf[i])==1:
                config[i]=conf[i]
            elif len(conf[i])>2:
                config[i]=trial.suggest_categorical(i, conf[i])
            elif len(conf[i])==2:
                if type(conf[i][0]) is type(conf[i][1]) and type(conf[i][0]) is int:
                    config[i]=trial.suggest_int(i, conf[i][0], conf[i][1])
                elif type(conf[i][0]) in (float,int) and type(conf[i][1]) in (float,int):
                    config[i]=trial.suggest_loguniform(i, conf[i][0], conf[i][1])
                else:
                    print(f"Проверьте тип данных для {i}={conf[i]}, оба значения должны быть либо int, либо float")
            else:
                print(f"Проверьте тип данных для {i}={conf[i]}, значения должны либо задавать промежуток от и до, либо содержать перечисление возможных значений")
        
        pprint({**config})
        
        model = model_name(**config)
        print('Starting training...', datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"))
        model.fit(X_train, y_train)
        y_pred_train = getattr(model, prediction)(X_train)
        print('Ending training...', datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"))

        metric_train = getattr(sklearn.metrics, optimize_metric)(y_train, y_pred_train)

        try:
            y_pred_test = getattr(model, prediction)(X_test)
            metric_test = getattr(sklearn.metrics, optimize_metric)(y_test, y_pred_test)
        except:
            metric_test = None
        try:
            y_pred_oot = getattr(model, prediction)(X_oot)
            metric_oot = getattr(sklearn.metrics, optimize_metric)(y_oot, y_pred_oot)
        except:
            metric_oot = None
            
        if X_test==None and y_test==None:
            print("Выбор гиперпараметров происходит на основе метрики обучающих данных")
            metric_test=metric_train
        if metric_test<float("Inf") and metric_test>-float("Inf"):
            if direction=='minimize':
                if metric_test < BEST_LOSS:
                    BEST_LOSS = metric_test
                    with open(LOSS_FILE, 'w') as loss_history:
                        loss_history.write(str(BEST_LOSS))
                    with open(PARAM_FILE, 'w') as param_dict:
                        param_dict.write(str(config))
                    joblib.dump(model, path+f'/model_{CNT}.dat')
            else:
                if metric_test > BEST_LOSS:
                    BEST_LOSS = metric_test
                    with open(LOSS_FILE, 'w') as loss_history:
                        loss_history.write(str(BEST_LOSS))
                    with open(PARAM_FILE, 'w') as param_dict:
                        param_dict.write(str(config))
                    joblib.dump(model, path+f'/model_{CNT}.dat')
                    
        CNT+=1
        return metric_test
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

def find_meta_params_optuna(X,
                            Y,
                            params_dictionary,
                            pass_model,
                            sort_by_var,
                            list_of_vars_for_strat,
                            n_folds,
                            second_target,
                            yeo_johnson,
                            attribute_list,
                            var_col,
                            categorial_list=None,
                            cols_outlier=None,
                            need_business=True,
                            draw=True,
                            draw_by_approval_rate=False,
                            simple_b_score=None,
                            business_dict=None,
                            business_dict_sec=None,
                            scale='mean',
                            median='median',
                            high_outlier=None,
                            low_outlier=None,
                            check_percentile=1,
                            random_state=None,
                            task='binary',
                            multiclass_aggregator_model=None,
                            k_logs=10,
                            cut_non_out_9999=True,
                            optimize_metric="AUC",
                            direction='maximize',
                            n_trials=1,
                            path="",
                            n_jobs=2,
                            a=0.01,
                            k_fold_method="mean",
                            additional_find=False,
                            save_best_models=True, blend=False, second_score=None, tfidf_preparation=None, embeding=False, catboost=False):
    
    """
    Функция find_meta_params получается meta файл, в котором содержится brute-force поиск по сетке. Функция может применяться к различным моделям и для различных параметров, аналогично GridSearchCV. Параметры:

    - X, Y - матрица данных и таргет. Матрица X не должна быть предобработана, так как предобработка делается внутри функции! 

    - params_dictionary - словарь с параметрами, параметры модели в форматах
                                        ({param_name : [int(a), int(b)] - перебор целых чисел от a до b
                                          param_name : [float(a), int(b)] - перебор в диапазоне от a до b
                                          param_name : [а, b, с] - выбор одного из списка
                                          param_name : а - присвоение занчения а
                                        })

    - pass_model - вызываемая функция. Пример: pass_model = LogisticRegression или pass_model = DecisionTreeClassifier

    - sort_by_var - переменная для деления (пример - id клиента, клиенты должны попасть либо в тест, либо в трейн)
    
    - list_of_vars_for_strat - список переменных для стратификации. Пример: распределение по регионам, распределение по месяцам
    
    - n_folds - количество фолдов

    - second_target - используется ли второй таргет (как в research для модели CRM)

    - yeo_johnson - используется ли преобразование Йео-Джонсона. Если его нет, делается стандартизация StandardScaler (Можно передовать list для подбора)
    
    - attribute_list - аттрибут лист для использования его в импутациях пропусков и обрезании выбросов
    
    - var_col - имя поля, в котором в attribute_list находятся названия переменных
    
    - need_business - флажок True/False. Считать ли бизнес метрику. По умолчанию True
    
    - draw - флажок True/False. Рисовать ли картинки. По умолчанию True
    
    - draw_by_approval_rate - флажок True/False. Рисовать ли картинки по Approval_rate вместо treshold. По умолчанию False
    
    - simple_b_score - функция, которая расчитывает бизнес метрику
    
    - business_dict - словарь с параметрами для бизнес метрики. Пример для CRM:business_dictionary = {'t0': 0.1, 'm_s': 19000, 'fund': 1, 'k': 20, 'c': 3}
    
    - business_dic_sec - словарь с параметрами для бизнес метрики для второго таргета, если он используется
    
    - scale - делать ли стандартизацию. По дефолту True (Можно передовать list для подбора)
    
    - median - импутация пропусков. Возможные принимаемые значения:
        - 'median' - тогда на train куске рассчитываются медианы, и импутация ими
        - 'min-1' - тогда на train куске рассчитываются минимальные значения - 1, и импутация ими
        - число - если задать число, то пропуски будут заполняться этим числом
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
        - None. Тогда пропуски не импутируются
        По дефолту задано значение 'median'
    - high_outlier - импутация выбросов вверх. Возможные принимаемые значения: (Можно передовать list для подбора)
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 99 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
        - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
        - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
        - None. Тогда выбросы вверх не импутируются.
    - low_outlier - импутация выбросов вниз. Возможные принимаемые значения: (Можно передовать list для подбора)
        - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 1 перцентиль
        - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
        - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
        - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
        - None. Тогда выбросы вниз не импутируются. 
        
    - task - задача регрессии или классификации. binar для задачи классификации, numeric - для задачи регрессии. По умолчанию binar
    - k_logs - как часто отображать результат. По умолчанию отображать каждую десятую итерацию
    - optimize_metric - метрика выбора лучшей модели, одна из 
                ['ScoreF1','Acc','Pre','Rec', 'APS','Brier_score','AUC','b_best','cutoff',
                'Bad_Rate','ScoreF1_second_target','Acc_second_target',
                'Pre_second_target','Rec_second_target','APS_second_target','Brier_score_second_target',
                'AUC_second_target','b_best_second_target','cutoff_second_target','Bad_Rate_second_target',
                'R2','MSE','MAE','MedianAE','RMSE','R2_second_target','MSE_second_target',
                'MAE_second_target','MedianAE_second_target','RMSE_second_target']
    - direction - правило выбора лучшей модели ('minimize'-минимум значение метрики, 'maximize'- максимум значение метрики)
    - n_trials - кол-во шагов подбора
    - path - путь сохранения модели и лосса
    - additional_find - продолжать предыдущий поиск
    ВАЖНО!!!!!!!!!!!! Для поиска по сетке не нужно подавать данные с импутированными пропусками/выбросами! Более того, не стоит 
    использовать attribute_list, так как на каждом разбиении должна считаться собственная импутация.
    
    random_state - random_state для биения данных на куски. random_state для модели следует подавать в словаре для обучения!!!!

    Возвращает таблицу meta с сеткой и показателями. 
    """
    from collections.abc import Iterable
    from datetime import datetime
    from pprint import pprint

    import joblib
    import optuna
    
    def train_and_receive_stats_binar(model, xtrain, ytrain, xtest, ytest, scores, 
                                      second_target = None, y_train_2 = None, y_test_2 = None,
                                     draw = True, draw_by_approval_rate = False, catboost=False, categorial_list=None):
        if catboost:
            from catboost import Pool
            cat_features_names = categorial_list
            cat_features = [xtrain.columns.get_loc(col) for col in cat_features_names]
            print(cat_features)
            #train_pool = Pool(xtrain, ytrain, cat_features=cat_features)
            #test_pool = Pool(xtest, ytest, cat_features=cat_features)
            #model.fit(train_pool)
            print(xtrain.columns)
            model.fit(xtrain, ytrain, eval_set=(xtest, ytest), use_best_model=True, plot=False)
            # Predict
            yhat_test = model.predict(xtest)
          
            yhat_test_proba = model.predict_proba(xtest)[:,1]
            yhat_train_proba = model.predict_proba(xtrain)[:,1]
            #yhat_test_proba = resample(yhat_test_proba, n_samples = len(ytest))
        else:
            model.fit(xtrain, ytrain)
            yhat_test = model.predict(xtest)
            
            yhat_test_proba = model.predict_proba(xtest)[:,1]
            yhat_train_proba = model.predict_proba(xtrain)[:,1]
        
        scores['ScoreF1'].append(metrics.f1_score(ytest, yhat_test))
        scores['Acc'].append(metrics.accuracy_score(ytest, yhat_test))
        scores['Pre'].append(metrics.precision_score(ytest, yhat_test))
        scores['Rec'].append(metrics.recall_score(ytest, yhat_test)) 
        scores['APS'].append(metrics.average_precision_score(ytest, yhat_test_proba))
        scores['Brier_score'].append(metrics.brier_score_loss(ytest, yhat_test_proba))
        scores['AUC'].append(metrics.roc_auc_score(ytest, yhat_test_proba))
        scores['Score'].append(yhat_test_proba)
        scores['Bad_Rate'].append(ytest.value_counts()[1]/len(ytest))
        # находим лучший cut-off по трейн и применяем его для тест!!
        #best_score_max, cut_off_max, best_score_thr, cut_off_thr
        
        if need_business == True:
            b_best_train_max, cutoff_train_max, b_best_max = b_score_train_and_test(ytrain,
                                                                        yhat_train_proba, ytest, 
                                                                    yhat_test_proba, simple_b_score, business_dict)
            if draw == True:
                b_score_array, approval_rate, cutoff, best_sc, best_cutoff = max_prof_corve(ytest, yhat_test_proba, simple_b_score,
                                                                                                    business_dict)
                b_score_array_train, approval_rate_train, cutoff_train, best_sc_train, best_cutoff_train = max_prof_corve(ytrain, 
                                                                                                                yhat_train_proba, 
                                                                                                                simple_b_score,
                                                                                                                business_dict)
                if draw_by_approval_rate == False:
                    x_plot = cutoff
                    y_plot = b_score_array
                    c = next(color) 
                    x_plot_train = cutoff_train
                    y_plot_train = b_score_array_train #/len(y_test)

                    if k/k_logs == int(k/k_logs) or k == 1:
                        ax_each.scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.1)
                        ax_each.scatter(x_plot_train, y_plot_train, s = 0.1, color=c, alpha=0.1)
                        ax_each.plot([best_cutoff_train, best_cutoff_train], [0, best_sc_train], '--', color=c, alpha=0.8)
                        ax_each.plot([best_cutoff, best_cutoff], [0, best_sc], '--', color=c, alpha=0.8)

                        #axs[k].scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.5, linewidths = 0.2)
                        #axs[k].tick_params(labelsize=2, which='both', labelbottom=True, labelleft=True, width = 0.2)

                    axs[k].scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.1, linewidth=0.2)
                    axs[k].scatter(x_plot_train, y_plot_train, s = 0.1, color=c, alpha=0.1, linewidth=0.2)
                    axs[k].plot([best_cutoff_train, best_cutoff_train], [0, best_sc_train], '--', linewidth=0.2, color=c, alpha=0.8)
                    axs[k].plot([best_cutoff, best_cutoff], [0, best_sc], '--', linewidth=0.2, color=c, alpha=0.8)
                    axs[k].tick_params(labelsize=2, which='both', labelbottom=True, labelleft=True, width = 0.2)

                else:
                    x_plot = approval_rate
                    y_plot = b_score_array
                    c = next(color) 
                    x_plot_train = approval_rate_train
                    y_plot_train = b_score_array_train #/len(y_test)

                    if k/k_logs == int(k/k_logs) or k == 1:
                        ax_each.scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.1)
                        ax_each.scatter(x_plot_train, y_plot_train, s = 0.1, color=c, alpha=0.1)

                    axs[k].scatter(x_plot, y_plot, s = 0.1, color=c, alpha=0.1, linewidth=0.2)
                    axs[k].scatter(x_plot_train, y_plot_train, s = 0.1, color=c, alpha=0.1, linewidth=0.2)
                    axs[k].tick_params(labelsize=2, which='both', labelbottom=True, labelleft=True, width = 0.2)

            scores['b_best'].append(b_best_max)
            scores['cutoff'].append(cutoff_train_max)
                                    
        if type(second_target) != type(None):
            
            scores['ScoreF1_second_target'].append(metrics.f1_score(y_test_2, yhat_test))
            scores['Acc_second_target'].append(metrics.accuracy_score(y_test_2, yhat_test))
            scores['Pre_second_target'].append(metrics.precision_score(y_test_2, yhat_test))
            scores['Rec_second_target'].append(metrics.recall_score(y_test_2, yhat_test)) 
            scores['APS_second_target'].append(metrics.average_precision_score(y_test_2, yhat_test_proba))
            scores['Brier_score_second_target'].append(metrics.brier_score_loss(y_test_2, yhat_test_proba))
            scores['AUC_second_target'].append(metrics.roc_auc_score(y_test_2, yhat_test_proba))
            scores['Bad_Rate_second_target'].append(y_test_2.value_counts()[1]/len(y_test_2))
            # находим лучший cut-off по трейн и применяем его для тест!!
            #best_score_max, cut_off_max, best_score_thr, cut_off_thr
                    
            if need_business == True:
                b_best_train_max, cutoff_train_max, b_best_max  = b_score_train_and_test(y_train_2, 
                                                            yhat_train_proba, y_test_2, yhat_test_proba, simple_b_score,
                                                                                     business_dict_sec)           

                scores['b_best_second_target'].append(b_best_max)
                scores['cutoff_second_target'].append(cutoff_train_max)
        
        if need_business == True:
            if draw == True:
                if draw_by_approval_rate == False:
                    if k/10 == int(k/10) or k == 1:
                        ax_each.set_xlabel('Treshold')
                        ax_each.set_ylabel('Profit')
                        ax_each.set_title(parameters)
                        plt.show()

                    axs[k].set_xlabel('Treshold', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_ylabel('Profit', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_title(parameters, fontdict = {'fontsize': 2, 'fontweight' : 2})

                else:
                    if k/10 == int(k/10) or k == 1:
                        ax_each.set_xlabel('Approval Rate')
                        ax_each.set_ylabel('Profit')
                        ax_each.set_title(parameters)
                        plt.show()

                    axs[k].set_xlabel('Approval Rate', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_ylabel('Profit', fontdict = {'fontsize': 2, 'fontweight' : 2})
                    axs[k].set_title(parameters, fontdict = {'fontsize': 2, 'fontweight' : 2})
        
        plt.close()
        
        return model, scores
    
    def train_and_receive_stats_numeric(model, xtrain, ytrain, xtest, ytest, 
                                        scores, second_target = None, y_train_2 = None, y_test_2 = None):
        
        model.fit(xtrain, ytrain)
        yhat_test = model.predict(xtest)
        
        scores['R2'].append(metrics.r2_score(ytest, yhat_test))
        scores['MSE'].append(metrics.mean_squared_error(ytest, yhat_test))
        scores['MAE'].append(metrics.mean_absolute_error(ytest, yhat_test))
        scores['MedianAE'].append(metrics.median_absolute_error(ytest, yhat_test))
        #scores['MSLE'].append(metrics.mean_squared_log_error(ytest, yhat_test))
        scores['RMSE'].append(np.sqrt(metrics.mean_squared_error(ytest, yhat_test)))
        #scores['RMSLE'].append(np.sqrt(metrics.mean_squared_log_error(ytest, yhat_test)))
            
        if type(second_target) != type(None):
		    # Исправлено
			# Исправлено
            scores['R2_second_target'].append(metrics.r2_score(y_test_2, yhat_test))
            scores['MSE_second_target'].append(metrics.mean_squared_error(y_test_2, yhat_test))
            scores['MAE_second_target'].append(metrics.mean_absolute_error(y_test_2, yhat_test))
            scores['MedianAE_second_target'].append(metrics.median_absolute_error(y_test_2, yhat_test))
            #scores['MSLE_second_target'].append(metrics.mean_squared_log_error(y_test_2, yhat_test))
            scores['RMSE_second_target'].append(np.sqrt(metrics.mean_squared_error(y_test_2, yhat_test)))
            #scores['RMSLE_second_target'].append(np.sqrt(metrics.mean_squared_log_error(y_test_2, yhat_test)))
            
        return model, scores

    def train_and_receive_stats_multiclass(model, xtrain, ytrain, xtest, ytest, scores):
        model.fit(xtrain, ytrain)

        yhat_test = model.predict(xtest)
        yhat_test_proba = model.predict_proba(xtest)

        scores['ScoreF1'].append(metrics.f1_score(ytest, yhat_test, average = 'weighted'))
        scores['AUC'].append(metrics.roc_auc_score(ytest, yhat_test_proba, average = 'weighted', multi_class='ovr'))
        scores['Score'].append(yhat_test_proba)

        return model, scores

    def data_preprocessing_meta(X_1, y_1, X_2, y_2, technical_values, categorial_list = None, yeo_johnson = False, 
                                attribute_list = None, var_col = None, scale = 'mean', median = 'median',
                                high_outlier = None, low_outlier = None, cols_outlier = None, cut_non_out_9999 = True, 
                                check_percentile = 1):
    
      
        """
        Проводит препроцессинг для train и test выборки.

        X_1, y_1, X_2, y_2 - данные
        technical_values - список технических переменных
        technical_values исключены из анализа и удаляются из выборки. Если технических переменных нет, можно задать пустой список. 

        yeo_johnson - проводить ли нормализацию Йео-Джонсона (приведение распределения данных к нормальному
        виду). По дефолту False
        attribute_list - данные attribute_list. Могут быть None. По дефолту None
        var_col - в каком поле attribute_list находятся названия фичей. Если attribute_list не задан, то в var_col нет нужды.
        По дефолту None.
        scale - проводить ли стандартизацию StandardScaler. По дефолту True.

        median - импутация пропусков. Возможные принимаемые значения:
            - 'median' - тогда на train куске рассчитываются медианы, и импутация ими и train и test кусков
            - 'min-1' - тогда на train куске рассчитываются минимальные значения - 1, и импутация ими
            - число - если задать число, то пропуски будут заполняться этим числом
            - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'val_mediana')
            - None. Тогда пропуски не импутируются
            По дефолту задано значение 'median'
        high_outlier - импутация выбросов вверх. Возможные принимаемые значения:
            - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 99 перцентиль
            - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_99')
            - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
            - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
            - None. Тогда выбросы вверх не импутируются.
        low_outlier - импутация выбросов вниз. Возможные принимаемые значения:
            - число от 0 до 100. Тогда на train рассчитываются соответствующие значения перцентиля. Например, 1 перцентиль
            - поле из attribute_list, в котором находится показатель, которым импутируются пропуски (например 'percentile_1')
            - IQR. Тогда на train считается IQR с весом 1.5 и импутация с его помощью 
            - z-score. Тогда на train считается z-score с порогом 3 и импутация с его помощью
            - None. Тогда выбросы вниз не импутируются. 

         Возвращает измененные данные и обученный Scaler(или Йео_Джонсон)

        """

        xtrain = X_1.copy()
        xtest = X_2.copy()
        ytrain = y_1.copy()
        ytest = y_2.copy()

        for i in technical_values:
            if i in xtrain.columns:
                xtrain.drop(i, axis = 1, inplace = True)
            if i in xtest.columns:
                xtest.drop(i, axis = 1, inplace= True)

        if type(categorial_list) != type(None):
            categorial_cols = categorial_list
            numeric_cols = list(xtrain.columns) 
            for i in categorial_cols:
                if i in numeric_cols:
                    numeric_cols.remove(i)
        else:
            categorial_cols = []
            for cc in xtrain.columns:
                if xtrain[cc].nunique() == 2:
                    if sorted(xtrain[cc].unique())[0] == 0 and sorted(xtrain[cc].unique())[1] == 1:
                        categorial_cols.append(cc) 
            numeric_cols = list(xtrain.columns) 
            for i in categorial_cols:
                if i in numeric_cols:
                    numeric_cols.remove(i)

        if type(cols_outlier) == type(None):
            cols_outlier = xtrain.columns
            
        test_ind = xtest.index

        for oo in numeric_cols:
            if median != None:
                if median == 'median':
                    medians = xtrain[oo].median(skipna = True)
                elif median == 'min-1':
                    medians = xtrain[oo].min(skipna = True)-1
                elif type(attribute_list) != type(None) and median in attribute_list.columns:
                    medians = list(attribute_list.loc[attribute_list[var_col] == oo, median])[0]
                else:
                    medians = median

            if high_outlier != None:
                if oo in cols_outlier:
                    if type(attribute_list) != type(None) and high_outlier in attribute_list.columns:
                        to_replace_high = list(attribute_list.loc[attribute_list[var_col] == oo, high_outlier])[0]
                    elif high_outlier == 'IQR':
                        check_25 = np.nanpercentile(xtrain[oo], 25)
                        check_75 = np.nanpercentile(xtrain[oo], 75)
                        check_99 = np.nanpercentile(xtrain[oo], 100-check_percentile)
                        maximum = xtrain[oo].max()

                        if check_25 != check_75:
                            q_25 = np.nanpercentile(xtrain[oo], 25)
                            q_75 = np.nanpercentile(xtrain[oo], 75)
                            iqr = q_75-q_25
                            right_border = q_75+iqr*1.5
                        else:
                            x = xtrain.loc[xtrain[oo] != check_25, oo]
                            q_25 = np.nanpercentile(x, 25)
                            q_75 = np.nanpercentile(x, 75)
                            iqr = q_75-q_25
                            right_border = q_75+iqr*1.5

                        if right_border > maximum:
                            to_replace_high = maximum
                        elif check_99 > right_border:
                            to_replace_high = check_99
                        else:
                            to_replace_high = right_border
                        
                        
                    elif high_outlier == 'z-score':
                        to_replace_high = 3*xtrain[oo].std()+xtrain[oo].mean()
                    else:
                        to_replace_high = np.nanpercentile(xtrain[oo], high_outlier)
                elif oo not in cols_outlier:
                    if oo in numeric_cols:
                        if cut_non_out_9999 == True:
                            to_replace_high = np.nanpercentile(xtrain[oo], 99.99)
            elif high_outlier == None:
                to_replace_high = None

            if low_outlier != None:
                if oo in cols_outlier:
                    if type(attribute_list) != type(None) and low_outlier in attribute_list.columns:
                        to_replace_low = list(attribute_list.loc[attribute_list[var_col] == oo, low_outlier])[0]
                        
                    elif low_outlier == 'IQR':
                        check_25 = np.nanpercentile(xtrain[oo], 25)
                        check_75 = np.nanpercentile(xtrain[oo], 75)
                        check_1 = np.nanpercentile(xtrain[oo], check_percentile)
                        minimum = xtrain[oo].min()
                        if check_25 != check_75:
                            q_25 = np.nanpercentile(xtrain[oo], 25)
                            q_75 = np.nanpercentile(xtrain[oo], 75)
                            iqr = q_75-q_25
                            left_border = q_25-iqr*1.5
                        else:
                            x = xtrain.loc[xtrain[oo] != check_25, oo]
                            q_25 = np.nanpercentile(x, 25)
                            q_75 = np.nanpercentile(x, 75)
                            iqr = q_75-q_25
                            left_border = q_25-iqr*1.5

                        if left_border < minimum:
                            to_replace_low = minimum
                        elif check_1 < left_border:
                            to_replace_low = check_1
                        else:
                            to_replace_low = left_border
                        
                    elif low_outlier == 'z-score':
                        to_replace_low = (-1)*3*xtrain[oo].std()+xtrain[oo].mean()
                    else:
                        to_replace_low = np.nanpercentile(xtrain[oo], low_outlier)
                        
                elif oo not in cols_outlier:
                    if oo in numeric_cols:
                        to_replace_low = min(xtrain[oo])
            elif low_outlier == None:
                to_replace_low = None

            if median != None:
                xtrain[oo] = xtrain[oo].fillna(medians)
                xtest[oo] = xtest[oo].fillna(medians)
            if to_replace_high != None:
                if oo in cols_outlier:
                    xtrain.loc[xtrain[oo] > to_replace_high, oo] = to_replace_high
                    xtest.loc[xtest[oo] > to_replace_high, oo] = to_replace_high
            if to_replace_low != None:
                if oo in cols_outlier:
                    xtrain.loc[xtrain[oo] < to_replace_low, oo] = to_replace_low
                    xtest.loc[xtest[oo] < to_replace_low, oo] = to_replace_low


        if yeo_johnson == False and scale != False:
            if scale == 'mean':
                pr = preprocessing.StandardScaler()
                pr.fit(xtrain[numeric_cols])
                xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
                xtest[numeric_cols] = pr.transform(xtest[numeric_cols])
                return xtrain, xtest, ytrain, ytest
            elif scale == 'minmax':
                pr = preprocessing.MinMaxScaler()
                pr.fit(xtrain[numeric_cols])
                xtrain[numeric_cols] = pr.transform(xtrain[numeric_cols])
                xtest[numeric_cols] = pr.transform(xtest[numeric_cols])

                return xtrain, xtest, ytrain, ytest

        elif yeo_johnson == True:
            power = PowerTransformer(method = 'yeo-johnson', standardize = False).fit(xtrain[numeric_cols])
            xtrain[numeric_cols] = power.transform(xtrain[numeric_cols])
            xtest[numeric_cols] = power.transform(xtest[numeric_cols])
            pr = power

            pr2 = preprocessing.StandardScaler()
            pr2.fit(xtrain[numeric_cols])
            xtrain[numeric_cols] = pr2.transform(xtrain[numeric_cols])
            xtest[numeric_cols] = pr2.transform(xtest[numeric_cols])

            return xtrain, xtest, ytrain, ytest
        else:
            return xtrain, xtest, ytrain, ytest
    
    def max_prof_corve(y_true, y_score, simple_score, business_dictionary, pos_label=None, sample_weight=None):
    
        """

        Создает векторы для отрисовки кривой max_profit.
        Пример использования:

        b_auc, tp_fps_auc, cut_auc, best_auc, cutoff_auc = max_prof_corve(y_2, auc_test_pred, simple_b_score_crm, business_dictionary)

        plt.figure(figsize = (10, 10))

        plt.title('Business score test')

        plt.plot(tp_fps_auc, b_auc, color='green',
                 lw=lw, label='maxProfit AUC model')

        """

        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score = b_score.max()
        cut_off = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
        #return  tns, fns, fps, tps, y_score[threshold_idxs]
        return b_score, tp_fp, y_score[threshold_idxs], best_score, cut_off
    
    def b_score_train_and_test(y_true, y_score, y_test, y_test_score, simple_score, business_dictionary, pos_label=None, 
                               sample_weight=None):
        """
        ----------
        y_true : array, shape = [n_samples]
            True targets of binary classification
        y_score : array, shape = [n_samples]
            Estimated probabilities or decision function
        y_test: True TEST targets
        y_test_score: predictions on target
        pos_label : int or str, default=None
            The label of the positive class
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.
        Returns

            !!!!ВАЖНО!!!!

            В функцию надо подавать как прогнозы и истинные метки трейна, так и прогнозы и истинные метки теста, так как на трейне
            рассчитывается оптимальный порог, по которому считается бизнес метрика, но итоговые результаты и выводы о бизнес метрике надо
            делать на ТЕСТЕ!

        """
        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        tns =  fps[-1] - fps    
        fns =  tps[-1] - tps # Оставить, часть матрицы сопряженности. может пригодиться
        tp_fp = (tps + fps)/y_true.size 
        #b_score = ((t0*tns)/(1-t0)) - fns
        b_score = simple_score(_tn = tns, _fp = fps, _fn = fns, _tp = tps, **business_dictionary) 
        best_score_max = b_score.max()
        cut_off_max = y_score[threshold_idxs][np.where(b_score == b_score.max())[0][0]]
        #idx_max = max(np.where(tp_fp <= t0)[0])
        #best_score_thr = b_score[idx_max]
        #cut_off_thr = y_score[threshold_idxs][idx_max]
        #return  tns, fns, fps, tps, y_score[threshold_idxs]

        y_best_test_max = pd.Series(np.where(y_test_score >= cut_off_max , 1, 0))
        #y_best_test_thr = pd.Series(np.where(y_test_score >= cut_off_thr  , 1, 0))

        _tn, _fp, _fn, _tp = metrics.confusion_matrix(y_test, y_best_test_max).ravel()
                    #m_s*fund*tps - c*k*tp_fp*y_true.size 
        b_best_max = simple_score(_tn = _tn, _fp = _fp, _fn = _fn, _tp = _tp, **business_dictionary) 

        return best_score_max, cut_off_max, b_best_max#, best_score_thr, cut_off_thr # b_score , tns, fns, fps, tps, y_score[threshold_idxs]
    
    data = X.join(Y)
    
    target = data.columns[-1]

    if type(cols_outlier) == type(None):
        cols_outlier = list(data.columns)
        
    for i in list_of_vars_for_strat:
        if i in cols_outlier:
            cols_outlier.remove(i)
    if target in cols_outlier:
        cols_outlier.remove(target)
    if sort_by_var in cols_outlier:
        cols_outlier.remove(sort_by_var)
    
    max_target = data.groupby(sort_by_var).aggregate({target: 'max'})
    max_target = max_target.reset_index()

    data = pd.merge(data, max_target, on = sort_by_var, suffixes = ["", "_max"])  
    
    target1 = target+"_max"
    
    list_of_vars_for_strat1 = list_of_vars_for_strat.copy()
    if task == 'binary':
        if len(list_of_vars_for_strat1) == 0:
            list_of_vars_for_strat1 = [target1]
        if target in list_of_vars_for_strat1:
            list_of_vars_for_strat1.remove(target)
            list_of_vars_for_strat1.append(target1)
        else:
            list_of_vars_for_strat1.append(target1)
                    
    for i in list_of_vars_for_strat1:
        if i == list_of_vars_for_strat1[0]:
            data['For_stratify'] = data[i].astype('str')
        else:
            data['For_stratify'] += data[i].astype('str')

    data_nodup = data[[sort_by_var, 'For_stratify', target1]].drop_duplicates(subset = sort_by_var)
    
    cross_val = StratifiedKFold(n_splits=n_folds, shuffle = True, random_state = random_state)
    meta_container = pd.DataFrame()
    
    k = 0
          
    if draw == True:
        fig, axs = plt.subplots(n_trials, 1, figsize=(2.2, n_trials), sharey='all', sharex='all', 
                                constrained_layout=True)
        #fig = plt.figure(figsize = (5, len(combs)))
        fig.suptitle('Graphs of Max profit', fontsize=3)
        plt.close(fig)
        
    BEST_LOSS = float(0) if direction=='maximize' else float('Inf')
    LOSS_FILE = path+'/loss.txt'
    PARAM_FILE = path+'/param.txt'
    PARAM_BLEND_FILE = path+'/param_blend.txt'
    PREPROC_FILE = path+'/preproc_params.txt'
    CNT=0 

    def objective(trial):
        nonlocal BEST_LOSS
        nonlocal CNT
        nonlocal k
        nonlocal meta_container
        if additional_find:
            try:
                f = open(LOSS_FILE)
                BEST_LOSS = float(f.read())
                print(BEST_LOSS)
                f.close()
            except IOError:
                print ("Отсутствует история обучения")
        # else:
            # print(f"Trial {k}")
        preproc_config={}
        preproc_param_list=[high_outlier, low_outlier, scale, yeo_johnson]
        preproc_param_list_name=["high_outlier", "low_outlier", "scale", "yeo_johnson"]
        for i,name in zip(preproc_param_list,preproc_param_list_name):
            if (isinstance(i, Iterable) & (type(i) is not str)):
                preproc_config[name]=trial.suggest_categorical(name, i)
            else:
                preproc_config[name]=i
        config={}

        for i in params_dictionary.keys():
                                
            if len(re.findall('class_weight', i)) > 0:
                weight_list=[]
                if len(re.findall('0', i)) > 0:

                    weight_list=trial.suggest_categorical('class_weight', params_dictionary[i])
                    if catboost:
                        config['class_weights']={0:weight_list, 1:1}
                    else:
                        config['class_weight']={0:weight_list, 1:1}
                elif len(re.findall('1', i)) > 0:

                    weight_list=trial.suggest_categorical('class_weight', j)
                    if catboost:
                        config['class_weights']={0:1, 1:weight_list}
                    else:
                        config['class_weight']={0:1, 1:weight_list}

            elif not(isinstance(params_dictionary[i], Iterable)) or type(params_dictionary[i]) is str:
                config[i]=params_dictionary[i]
            elif len(params_dictionary[i])==1:
                config[i]=params_dictionary[i][0]
            elif len(params_dictionary[i])>2 or type(params_dictionary[i][0]) is str:
                config[i]=trial.suggest_categorical(i, params_dictionary[i])
            elif len(params_dictionary[i])==2:
                if type(params_dictionary[i][0]) is type(params_dictionary[i][1]) and type(params_dictionary[i][0]) is int:
                    config[i]=trial.suggest_int(i, params_dictionary[i][0], params_dictionary[i][1])
                elif type(params_dictionary[i][0]) in (float,int) and type(params_dictionary[i][1]) in (float,int):
                    config[i]=trial.suggest_loguniform(i, params_dictionary[i][0], params_dictionary[i][1])
                else:
                    raise TypeError(f"Проверьте тип данных для {i}={params_dictionary[i]}, оба значения должны быть либо int, либо float")
            else:
                raise TypeError(f"Проверьте тип данных для {i}={params_dictionary[i]}, значения должны либо задавать промежуток от и до, либо содержать перечисление возможных значений")
        # pprint({**config})
        # pprint({**preproc_config})
        
        outputs = list(config.values())
        if not callable(pass_model):
            return 'Error! Model should be callable'
        else:
            model = pass_model(**config)
            if task == 'multiclass' and callable(multiclass_aggregator_model):
                model = multiclass_aggregator_model(model)
            
        if (type(second_target) != type(None)) and (task == 'binary'):
            if need_business == False:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],    
                'AUC': [],
                'Score': [],
                'Bad_Rate': [],
                'ScoreF1_second_target': [],
                'Acc_second_target': [],
                'Pre_second_target': [],
                'Rec_second_target': [] ,
                'APS_second_target': [],
                'Brier_score_second_target': [],
                'AUC_second_target': [],
                'Bad_Rate_second_target': []
                }
                
            else:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],    
                'AUC': [],
                'Score': [],
                'b_best' : [] ,
                'cutoff' : [] ,
                'Bad_Rate': [],
                'ScoreF1_second_target': [],
                'Acc_second_target': [],
                'Pre_second_target': [],
                'Rec_second_target': [] ,
                'APS_second_target': [],
                'Brier_score_second_target': [],
                'AUC_second_target': [],
                'b_best_second_target' : [] ,
                'cutoff_second_target' : [] ,
                'Bad_Rate_second_target': []
                }
    
        elif (type(second_target) == type(None)) and (task == 'binary'):
            if need_business == True:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],
                'AUC': [],
                'Score': [],
                'b_best' : [] ,
                'cutoff' : [],
                'Bad_Rate': []
            }
            else:
                scores = {
                'ScoreF1': [],
                'Acc': [],
                'Pre': [],
                'Rec': [] ,
                'APS': [],
                'Brier_score': [],
                'AUC': [],
                'Score': [],
                'Bad_Rate': []
            }
                
        elif (type(second_target) == type(None)) and (task == 'numeric'):
            scores = {
                'R2': [],
                'MSE':[],
                'MAE':[],
                'MedianAE': [],
                #'MSLE': [],
                'RMSE': [],
                #'RMSLE': []
            }
        elif (type(second_target) != type(None)) and (task == 'numeric'):    
            
            scores = {
                'R2': [],
                'MSE':[],
                'MAE':[],
                'MedianAE': [],
                #'MSLE': [],
                'RMSE': [],
                #'RMSLE': [],
                'R2_second_target': [],
                'MSE_second_target':[],
                'MAE_second_target':[],
                'MedianAE_second_target': [],
                #'MSLE_second_target': [],
                'RMSE_second_target': [],
                #'RMSLE_second_target': []
            }

        elif task == 'multiclass':
            scores = {
                'ScoreF1': [],
                'AUC': [],
                'Score': []}
            if need_business == True:
                scores.update({'b_best': [],'cutoff': []})
        
        if draw == True:
            color=iter(cm.rainbow(np.linspace(0, 1, n_folds))) # Оставить
            fig_each, ax_each = plt.subplots(1, 1, figsize=(10, 5))    
        first_score=pd.DataFrame()     
        num=0
        for idx_train, idx_test in cross_val.split(data_nodup[sort_by_var], data_nodup['For_stratify']):
            #  def fold_job(idx_train, idx_test):   
            xtrain_id, xtest_id = data_nodup.iloc[idx_train][sort_by_var], data_nodup.iloc[idx_test][sort_by_var]
            xtrain = data[data[sort_by_var].isin(xtrain_id)].copy()
            train_index = xtrain.index
            ytrain = data.iloc[train_index][target].copy()
            xtest = data[data[sort_by_var].isin(xtest_id)].copy()
            test_index = xtest.index
            ytest = data.iloc[test_index][target].copy()

            if type(second_target) != type(None):
                y_test_2 = data.iloc[test_index][second_target].copy()

            #Embeding
            if embeding:
                tfidf_prep = tfidf_preparation
                tfidf_prep.fit(xtrain)
                xtrain = tfidf_prep.tf_idf_transform(xtrain)
                xtest = tfidf_prep.tf_idf_transform(xtest)
            else:
                xtrain.drop(list_of_vars_for_strat1, axis = 1, inplace = True)
                xtrain.drop(sort_by_var, axis = 1, inplace = True)
                xtrain.drop(target, axis = 1, inplace = True)
                if target1 in xtrain.columns:
                    xtrain.drop(target1, axis = 1, inplace = True)
                xtrain.drop('For_stratify', axis = 1, inplace = True)

                xtest.drop(list_of_vars_for_strat1, axis = 1, inplace = True)
                xtest.drop(sort_by_var, axis = 1, inplace = True)
                xtest.drop(target, axis = 1, inplace = True)
                if target1 in xtest.columns:
                    xtest.drop(target1, axis = 1, inplace = True)
                xtest.drop('For_stratify', axis = 1, inplace = True)

            if type(second_target) != type(None):
                y_train_2 = xtrain[second_target]
                y_test_2 = xtest[second_target]
                xtrain.drop(second_target, axis = 1, inplace = True)
                xtest.drop(second_target, axis = 1, inplace = True)

            test_с = xtest.columns
            test_ind = xtest.index
            xtrain, xtest, ytrain, ytest = data_preprocessing_meta(xtrain, ytrain, xtest, ytest, technical_values = [], 
                                                                   categorial_list = categorial_list, 
                                                                   yeo_johnson = preproc_config["yeo_johnson"], attribute_list = attribute_list, 
                                                                   var_col = var_col, scale = preproc_config["scale"], median = median,
                                                                   high_outlier = preproc_config["high_outlier"], 
                                                                   low_outlier = preproc_config["low_outlier"], 
                                                                   check_percentile = check_percentile, 
                                                                   cols_outlier = cols_outlier,
                                                                   cut_non_out_9999 = cut_non_out_9999)
            if task == 'binary':
                model, scores = train_and_receive_stats_binar(model, xtrain, ytrain, xtest, ytest, 
                                                              scores, second_target = second_target, 
                                                              y_train_2 = None, y_test_2 = None, draw = draw, 
                                                              draw_by_approval_rate = draw_by_approval_rate, catboost=catboost, categorial_list=categorial_list)
            elif task == 'numeric': 
                model, scores = train_and_receive_stats_numeric(model, xtrain, ytrain, xtest, ytest, 
                                                                scores, second_target = second_target, 
                                                                y_train_2 = None, y_test_2 = None)
            elif task == 'multiclass': 
                model, scores = train_and_receive_stats_multiclass(model, xtrain, ytrain, xtest, ytest, scores)
            first_score=first_score.append(pd.DataFrame(scores['Score'][num],test_index))
            num+=1
        first_score.sort_index(inplace=True)    
        scores = pd.DataFrame(scores)
        outputs.extend(scores.mean().tolist())
        outputs.extend(scores.std().tolist())
        
        score_cols = scores.columns.tolist()
        score_cols_std = [c+'_std' for c in score_cols]
        cols = list(params_dictionary.keys()) + score_cols + score_cols_std
        cols.remove("Score_std")
        cols.remove("Score")

        outputs = pd.DataFrame([outputs], columns=cols)
        meta_container = meta_container.append(outputs)
        if k/k_logs == int(k/k_logs) or k == 1:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            if need_business == True and task == 'binary':
                print(20*'-',tm, k, 20*'-', '\n',
                  'Параметры:', config, '\n', 'Среднее значение бизнес метрики =',
                  scores['b_best'].mean(), '\n', 'Среднее значение AUC =', scores['AUC'].mean(),'\n',
                  'Минимальное значение AUC =', scores['AUC'].min())
            elif need_business == False and task == 'binary':
                print(20*'-',tm, k, 20*'-', '\n',
                  'Параметры:', config, '\n', 'Среднее значение APS =',
                  scores['APS'].mean(), '\n', 'Среднее значение AUC =', scores['AUC'].mean(),'\n',
                  'Минимальное значение AUC =', scores['AUC'].min())
            elif task == 'numeric':
                print(20*'-',tm, k, 20*'-', '\n',
                  'Параметры:', config, '\n', 'Среднее значение R2 =',
                  scores['R2'].mean(),'\n',
                  'Минимальное значение R2 =', scores['R2'].min())
        if blend:
            from sklearn.linear_model import LogisticRegression
            #first_score=pd.Series([item for sublist in scores['Score'] for item in sublist])
            blend_data=pd.concat([first_score,second_score], axis=1)
            display(blend_data)
            once_rows = sum(Y)
            second_rows = (len(Y) - sum(Y))
            y_unique = sorted(Y.unique())
            all_rows = len(Y)
            once_rows_share = once_rows/all_rows
            second_rows_share = second_rows/all_rows
            second_rows_share/once_rows_share

            if once_rows < second_rows:
                w_b = once_rows/second_rows
            else:
                w_b = second_rows/once_rows
            class_weight_0=trial.suggest_categorical('class_weight_01', [0.01*w_b, 6*w_b])
            params = { 
                'C': trial.suggest_uniform("C1", 0.000001, 0.001),
                "penalty": trial.suggest_categorical("penalty1", ['l2']), 
                'solver': trial.suggest_categorical("solver1", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']), 
                #'class_weight': trial.suggest_categorical('class_weight_01', [0.01*w_b, 6*w_b]),
                'class_weight' : {0:class_weight_0, 1:1},
                'max_iter': trial.suggest_categorical("max_iter1" , [50, 100, 150]), 
                'random_state': 241
            }
    
            logreg = LogisticRegression(**params)
            print('gini without NLP = ', metrics.roc_auc_score(Y, second_score)*2-1, 
          'average precision score without NLP = ', metrics.average_precision_score(Y, second_score))
            blend_model = logreg.fit(blend_data, Y)
            y_train_pred = blend_model.predict_proba(blend_data)[:, 1]
            print('gini blend = ', metrics.roc_auc_score(Y, y_train_pred)*2-1, 
          'average precision score blend = ', metrics.average_precision_score(Y, y_train_pred))
            best_score = metrics.roc_auc_score(Y, y_train_pred)*2-1
        else:
            best_score=getattr(scores[optimize_metric], k_fold_method)()+a*(scores[optimize_metric].min()/scores[optimize_metric].max())
        k += 1
        if best_score<float("Inf") and best_score>-float("Inf"):
            if direction=='minimize':
                if best_score < BEST_LOSS:
                    BEST_LOSS = best_score
                    with open(LOSS_FILE, 'w') as loss_history:
                        loss_history.write(str(BEST_LOSS))
                    with open(PARAM_FILE, 'w') as param_dict:
                        param_dict.write(str(config))
                    if blend:
                        with open(PARAM_BLEND_FILE, 'w') as param_blend_dict:
                            param_blend_dict.write(str(params))
                    with open(PREPROC_FILE, 'w') as preproc_dict:
                        preproc_dict.write(str(preproc_config))    
                    if save_best_models == True:
                        joblib.dump(model, path+f'/model_{CNT}.dat')
            else:
                if best_score > BEST_LOSS:
                    BEST_LOSS = best_score
                    with open(LOSS_FILE, 'w') as loss_history:
                        loss_history.write(str(BEST_LOSS))
                    with open(PARAM_FILE, 'w') as param_dict:
                        param_dict.write(str(config))
                    if blend:
                        with open(PARAM_BLEND_FILE, 'w') as param_blend_dict:
                            param_blend_dict.write(str(params))
                    with open(PREPROC_FILE, 'w') as preproc_dict:
                        preproc_dict.write(str(preproc_config))
                    if save_best_models == True:
                        joblib.dump(model, path+f'/model_{CNT}.dat')

        CNT += 1
        return best_score
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)                 
    meta_container.reset_index(drop=True, inplace=True)
    if draw == True:
        fig.savefig('All_Max_Profit.png', dpi = 300)
    return meta_container



def get_stats_on_target(data, columns, time, target, model_name, technical_values=None, category_list = None, 
                       group = True, n_jobs=1, verbose=20, metric_treshold = 0.05,
                       category_target = True,pvalue = 0.05, prep_dict=None, period_verbose=False, check_metric=True, 
                       check_change_metric=True,check_change_betas=True,check_pval_target=True):
    
    """
    Посчитать стабильности переменных по периодам.
    Используются проверки:
        check_metric метрика < metric_treshold
        check_change_metric Знакопеременная метрика
        check_change_betas Знакопеременность betas
        check_pval_target Значимость связи переменной и таргета
    data - данные
	columns - фичи							  
    time - переменная, которая отвечает за время
    model_name - модель обучения
    technical_values - список технических переменных
    n_jobs - кол-во потоков
    pvalue - порог отсечения значимости
    group - агрегация данных
    prep_dict - словарь с параметрами data_preprocessing_test,
                при отсутствии словаря, запускается data_preprocessing_train
                при параметра в ловаре используется дефолтный из data_preprocessing_test
    category_list - список категориальных переменных
    category_target - категориален ли таргет
    period_verbose - вывод информации о том, как побиты периоды

    """
    from joblib import Parallel, delayed
    from datetime import datetime
    from sklearn.metrics import roc_auc_score
										
    unique = np.sort(data[time].unique())
    time_len = len(unique) - 1
    Statistics = []
    time_len = len(unique) - 1
    Statistics = []
    stable=[]
    k_test = 0
    
    #Logit param################
    max_iter = 300
    solver="liblinear"
    once_rows = sum(data[target])
    second_rows = (len(data[target]) - sum(data[target]))
												  
									  
										
    if once_rows < second_rows:
        w_b = once_rows/second_rows
    else:
        w_b = second_rows/once_rows
    class_weight = {0: 0.1*w_b, 1: 1}
    ###########################
    
    train_betas=[]
    betas=[]
    def col_preproc(col):
        nonlocal k_test
        nonlocal Statistics
        nonlocal stable
        nonlocal betas
        nonlocal train_betas
        
        k_test = k_test+1
        if k_test ==1:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Number of finished repetitions:', k_test , '| time: ' , tm)
            
        if k_test % 10 ==0:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Number of finished repetitions:', k_test , '| time: ' , tm)
        
        #На всем периоде
        x = data[col].to_frame()
        y = data[target]
        if(category_target):
            model_all=model_name(max_iter=max_iter, class_weight=class_weight,solver=solver)
        else:
            model_all=model_name()
            
        x_label=x
        model_all.fit(x_label, y)
        if(category_target):
            y_pred=model_all.predict_proba(x_label)
            metric=roc_auc_score(y, y_pred[:,1])*2-1
            coef = model_all.coef_[0][0]
        else:
            y_pred=model_all.predict(x_label)
            metric=metrics.r2_score(y, y_pred)
            coef = model_all.coef_[0]

        train_betas.append([col,coef]) 
            
        stabilty="Stable"
        stable_gini=""
        stable_gini_period=""
        stable_betas=""
        stable_pvalue=""
        all_coef=0
        #Попериодно
        for n_periods_in_group in range(2,len(unique)):
            for i in range(0,len(unique),n_periods_in_group):
                if (i == time_len) or (((n_periods_in_group-(len(unique)%n_periods_in_group))>1) and (len(unique)%n_periods_in_group)!=0):
                    break
                else:
                    if period_verbose:
                        print("n_periods_in_group=",n_periods_in_group,
                              data.loc[data[time].isin(unique[i:i+n_periods_in_group]), time].unique())
                    x = data.loc[data[time].isin(unique[i:i+n_periods_in_group]), col].to_frame()
                    y = data.loc[data[time].isin(unique[i:i+n_periods_in_group]), target]

                    x_label=x

                    x_label.name=f"{col}{unique[i]}_{n_periods_in_group}"
                    
                    if(category_target):
                        model_month=model_name(max_iter=max_iter, class_weight=class_weight,solver=solver)
                    else:
                        model_month=model_name()

                    model_month.fit(x_label, y)

                    if(category_target):
                        y_pred=model_month.predict_proba(x_label)
                        metric_month=roc_auc_score(y, y_pred[:,1])*2-1
                        month_coef = model_month.coef_[0][0]
                    else:
                        y_pred=model_month.predict(x_label)
                        metric_month=metrics.r2_score(y, y_pred)
                        month_coef = model_month.coef_[0]
                    betas.append([col ,month_coef]) 
                    
                    #Отрицательный GINI или R2 или метрики меняют свой знак
                    if(check_metric and (metric<metric_treshold)):
                        stabilty="Unstable"  
                        stable_gini="+"
                    if(check_change_metric and (metric>0 and metric_month<0)):
                        stabilty="Unstable"  
                        stable_gini_period="+"
                    #########################################################
                    if(month_coef>0 and all_coef>0):
                        None
                    elif(month_coef<0 and all_coef<0):
                        None
                    #Знак b меняется
                    elif(check_change_betas and month_coef<0 and all_coef>0):
                        stabilty="Unstable"
                        stable_betas="+"
                    elif(check_change_betas and month_coef>0 and all_coef<0):
                        stabilty="Unstable"
                        stable_betas="+"
                    ########################################################
                    else:
                        all_coef=month_coef
                    if (check_pval_target):
                        try:
                            stat_with_target=statistics_with_target(x_label, list(x_label.columns), y, category_list, False)
                        except:
                            stat_with_target=pd.DataFrame(columns = ['variable', 'stat pvalue', 'corr', 'corr name'])
                            stat_with_target=stat_with_target.append({'variable':None,'stat pvalue':0,
                                                                      'corr':0, 'corr name':None},ignore_index=True)
                        #Cвязь переменной и таргета через тест незначим
                        if(stat_with_target["stat pvalue"][0]>pvalue):
                            stabilty="Unstable"
                            stable_pvalue="+"
                    else:
                        stat_with_target=pd.DataFrame(columns = ['variable', 'stat pvalue', 'corr', 'corr name'])
                        stat_with_target=stat_with_target.append({'variable':None,'stat pvalue':0,
                                                                      'corr':0, 'corr name':None},ignore_index=True)
                    #######################################################
                    Statistics.append([col, f"{unique[i]}+{n_periods_in_group}", metric, metric_month, 
                                         month_coef, 
                                         stat_with_target["stat pvalue"][0],
                                         stable_gini, stable_gini_period, stable_betas, stable_pvalue])

        stable.append([col, stabilty])
        
    dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Start of preprocessing', dd1)
    if type(prep_dict) is not type(None):
        
        data=data_preprocessing_test(data[columns+technical_values], target, technical_values=technical_values,
                                     categorial_list = category_list, drop_technical = False,
                                    **prep_dict)
    else:
        data=data_preprocessing_train(data[columns+technical_values], target, technical_values=technical_values,
                                     categorial_list = category_list, drop_technical = False,
                                    yeo_johnson = False, attribute_list = None, var_col = None,
                                    low_outlier = 1, high_outlier = 99)[0]
        
    dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Finish of preprocessing', dd1)
    parallel = Parallel(n_jobs=n_jobs, require='sharedmem', verbose = verbose)
    dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Now', dd1)
    with parallel:
        par_res = parallel((delayed(col_preproc)(col) for col in data.columns))

    dd2 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Now', dd2)

    print('Time spent in hours:', (datetime.strptime(dd2,"%d.%m.%Y %H:%M:%S")-datetime.strptime(dd1,"%d.%m.%Y %H:%M:%S"))) 
    
    labels = ["columns","time", "metric","metric_period", "coef", "p_value_target",
              "Отрицательная метрика","Знакопеременность метрики","Знакопеременный коэффициент b", "Незначима связь переменной и таргета b"]
    columns_for_stat=["metric_period", "coef", "p_value_target"]
    definition = pd.DataFrame.from_records(Statistics, columns = labels)
    stable = pd.DataFrame.from_records(stable, columns = ["columns","stable"])
    betas = pd.DataFrame.from_records(betas, columns = ["columns","betas_period"])
    train_betas = pd.DataFrame.from_records(train_betas, columns = ["columns","betas_period_on_train"])
    if group == True:
        res_max = definition.groupby('columns').aggregate({'metric':"min",
                                                           'metric_period': 'min',
                                                           'coef': 'min',
                                                           'p_value_target': 'max',
                                                           "Отрицательная метрика": 'max',
                                                           "Знакопеременность метрики": 'max',
                                                           "Знакопеременный коэффициент b": 'max',
                                                           "Незначима связь переменной и таргета b": 'max'
                                                          })

        res_max = res_max.reset_index()

        return res_max, definition[columns_for_stat].groupby([definition["columns"]]).describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]), stable, betas, train_betas
    else:

        return definition, definition[columns_for_stat].groupby([definition["columns"]]).describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]), stable, betas, train_betas


def get_stats_on_target_by_month(data, columns, time, target, model_name,
                                 category_list = None, technical_values = None,
                                 n_jobs=1, verbose=20, category_target = True, metric_treshold = 0.05,
                                 pvalue = 0.05, prep_dict=None, check_metric=True, 
                                check_change_metric=True,check_change_betas=True,check_pval_target=True):
    

    """
    Посчитать стабильности переменных по месяцам. Данные не агрегируются.
    Используются проверки:
        check_metric метрика < metric_treshold
        check_change_metric Знакопеременная метрика
        check_change_betas Знакопеременность betas
        check_pval_target Значимость связи переменной и таргета):
    data - данные
	columns - фичи							  
    time - переменная, которая отвечает за время
    model_name - модель обучения
    technical_values - список технических переменных
    n_jobs - кол-во потоков
    pvalue - порог отсечения значимости
    prep_dict - словарь с параметрами data_preprocessing_test,
                при отсутствии словаря, запускается data_preprocessing_train
                при параметра в ловаре используется дефолтный из data_preprocessing_test
    category_list - список категориальных переменных
    category_target - категориален ли таргет

    """
    from joblib import Parallel, delayed
    from datetime import datetime
    from sklearn.metrics import roc_auc_score
										
    unique = np.sort(data[time].unique())
    time_len = len(unique) - 1
    Statistics = []
    stable=[]
    k_test = 0
    
    #Logit param################
    max_iter = 300
    solver="liblinear"
    once_rows = sum(data[target])
    second_rows = (len(data[target]) - sum(data[target]))
												  
									  
										
											
									 

    if once_rows < second_rows:
        w_b = once_rows/second_rows
    else:
        w_b = second_rows/once_rows
    class_weight = {0: 0.1*w_b, 1: 1}
    ###########################
    
    betas=[]
    train_betas=[]
    def col_preproc(col):

        nonlocal k_test
        nonlocal Statistics
        nonlocal stable
        nonlocal betas
        nonlocal train_betas
        k_test = k_test+1
        if k_test ==1:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Number of finished repetitions:', k_test , '| time: ' , tm)
            
        if k_test % 10 ==0:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Number of finished repetitions:', k_test , '| time: ' , tm)
        
        
        #На всем периоде
        x = data[col].to_frame()
        y = data[target]
        if(category_target):
            model_all=model_name(max_iter=max_iter, class_weight=class_weight,solver=solver)
        else:
            model_all=model_name()

        x_label=x
        model_all.fit(x_label, y)
        if(category_target):
            y_pred=model_all.predict_proba(x_label)
            metric=roc_auc_score(y, y_pred[:,1])*2-1
            coef = model_all.coef_[0][0]
        else:
            y_pred=model_all.predict(x_label)
            metric=metrics.r2_score(y, y_pred)
            coef = model_all.coef_[0]

        train_betas.append([col, coef])
        stabilty="Stable"
        stable_gini=""
        stable_gini_period=""
        stable_betas=""
        stable_pvalue=""
        all_coef=0
        #Попериодно
        for i in range(len(unique)):
            
            if i == time_len:
                break
            else:
                x = data.loc[data[time] == unique[i], col].to_frame()
                
                y = data.loc[data[time] == unique[i], target]
                x_label=x
                x_label.name=f"{col}{unique[i]}"
                if(category_target):
                    model_month=model_name(max_iter=max_iter, class_weight=class_weight,solver=solver)
                else:
                    model_month=model_name()
                model_month.fit(x_label, y)
                
                if(category_target):
                    y_pred=model_month.predict_proba(x_label)
                    metric_month=roc_auc_score(y, y_pred[:,1])*2-1
                    month_coef = model_month.coef_[0][0]
                else:
                    y_pred=model_month.predict(x_label)
                    metric_month=metrics.r2_score(y, y_pred)
                    month_coef = model_month.coef_[0]
                    
                betas.append([col,month_coef]) 
                #Отрицательный GINI или R2 или метрики меняют свой знак
                if(check_metric and metric<metric_treshold):
                    stabilty="Unstable"  
                    stable_gini="+"
                if(check_change_metric and metric>0 and metric_month<0):
                    stabilty="Unstable"  
                    stable_gini_period="+"
                #########################################################
                if(month_coef>0 and all_coef>0):
                    None
                elif(month_coef<0 and all_coef<0):
                    None
                #Знак b меняется
                elif(check_change_betas and month_coef<0 and all_coef>0):
                    stabilty="Unstable"
                    stable_betas="+"
                elif(check_change_betas and month_coef>0 and all_coef<0):
                    stabilty="Unstable"
                    stable_betas="+"
                ########################################################
                else:
                    all_coef=month_coef
               
                if (check_pval_target):
                    try:
                        stat_with_target=statistics_with_target(x_label, list(x_label.columns), y, category_list, False)
                    except:
                        stat_with_target=pd.DataFrame(columns = ['variable', 'stat pvalue', 'corr', 'corr name'])
                        stat_with_target=stat_with_target.append({'variable':None,'stat pvalue':0,
                                                                  'corr':0, 'corr name':None},ignore_index=True)
                    #Cвязь переменной и таргета через тест незначим
                    if(stat_with_target["stat pvalue"][0]>pvalue):
                        stabilty="Unstable"
                        stable_pvalue="+"
                else:
                    stat_with_target=pd.DataFrame(columns = ['variable', 'stat pvalue', 'corr', 'corr name'])
                    stat_with_target=stat_with_target.append({'variable':None,'stat pvalue':0,
                                                                      'corr':0, 'corr name':None},ignore_index=True)
                #######################################################
                Statistics.append([col, unique[i], metric, metric_month, 
                                     month_coef, 
                                     stat_with_target["stat pvalue"][0],
                                    stable_gini, stable_gini_period, stable_betas, stable_pvalue])

        stable.append([col, stabilty])
        
    dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Start of preprocessing', dd1)
    if type(prep_dict) is not type(None):
        
        data=data_preprocessing_test(data[columns+technical_values], target, technical_values=technical_values,
                                     categorial_list = category_list, drop_technical = False,
                                    **prep_dict)
    else:
        data=data_preprocessing_train(data[columns+technical_values], target, technical_values=technical_values,
                                     categorial_list = category_list, drop_technical = False,
                                    yeo_johnson = False, attribute_list = None, var_col = None,
                                    low_outlier = 1, high_outlier = 99)[0]
    dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Finish of preprocessing', dd1)
    parallel = Parallel(n_jobs=n_jobs, require='sharedmem', verbose = verbose)
    dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Now', dd1)
    with parallel:
        par_res = parallel((delayed(col_preproc)(col) for col in data.columns))

    dd2 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Now', dd2)

    print('Time spent in hours:', (datetime.strptime(dd2,"%d.%m.%Y %H:%M:%S")-datetime.strptime(dd1,"%d.%m.%Y %H:%M:%S"))) 
    
    labels = ["columns","time", "metric","metric_month", "coef", "p_value_target",
              "Отрицательная метрика","Знакопеременность метрики","Знакопеременный коэффициент b", "Незначима связь переменной и таргета b"]
    columns_for_stat=["metric_month", "coef", "p_value_target"]
    definition = pd.DataFrame.from_records(Statistics, columns = labels)
    stable = pd.DataFrame.from_records(stable, columns = ["columns","stable"])
    betas = pd.DataFrame.from_records(betas, columns = ["columns","betas_month"])
    train_betas = pd.DataFrame.from_records(train_betas, columns = ["columns","betas_month_on_train"])
    
    return definition, definition[columns_for_stat].groupby([definition["columns"]]).describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]), stable, betas, train_betas


def corr_by_month(data, columns, time, n_jobs=1, verbose=20, treshold=0.7):
    
    """
    Посчитать корреляцию переменных
    data - данные
    columns - переменные
    time - переменная, которая отвечает за время
    n_job - кол-во потоков
    treshold - порог отсечения
    """
    from datetime import datetime

    from joblib import Parallel, delayed
    unique = np.sort(data[time].unique())
    time_len = len(unique) - 1
    Statistics = []
    k_test=0
    def col_preproc(i):
        nonlocal k_test
        if i == time_len:
            None
        else:
            k_test = k_test+1
            if k_test ==1:
                tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
                print ('Number of finished repetitions:', k_test , '| time: ' , tm)

            if k_test % 10 ==0:
                tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
                print ('Number of finished repetitions:', k_test , '| time: ' , tm)

                    
            x = data.loc[data[time] == unique[i], data.columns != target]
            y = data.loc[data[time] == unique[i], target]

            result_tab, non_doubles, col_doubles, double_dic=find_doubles_corr(
                x, x.columns, x.corr(method="pearson"), lvl = treshold)
            Statistics.append([unique[i],result_tab, non_doubles, col_doubles, double_dic])

    parallel = Parallel(n_jobs=n_jobs, require='sharedmem', verbose = verbose)
    dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Now', dd1)
    with parallel:
        par_res = parallel((delayed(col_preproc)(o) for o in range(len(unique))))

    dd2 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Now', dd2)

    print('Time spent in hours:', (datetime.strptime(dd2,"%d.%m.%Y %H:%M:%S")-datetime.strptime(dd1,"%d.%m.%Y %H:%M:%S"))) 
    
    labels = ["time", "пара дубль - оставшаяся переменная","список оставшихся переменных", "список дублированных переменных",
              "словарь соответствия дубль - оставшаяся переменная"]

    definition = pd.DataFrame.from_records(Statistics, columns = labels)
    first_tuple_list = [lst for sublst in definition["список дублированных переменных"] for lst in sublst]
    feature_to_drop = set(first_tuple_list)
    return list(feature_to_drop), definition


def p_value_diff_gini(data, columns, time, target, model_name, category_list = None, group = True,
                      n_jobs=1, verbose=20, category_target = True, period_verbose=False):
    
    """
    Посчитать занчимость по методике валидаторов
    data - данные
    columns - переменные
    time - переменная, которая отвечает за время
    category_list - список категориальных переменных
    period_verbose - вывод информации о текущей разбивки данных
    """
    from datetime import datetime

    from joblib import Parallel, delayed
    from scipy.stats import norm
    def diff(sample, x, k, Gini_valid, nbuckets=20):
        conc=sample[[x,k]].copy()
        conc.sort_values(by=[k,x], ascending=[False, False], inplace=True)
        var1=np.asarray(conc[x])
        var2=np.asarray(conc[k])
        var3=np.asarray(conc[k].unique())
        n=len(var3)
        B = var1.sum()
        G = var1.size - B  
        Gini = 0
        count = 0
        count1 = 0  
        Sum1=0
        Sum2=0

        if n<=nbuckets:
            for i in range (0,n):
                count+=np.count_nonzero(var2 == var3[i])
                count1+=np.count_nonzero((var2 == var3[i-1])&(i>0))
                b=var1[:count].sum()
                b1=var1[:count1].sum()
                g=var1[:count].size-b
                g1=var1[:count1].size-b1
                Sum1+=((b/B+b1/B-math.fabs(Gini_valid)-1)**2)*(g/G-g1/G)
                Sum2+=((g/G+g1/G-math.fabs(Gini_valid)-1)**2)*(b/B-b1/B)
            se=(G/((G-1)*B)*Sum1+B/((B-1)*G)*Sum2)**0.5

        else:

            smpl=percnt(sample, k, nbuckets)

            smpl=percnt(sample, k, nbuckets)
            var4=np.asarray(smpl['firstquantile'])
            n1=max(smpl['firstquantile'].unique())

            for i in range (0,n1+1):
                count+=np.count_nonzero(var4 == n1-i)
                count1+=np.count_nonzero(var4 == n1+1-i)
                b=var1[:count].sum()
                b1=var1[:count1].sum()
                g=var1[:count].size-b
                g1=var1[:count1].size-b1
                Sum1+=((b/B+b1/B-math.fabs(Gini_valid)-1)**2)*(g/G-g1/G)
                Sum2+=((g/G+g1/G-math.fabs(Gini_valid)-1)**2)*(b/B-b1/B)
            se=(G/((G-1)*B)*Sum1+B/((B-1)*G)*Sum2)**0.5

        return se
    
    unique = np.sort(data[time].unique())
    time_len = len(unique) - 1
    Statistics = []
    stable=[]
    k_test = 0
    max_iter = 300
    solver="liblinear"
    once_rows = sum(data_check[target])
    second_rows = (len(data_check[target]) - sum(data_check[target]))
    y_unique = sorted(data_check[target].unique())
    all_rows = len(data_check[target])

    once_rows_share = once_rows/all_rows
    second_rows_share = second_rows/all_rows
    second_rows_share/once_rows_share


    w_b = once_rows/second_rows
    class_weight = {0: 0.1*w_b, 1: 1}
    betas=[]
    result = pd.DataFrame()
    def col_preproc(o):
#     for o in range(len(columns)):
        nonlocal k_test
        nonlocal Statistics
        nonlocal stable
        nonlocal betas
        nonlocal result
        k_test = k_test+1
        if k_test ==1:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Number of finished repetitions:', k_test , '| time: ' , tm)
            
        if k_test % 10 ==0:
            tm = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
            print ('Number of finished repetitions:', k_test , '| time: ' , tm)
            
        #Попериодно
        for n_periods_in_group in range(1,len(unique)):
            
            if(columns[o]==target):
                break
            
            x = data.loc[data[time].isin(unique[:n_periods_in_group]), columns[o]]
            y = data.loc[data[time].isin(unique[:n_periods_in_group]), target]
            if period_verbose:
                print("n_periods_train=",n_periods_in_group,
                              data.loc[data[time].isin(unique[:n_periods_in_group]), time].unique())
                print("n_periods_val=",len(unique)-n_periods_in_group,
                              data.loc[data[time].isin(unique[n_periods_in_group:]), time].unique())
            x_val = data.loc[data[time].isin(unique[n_periods_in_group:]), columns[o]]
            y_val = data.loc[data[time].isin(unique[n_periods_in_group:]), target]

            if(columns[o] not in category_list):
                x=data_preprocessing_train(x.to_frame(), target, technical_values=["None",], categorial_list = category_list, drop_technical = False,
                                            yeo_johnson = False, attribute_list = None, var_col = None,
                                           low_outlier = 1, high_outlier = 99)
                x_val=data_preprocessing_train(x_val.to_frame(), target, technical_values=["None",], categorial_list = category_list, drop_technical = False,
                                            yeo_johnson = False, attribute_list = None, var_col = None,
                                            low_outlier = 1, high_outlier = 99)
                x_label=x[0]
                x_val_label=x_val[0]
            else:
                x_label=x.fillna(0).to_frame()
                x_val_label=x_val.fillna(0).to_frame()
                                               
            x_label.name=f"{columns[o]}{unique[n_periods_in_group]}_train{n_periods_in_group}"
            x_val_label.name=f"{columns[o]}_val{n_periods_in_group}"
                                               
            if(category_target):
                model_month=model_name(max_iter=max_iter, class_weight=class_weight,solver=solver)
            else:
                model_month=model_name()
            model_month.fit(x_label, y)

            if(category_target):
                y_pred=model_month.predict_proba(x_label)
                Gini_train=roc_auc_score(y, y_pred[:,1])*2-1
                month_coef = model_month.coef_[0][0]
                y_val_pred=model_month.predict_proba(x_val_label)
                Gini_valid=roc_auc_score(y_val, y_val_pred[:,1])*2-1
            else:
                y_pred=model_month.predict(x_label)
                Gini_train=metrics.r2_score(y, y_pred)
                month_coef = model_month.coef_[0]
                y_val_pred=model_month.predict_proba(x_val_label)
                Gini_valid=metrics.r2_score(y_val, y_val_pred)
            se=diff(pd.concat([x_val_label,y_val],axis=1), target, columns[o], Gini_valid)
            if ((Gini_train >= 0) and (Gini_valid >= 0)) or ((Gini_train < 0) and (Gini_valid < 0)):
                p = 1-norm.cdf((0.9*math.fabs(Gini_train) - math.fabs(Gini_valid))/se)
            else:
                p = 1-norm.cdf((0.9*math.fabs(Gini_train) + math.fabs(Gini_valid))/se)

            giniresult=pd.DataFrame({'Фактор': [x_val_label.name], 'Gini_train': [Gini_train], 'Gini_valid': [Gini_valid], 'P-value': [p]})
            result = result.append(giniresult, ignore_index = True)
            
    parallel = Parallel(n_jobs=n_jobs, require='sharedmem', verbose = verbose)
    dd1 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Now', dd1)
    with parallel:
        par_res = parallel((delayed(col_preproc)(o) for o in range(len(columns))))

    dd2 = datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    print ('Now', dd2)

    print('Time spent in hours:', (datetime.strptime(dd2,"%d.%m.%Y %H:%M:%S")-datetime.strptime(dd1,"%d.%m.%Y %H:%M:%S"))) 
    return(result)



def feature_importance(X_1, y_1, X_2, y_2, w_b, task="binary", boosting=False, treshold=0.005, n_jobs=1, selection="SHAP", selection_politic="Hard", bs_plot=True):
    
    """
    Функция проводит отбор на деревьях. Параметры:

    - X, Y - матрица данных и таргет. Матрица X не должна быть предобработана, так как предобработка делается внутри функции! 

    - task - binary or numeric

    - boosting - Использовать LightGBM
    
    - treshold - порог потери качества
    
    - selection - SHAP или Boruta_SHAP
    
    - selection_politic - Если Hard, то отбрасываются фичи хоть раз папавшие на выброс. Light оставляет хоть раз оставленные фичи

    - bs_plot - Выводить ли график для отбора Boruta_SHAP
    """
    from datetime import datetime
    
    def tree_study(X_1,y_1,X_2,y_2,w_b,task,boosting,treshold):
        if boosting:
            param_dict = {
                            'boosting_type': 'gbdt',
                            'max_depth': 3,
                            'num_leaves': 31,
                            'learning_rate': 0.1,
                            'feature_fraction': 0.9,
                            'bagging_fraction': 0.9,
                            'bagging_freq': 5,
                            'verbose': 0,
                            'n_estimators': 100,
                            'class_weight': {0: w_b, 1: 1},
                            'random_state': 241, 
                            'n_jobs': 5}

            if task == 'binary':
                from lightgbm import LGBMClassifier
                rf = LGBMClassifier(**param_dict)
            elif task == 'numeric':
                from lightgbm import LGBMRegressor
                rf = LGBMRegressor(**param_dict)  # Исправлено
        else:
            
            param_dict = {'max_depth': 4, 'class_weight': {0: w_b, 1: 1}, 'n_estimators': X_1.shape[1], 
                          'max_features': 0.15, 'min_samples_leaf': 5, 
                          'min_samples_split': 10, 'random_state': 241}
            if task == 'binary':
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(**param_dict)
            elif task == 'numeric':
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(**param_dict)  # Исправлено
        if selection=="Boruta_SHAP":
            from BorutaShap import BorutaShap
            Feature_Selector = BorutaShap(model=rf,importance_measure='shap', classification=True, percentile=80)

            Feature_Selector.fit(X=X_1, y=y_1, n_trials=50, random_state=0)
            if bs_plot == True:
                Feature_Selector.plot(which_features='all', figsize=(16,12))
            final_features_global.append(Feature_Selector.Subset().columns)
        elif selection=="SHAP":  
            import shap
            rf.fit(X_1, y_1)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                explainer = shap.TreeExplainer(rf)
                shap_values.append(explainer.shap_values(X_2))
            if task == 'binary':
                from sklearn.metrics import roc_auc_score
                y_pred_test = rf.predict_proba(X_2)[:,1]
                auc.append(roc_auc_score(y_2, y_pred_test)*2-1)
            elif task == 'numeric':
                y_pred_test = rf.predict(X_2)
                auc.append(metrics.r2_score(y_2, y_pred_test))

    imp_test = X_1.columns
    
    X1s=[X_1[imp_test],X_2[imp_test]]
    y1s=[y_1,y_2]
    X2s=[X_2[imp_test],X_1[imp_test]]
    y2s=[y_2,y_1]

    shap_values = []
    auc = []
    final_features_global=[]
    from joblib import Parallel, delayed
    parallel = Parallel(n_jobs=n_jobs, require='sharedmem')
    with parallel:
        par_res = parallel((delayed(tree_study)(X1,y1,X2,y2,w_b,task,boosting,treshold) for X1,y1,X2,y2 in zip(X1s,y1s,X2s,y2s)))
    if selection=="Boruta":
        raise ValueError('Метод отбора Boruta был удален, используйте методы SHAP или Boruta_SHAP.')
    if selection=="Boruta_SHAP":
        if(selection_politic=="Light"):
            print("Отбор закончен", datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"))
            return list(set(final_features_global[0]+final_features_global[1]))
        elif(selection_politic=="Hard"):
            print("Отбор закончен", datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"))
            return list(set(final_features_global[0]) & set(final_features_global[1]))
        else:
            raise ValueError("Use Light or Hard politic of selection")
    
    elif (selection=="SHAP"):

        metric1=(auc[0]+auc[1])/2
        metric2=metric1

        for step in range(len(imp_test)):

            shap_sum = np.abs(shap_values[0][0]).mean(axis=0)
            importance_df = pd.DataFrame([X_1[imp_test].columns.tolist(), shap_sum.tolist()]).T
            importance_df.columns = ['column_name', 'shap_importance1']

            shap_sum2 = np.abs(shap_values[1][0]).mean(axis=0)
            importance_df2 = pd.DataFrame([X_2[imp_test].columns.tolist(), shap_sum2.tolist()]).T
            importance_df2.columns = ['column_name', 'shap_importance2']
            importance_df["shap_importance2"] = importance_df2["shap_importance2"]

            importance_df["shap_importance"] = (importance_df['shap_importance1']+importance_df['shap_importance2'])/2
            importance_df = importance_df.sort_values('shap_importance', ascending=False)

            if ((importance_df.iloc[-1]["shap_importance"] != 0) | (step>0)):
                print('Drop last (' +  importance_df.iloc[-1, importance_df.columns.get_loc('column_name')] + 
                      ', shap=' + str(round(importance_df.iloc[-1, importance_df.columns.get_loc('shap_importance')], 5)) +')')
                importance_df_copy=importance_df.iloc[:-1].copy()

            else:
                print('Drop zero (' +  ', '.join(importance_df.loc[importance_df.shap_importance == 0, 'column_name'].tolist()) + 
                      ', shap=' + ', '.join(importance_df.loc[importance_df.shap_importance == 0,'shap_importance'].astype('str').tolist()) +')')

                importance_df_copy=importance_df[importance_df["shap_importance"]!=0].copy()

            imp_test=importance_df_copy["column_name"].to_list()
            print(f"осталось {len(imp_test)} фичей")
            X1s=[X_1[imp_test],X_2[imp_test]]
            y1s=[y_1,y_2]
            X2s=[X_2[imp_test],X_1[imp_test]]
            y2s=[y_2,y_1]
            shap_values = []
            auc = []
            with parallel:
                par_res = parallel((delayed(tree_study)(X1,y1,X2,y2,w_b,task,boosting,treshold) for X1,y1,X2,y2 in zip(X1s,y1s,X2s,y2s)))

            metric2=(auc[1]+auc[0])/2
            print(f'Изменение Gini: {"{:.2%}".format(round(metric1-metric2, 5))}', f'Предел: {"{:.2%}".format(treshold*metric1)}', '\n')
            if (metric1-metric2)<=(treshold*metric1):
                importance_df = importance_df_copy.copy()
                continue
            else:
                break

        print("Отбор закончен", datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"))
        return importance_df['column_name'].to_list()
        
        
def one_factor_model(data, target, columns, model_name, technical_values=None, category_list=None, category_target=True):
    """
    Функция расчета однофакторных моделей.
    """
    from sklearn.metrics import roc_auc_score
    factors = pd.DataFrame(columns = ['variable', 'metric', 'coef'])
                        
    
    max_iter = 300
    solver="liblinear"
    once_rows = sum(data[target])
    second_rows = (len(data[target]) - sum(data[target]))


    if once_rows < second_rows:
        w_b = once_rows/second_rows
    else:
        w_b = second_rows/once_rows
    class_weight = {0: 0.1*w_b, 1: 1}

    data=data_preprocessing_train(data[columns+technical_values], target, technical_values=technical_values,
                                     categorial_list = category_list, drop_technical = False,
                                    yeo_johnson = False, attribute_list = None, var_col = None,
                                    low_outlier = 1, high_outlier = 99)[0]
    for i in [i for i in data.columns if i not in technical_values]:
        if(category_target):
            model=model_name(max_iter=max_iter, class_weight=class_weight,solver=solver)
        else:
            model=model_name()
        x_label = data[i].to_frame()
                
        y = data[target]   
        model.fit(x_label, y)

        if(category_target):
            y_pred=model.predict_proba(x_label)
            metric=roc_auc_score(y, y_pred[:,1])*2-1
            coef = model.coef_[0][0]
        else:
            y_pred=model_month.predict(x_label)
            metric=metrics.r2_score(y, y_pred)
            coef = model.coef_[0]
        factors=factors.append({'variable':i, 'metric':metric, 'coef':coef},ignore_index=True)
    return factors



class GiniDegrad:

    def __init__(self,
                 train,
                 test,
                 oot,
                 model,
                 pass_model,
                 it,
                 group_by_var_strat,
                 old_model=False, #Для старых карт True (2019 год)
                 spline_split_points=None,
                 splines_depth = 2,
                 mpp=None,
                 fit_model=False, #Убираем
                 n_reps=0, # Если не 0, то после семплирования запускается бутстреп
                 task='binary',
                 date_column='date_report',
                 gini_degradation_path='Gini_degradation',
                 prep_dict=None,
                 random_state=241,
                 n_jobs=3,
                 verbose=30,
                 reject_inf = False):

        self.train = train
        self.test = test
        self.oot = oot

        self.train.columns = [i.upper() for i in self.train.columns]
        self.test.columns = [i.upper() for i in self.test.columns]
        self.oot.columns = [i.upper() for i in self.oot.columns]

        self.model = model
        self.n_reps = n_reps
        self.task = task

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pass_model = pass_model
        self.verbose = verbose

        self.it = it.upper()
        self.group_by_var_strat = group_by_var_strat.upper()
        self.date_column = date_column.upper()

        self.fit_model = fit_model
        self.old_model = old_model
        self.gini_degradation_path = gini_degradation_path
        self.prep_dict = prep_dict
        self.spline_split_points = spline_split_points
        self.splines_depth = splines_depth
        self.reject_inf = reject_inf
#         self.itp = 'IT_PREDICTION'
#         self.itpp = 'IT_PREDICTION_PROBA'

        if self.prep_dict is not None:
            self.prep_dict['categorical_features_df']['feature'] = (
            self.prep_dict['categorical_features_df']['feature'].str.upper())

            self.prep_dict['selected_features_df'][['new variable',
                                                    'genuine variable']] = (
            self.prep_dict['selected_features_df'][['new variable',
                                                    'genuine variable']
                                               ].apply(lambda x: x.str.upper()))
            self.category_list = prep_dict['selected_features_df'].loc[prep_dict['selected_features_df']['rule'] != 'Missing',
                                                                         'new variable'].to_list()
            self.prep_dict['technical_vars'] = list(
                            map(str.upper, self.prep_dict['technical_vars']))
            self.technical_vars = self.prep_dict['technical_vars']

        if mpp is not None:
            self.mpp = mpp
        else:
            clf = MPP(model=self.model,
                      it=self.it,
                      date_column=self.date_column,
                      mpp_folder_path=self.gini_degradation_path,
                      use_custom_treshold='best',
                      spline_split_points=self.spline_split_points,
                      prep_dict=self.prep_dict,
                      old_model=self.old_model, plot=False)

            clf.fit(test_and_oot_data={'Test': self.test, 'Oot': self.oot},
                          train_data=self.train,
                          preprocessed=False,
                          group_by_var_strat=self.group_by_var_strat,
                          list_of_vars_for_strat=[self.date_column],
                          save_or_show_plots='save',
                          oot_n_months=3,
                          mpp_metrics=True,
                          params_dictionary=self.model.get_params(),
                          pass_model=self.pass_model,
                          n_folds=5,
                          params_lgbm={'n_estimators': 120})
            self.mpp = clf

    def attributes_list_new(self, data, columns, percentiles_list=None):
        """
        Рассчитывает статистики по выборке. Подаются данные и список колонок

        Статистики:

        Тип переменной - type_val
        Количество значений - count_val
        Количество уникальных значений - count_dist
        Количество пропусков - count_miss
        Минимальное значение - min_val
        Максимальное значение - max_val
        Медиана - val_mediana
        Мода - moda_val
        Количество наблюдений, равных моде - count_value_moda
        Стандартное отклонение - stand_d_val
        1й перцентиль - percentile_1
        2й перцентиль - percentile_2
        5й перцентиль - percentile_5
        95й перцентиль - percentile_95
        98й перцентиль - percentile_98
        99й перцентиль - percentile_99
        """

        if percentiles_list is None:
            percentiles = 0.01 * np.array([1, 2, 5, 95, 98, 99])
        else:
            percentiles = 0.01 * np.array(percentiles_list)

        data_lenght = len(data)
        data_width = data.shape[1]
        D = data[columns].describe(include = 'all', percentiles=percentiles).T
        D.rename(columns={'50%':'val_mediana', 
                          'min':'min_val', 
                          'max':'max_val', 
                          'count':'count_val', 
                          'std':'stand_d_val'}, inplace = True)

        attribute_list = pd.DataFrame()
        attribute_list['attribute'] = columns
        cm = data.isna().sum(axis=0)

        moda_val = []
        count_dist = []
        count_value_moda = [] #кол-во элементов со значением моды

        iterable = data.iteritems()

        for feature_name, feature_column in iterable:
            vc = feature_column.value_counts()
            count_dist.append(len(vc))
            if feature_column.dtype != 'O':
                if cm[feature_name] == data_lenght:
                    k11 = np.nan
                    k12 = np.nan
                else:
                    k11 = vc.index[0]
                    k12 = vc.iloc[0]
            else:
                k11 = -1000
                k12 = -1000

            moda_val.append(k11)
            count_value_moda.append(k12) 

        attribute_list['moda_val'] = moda_val
        attribute_list['count_miss'] = cm.values
        attribute_list['type_val'] = data.dtypes.values
        attribute_list['count_dist'] = count_dist
        attribute_list['count_value_moda'] = count_value_moda

        return pd.merge(attribute_list, D, left_on = 'attribute', right_index = True)

    def receive_train_period(self):
    
        train_cols = sorted(self.train[self.date_column].unique())
#         test_cols = sorted(self.test[self.date_column].unique())
#         oot_cols = sorted(self.oot[self.date_column].unique())

        train_periods = sorted(list(set(train_cols)))
#         oot_periods = sorted(list(set(train_cols + test_cols + oot_cols)))[-1]

        return train_periods

    def sample_shuffle_periods(self, periods, num_rep):
        
        length = np.arange(len(periods))

        np.random.seed(self.random_state+num_rep)
        init_period = np.random.choice(length)

        np.random.seed(self.random_state+num_rep+1)
        period_len = np.random.choice(length)+1

        train_period = periods[init_period:init_period+period_len]

        return train_period
    
    def sample_all_periods(self, periods):
    
        train_periods = []

        length = np.arange(len(periods))
        all_length = len(periods)

        for init in length:
            for l in length: 
                if init+l+1 > all_length:
                    break
                train_periods.append(periods[init:init+l+1])

    #     train_periods = sorted(list(set(train_periods)))

        return train_periods
    
    def receive_new_all_train_test_oot(self, period):
    
        max_m = max(period)
    
        train = self.train.copy()
        test = self.test.copy()
    
        train_new = train.loc[train[self.date_column].isin(period)].copy()
        test_new = test.loc[test[self.date_column].isin(period)].copy()

        oot_train_new = train.loc[train[self.date_column] > max_m].copy()
        oot_test_new = test.loc[test[self.date_column] > max_m].copy()

        oot_new = self.oot.copy()
        oot_new = oot_new.append(oot_train_new)
        oot_new = oot_new.append(oot_test_new)

        train_new.reset_index(inplace = True, drop = True)
        test_new.reset_index(inplace = True, drop = True)
        oot_new.reset_index(inplace = True, drop = True)

        return train_new, test_new, oot_new
    
    def receive_new_shuffle_train_test_oot(self, period, num_rep):
    
        max_m = max(period)

        train_new = self.train.loc[self.train[self.date_column].isin(period)].copy()
        test_new = self.test.loc[self.test[self.date_column].isin(period)].copy()

        oot_train_new = self.train.loc[self.train[self.date_column] > max_m].copy()
        oot_test_new = self.test.loc[self.test[self.date_column] > max_m].copy()

        oot_new = self.oot.copy()
        oot_new = oot_new.append(oot_train_new)
        oot_new = oot_new.append(oot_test_new)

        train_new = resample(train_new, random_state = self.random_state+num_rep)
        test_new = resample(test_new, random_state = self.random_state+num_rep)
        oot_new = resample(oot_new, random_state = self.random_state+num_rep)

        train_new.reset_index(inplace = True, drop = True)
        test_new.reset_index(inplace = True, drop = True)
        oot_new.reset_index(inplace = True, drop = True)

        return train_new, test_new, oot_new
    
    def turn_variables_with_values(self,
                                     data, 
                                     rules_list):

        """
        Функция оставляет в выборке только интересующие нас переменные. 

        Параметры:
        ----------

        data : pd.DataFrame
            Передаваемые данные.

        rules_list : DataFrame с информацией о признаках, которые оставляем. 

            Структура датафрейма rules_list:

            new variable : 
                Новая переменная, которая попадет в модель.
            genuine variable : 
                Старая переменная, которую надо изменить.
            rule : 
                Правило, по которому производится изменение (постфикс).
            values : 
                значения, которые принимала истинная переменная (для бинов на 
                категории или Others)

        !ВАЖНО!
        Перед подачей pданных следует убедиться, что в поле "values" находятся 
        значения правильного типа! При загрузке данных rules_list из csv 
        все значения values могут превратиться в текст, что нарушит работу алгоритма!

        Возвращает:
        ---------- 
        Измененные данные.
        """
        data = data.copy()

        new_variables = rules_list['new variable'].to_list()

        for i in rules_list.index:
            if rules_list.loc[i, 'rule'] == '_bin':
                data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(pd.isnull(
                    data[rules_list.loc[i, 'genuine variable']]) == True , 0, 1), index=data.index)

            elif rules_list.loc[i, 'rule'] == 'Missing':
                continue
            elif rules_list.loc[i, 'rule'] == '_nan':
                data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(pd.isnull(
                    data[rules_list.loc[i, 'genuine variable']]) == True , 1, 0), index=data.index)

            elif rules_list.loc[i, 'rule'] == '_Other':
                found_values = rules_list.loc[i, 'values']
                if type(found_values) == type('One'):    
                    try:
                        found_val = eval(found_values)
                    except SyntaxError:
                        found_val = found_values
                    except NameError:
                        found_val = found_values
                else:
                    found_val = found_values

                data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(
                    data[rules_list.loc[i, 'genuine variable']].isin(found_val), 1, 0), index=data.index)

            else:
                number1 = rules_list.loc[i, 'values']
                if type(number1) == type('One'):            
                    try:
                        number = eval(number1)
                    except SyntaxError:
                        number = number1
                    except NameError:
                        number = number1
                else:
                    number = number1

                if type(number) != list:
                    data[rules_list.loc[i, 'new variable']] = pd.Series(np.where((
                        data[rules_list.loc[i, 'genuine variable']] == number), 1, 0), index=data.index)
                else:
                    data[rules_list.loc[i, 'new variable']] = pd.Series(np.where(
                        data[rules_list.loc[i, 'genuine variable']].isin(number), 1, 0), index=data.index)

        if self.it in new_variables:
            data = data[[*new_variables]]
        else:
            data = data[[*new_variables, self.it]]

        return data   
   
    def by_month_gini(self, X, pred_name):
    
        """
        Бьет выборку по месяцам и считает помесячные значение Gini.
        model - модель
        time_period - поле, в котором находятся временные периоды
        X, y - данные (полностью подготовленные) и таргет
        good_bad_dict - словарь вида {'good': 1, 'bad': 0}, нужен для определения, на основании чего считать good_rate и bad_rate статистики

        """
        time_period = self.date_column
        good_bad_dict = self.prep_dict['good_bad_dict']
        
        X.reset_index(drop=True, inplace = True)

        time_periods = sorted(X[time_period].unique())

        scores = []
        
        for i in time_periods:
            X_month = X[X[time_period] == i].copy()
            y_month = X_month[self.it]
            prediction = X_month[pred_name]
            
            if y_month.sum()>0:
                gini = metrics.roc_auc_score(y_month, prediction)*2-1
            else:
                gini = np.nan
            bad_rate = len(y_month[y_month == good_bad_dict['bad']])/len(y_month)
            good_rate = len(y_month[y_month == good_bad_dict['good']])/len(y_month)
            scores.append([i, len(y_month), bad_rate, good_rate, gini])

        col_names = [self.date_column, 'number', 'bad_rate', 'good_rate', 'score']
        scores = pd.DataFrame.from_records(scores, columns = col_names)
        return scores

    def by_month_r2(self, X, pred_name):
    
        """
        Бьет выборку по месяцам и считает помесячные значение Gini.
        model - модель
        time_period - поле, в котором находятся временные периоды
        X, y - данные (полностью подготовленные) и таргет
        good_bad_dict - словарь вида {'good': 1, 'bad': 0}, нужен для определения, на основании чего считать good_rate и bad_rate статистики

        """
        time_period = self.date_column 
        
        
        X.reset_index(inplace = True)
        X.drop('index', axis = 1, inplace = True)
        if target == None:
            target = y.name
        y_new = y.reset_index()[target]

        time_periods = sorted(X[time_period].unique())

        scores = []
        
        for i in time_periods:
            X_month = X.loc[X[time_period] == i].copy()
            X_index = X_month.index
            y_month = X_month[target]
            prediction = X_month[pred_name].copy()     

            if len(y_month)>0:
                y_mean = y_month.mean()
                r2 = metrics.r2_score(y_month, prediction)
                mae = metrics.mean_absolute_error(y_month, prediction)
            else:
                y_mean = np.nan
                r2 = np.nan
                mae = np.na

            scores.append([i, len(y_month), y_mean, r2, mae])

        col_names = [self.date_column, 'number', 'y_mean', 'score', 'mae']
        scores = pd.DataFrame.from_records(scores, columns = col_names)
        return scores
    
    def receive_attribute_vector(self, data):
        
        all_columns = data.columns.to_list()

        for i in self.technical_vars:
            if i in all_columns:
                all_columns.remove(i)

        attribute_list_model = self.attributes_list_new(data, all_columns, 
                                                              [1, 2, 5, 95, 98, 99])

        attribute_list_model['miss_share'] = attribute_list_model['count_miss']/data.shape[0]

        categories = self.category_list

        attribute_list_model.drop('type_val', axis = 1, inplace = True)
        attribute_list_model.drop('count_val', axis = 1, inplace = True)

        vector_postfix = attribute_list_model.columns.to_list()
        vector_postfix.remove('attribute')
        vector_postfix_cat = ['moda_val', 'count_value_moda']
        attribute_list_array = []
        attribute_list_names = []

        for i in all_columns:
            if i in categories:
                all_p = vector_postfix_cat.copy()
            else:
                all_p = vector_postfix.copy()
            for c in all_p:
                attribute_list_array.append(attribute_list_model.loc[attribute_list_model['attribute'] == i, c].to_list()[0])
                attribute_list_names.append(str(i)+'_'+str(c))

        attribute_list_vector = pd.DataFrame.from_records([attribute_list_array], columns = attribute_list_names)

        return attribute_list_vector
    
    def turn_variables_all(self, 
                             new_train,
                             new_test,
                             new_oot):
        selected_features_df = self.prep_dict['selected_features_df']
        
        new_train_sel = self.turn_variables_with_values(data=new_train,
                                                 rules_list=selected_features_df) 
        new_test_sel = self.turn_variables_with_values(data=new_test,
                                                 rules_list=selected_features_df) 
        new_oot_sel = self.turn_variables_with_values(data=new_oot,
                                                 rules_list=selected_features_df) 
        
        return new_train_sel, new_test_sel, new_oot_sel
    
    def preprocessing(self, 
                        new_train_sel,
                        new_test_sel,
                        new_oot_sel):
        '''
        Предобработка данных, в соответствии с правилами, полученными 
        при разработке исходной модели.

        Параметры:
        ---------

        new_train, new_test, new_oot : pandas DataFrame
            Сэмплированные новые train, test, oot.

        Возвращает:
        ----------
        Предобработанные данные, лист служебных переменных.

        '''      
        
        selected_features_df = self.prep_dict['selected_features_df']
        technical_vars = self.prep_dict['technical_vars']
        low_outlier = self.prep_dict['low_outlier']
        high_outlier = self.prep_dict['high_outlier']
        categorical_features_df = self.prep_dict['categorical_features_df']
        mediana = self.prep_dict['mediana']
        if self.prep_dict['yeo_johnson'] == None:
            yeo_johnson = False
        else:
            yeo_johnson = self.prep_dict['yeo_johnson']
        
        if self.prep_dict['preproc'] == None:
            preproc = False
        else:
            if hasattr(self.prep_dict['preproc'], 'mean_') == True:
                preproc = 'mean'
            else:
                preproc = 'minmax'
        
        if low_outlier != None:
            low_out = low_outlier
        else:
            low_out = 1
            
        if high_outlier != None:
            high_out = high_outlier
        else:
            high_out = 99
            
        attribute_list_model = self.attributes_list_new(new_train_sel, new_train_sel.columns, 
                                                          [low_out, high_out])   
        
        attribute_list_model['right_border'] = attribute_list_model[str(high_out)+'%']
        attribute_list_model['left_border'] = attribute_list_model[str(low_out)+'%']
        
        if low_outlier != None:
            left_border = 'left_border'
        else:
            left_border = None
            
        if high_outlier != None:
            right_border = 'right_border'
        else:
            right_border = None
            
            
        # Получаем лист разработанных признаков.
        features = selected_features_df['new variable'].to_list()
        # Удаляем служебные переменные из features.
        features_no_tech = [x for x in features if x not in technical_vars+[self.it]]
        # 'count_dist' == 2 соответствует категориальным переменным.
        categories = self.category_list
        non_outliers = attribute_list_model.loc[attribute_list_model['right_border']
                                                == attribute_list_model['left_border'], 
                                                    'attribute'].to_list()
        
        # cols_outliers - колонки, проходящие проверку на выбросы.
        cols_outliers = set(features_no_tech) - set(non_outliers)

        results = data_preprocessing_train(data=new_train_sel,
                                       target= self.it,
                                       technical_values=technical_vars,
                                       categorial_list=categories,
                                       drop_technical=False,
                                       attribute_list=attribute_list_model,
                                       var_col='attribute',
                                       median=mediana,
                                       high_outlier=right_border,
                                       low_outlier=left_border,
                                       scale=preproc,
                                       yeo_johnson=yeo_johnson,
                                       cols_outlier=cols_outliers)
                                       
        
        new_train_pr = results[0]

        if type(self.spline_split_points) is not type(None):
            new_train_pr1, splines_split_points = for_splines_train(new_train_pr[features_no_tech], new_train_pr[self.it], categories, max_d = self.splines_depth)
            new_train_pr1 = pd.concat([new_train_pr1, new_train_pr[technical_vars]], axis=1)
        else:
            new_train_pr1 = new_train_pr
        
        if len(results) == 3:
            preproc_file = results[2]
            yeo_johnson_file = results[1]
        else:
            preproc_file = results[1]
            yeo_johnson_file = None

        new_test_pr = data_preprocessing_test(data=new_test_sel,
                                       target= self.it,
                                       technical_values=technical_vars,
                                       categorial_list=categories,
                                       drop_technical=False,
                                       attribute_list=attribute_list_model,
                                       var_col='attribute',
                                       median=mediana,
                                       high_outlier=right_border,
                                       low_outlier=left_border,
                                       scale=preproc_file,
                                       yeo_johnson=yeo_johnson_file,
                                       cols_outlier=cols_outliers)
        
        new_oot_pr = data_preprocessing_test(data=new_oot_sel,
                                       target= self.it,
                                       technical_values=technical_vars,
                                       categorial_list=categories,
                                       drop_technical=False,
                                       attribute_list=attribute_list_model,
                                       var_col='attribute',
                                       median=mediana,
                                       high_outlier=right_border,
                                       low_outlier=left_border,
                                       scale=preproc_file,
                                       yeo_johnson=yeo_johnson_file,
                                       cols_outlier=cols_outliers)
        
        if type(self.spline_split_points) is not type(None):
            new_test_pr1 = for_splines_test(new_test_pr[features_no_tech], splines_split_points)
            new_oot_pr1 = for_splines_test(new_oot_pr[features_no_tech], splines_split_points)
            new_test_pr1 = pd.concat([new_test_pr1, new_test_pr[technical_vars]], axis=1)
            new_oot_pr1 = pd.concat([new_oot_pr1, new_oot_pr[technical_vars]], axis=1)
            features_no_tech=new_oot_pr1.columns
        else:
            new_test_pr1 = new_test_pr
            new_oot_pr1 = new_oot_pr
        
        return new_train_pr1, new_test_pr1, new_oot_pr1, attribute_list_model, features_no_tech
    
    def receive_threshold(self, y_train, y_train_pred):
        
        p_f_score, r_f_score, treshold_f_score = metrics.precision_recall_curve(y_train, y_train_pred)
        new_elements  = np.r_[0, treshold_f_score]
        f_score = (2*p_f_score*r_f_score)/(p_f_score+r_f_score)
        f_score = np.where(np.isnan(f_score), 0, f_score)
        cut_f = new_elements[np.argmax(f_score)]
        return cut_f
    
    def one_period_prediction_fit_model(self, train, test, oot):
        
        train1 = train.copy()
        test1 = test.copy()
        oot1 = oot.copy()
#         print('prepr')
        train_pr, test_pr, oot_pr, attribute_list_model, features_no_tech = self.preprocessing(train, test, oot)

        params_to_fit = self.model.get_params()
        model_to_fit = self.pass_model(**params_to_fit)

        fit_model = model_to_fit.fit(train_pr[features_no_tech], train_pr[self.it])
        
        if self.task == 'binary':
            
            train_pred = fit_model.predict_proba(train_pr[features_no_tech])[:, 1]
            test_pred = fit_model.predict_proba(test_pr[features_no_tech])[:, 1]
            oot_pred = fit_model.predict_proba(oot_pr[features_no_tech])[:, 1]
            
            Gini_train = metrics.roc_auc_score(train_pr[self.it], train_pred)*2-1 
            Gini_test = metrics.roc_auc_score(test_pr[self.it], test_pred)*2-1 
            
            train1['prediction'] = train_pred 
            test1['prediction'] = test_pred
            oot1['prediction'] = oot_pred
            
            train2 = train1[features_no_tech+['prediction', self.date_column, self.it]].copy()
            test2 = test1[features_no_tech+['prediction', self.date_column, self.it]].copy()
            oot2 = oot1[features_no_tech+['prediction', self.date_column, self.it]].copy()
            
            scores, vector_by_month = self.receive_all_vector_and_score_by_month(train2, oot2, 'prediction')
            oot_by_month1 = scores.reset_index()
            oot_by_month1['T'] = oot_by_month1['index']+1
            oot_by_month1['score_train'] = Gini_train
            oot_by_month1['score_test'] = Gini_test
            
            vector_by_month1 = vector_by_month.reset_index()
            vector_by_month1['T'] = vector_by_month1['index']+1
            vector_by_month1.drop('index', axis = 1, inplace = True)
            vector_by_month1['score_train'] = Gini_train
            vector_by_month1['score_test'] = Gini_test
            
            if self.reject_inf:
                cut_f = self.receive_threshold(train_pr[self.it], train_pred)
                
                train2_reject = train2.loc[train2['prediction'] > cut_f].reset_index(drop=True)
                test2_reject = test2.loc[test2['prediction'] > cut_f].reset_index(drop=True)
                oot2_reject = oot2.loc[oot2['prediction'] > cut_f].reset_index(drop=True)
                
                Gini_train_reject = metrics.roc_auc_score(train2_reject[self.it], train2_reject['prediction'])*2-1 
                Gini_test_reject = metrics.roc_auc_score(test2_reject[self.it], test2_reject['prediction'])*2-1 
                
                scores_reject, vector_by_month_reject = self.receive_all_vector_and_score_by_month(train2_reject, 
                                                                                                   oot2_reject, 
                                                                                                   'prediction')
                oot_by_month1_reject = scores_reject.reset_index()
                oot_by_month1_reject['T'] = oot_by_month1_reject['index']+1
                oot_by_month1_reject['score_train'] = Gini_train
                oot_by_month1_reject['score_test'] = Gini_test
                oot_by_month1_reject['score_train_reject'] = Gini_train_reject
                oot_by_month1_reject['score_test_reject'] = Gini_test_reject

                vector_by_month1_reject = vector_by_month_reject.reset_index()
                vector_by_month1_reject['T'] = vector_by_month1_reject['index']+1
                vector_by_month1_reject.drop('index', axis = 1, inplace = True)
                vector_by_month1_reject['score_train'] = Gini_train
                vector_by_month1_reject['score_test'] = Gini_test
                vector_by_month1_reject['score_train_reject'] = Gini_train_reject
                vector_by_month1_reject['score_test_reject'] = Gini_test_reject
            
        elif self.task == 'numeric':
            
            train_pred = fit_model.predict(train_pr[features_no_tech])
            test_pred = fit_model.predict(test_pr[features_no_tech])
            oot_pred = fit_model.predict(oot_pr[features_no_tech])
        
            r2_train = metrics.r2_score(train_pr[self.it], train_pred)
            r2_test = metrics.r2_score(test_pr[self.it], test_pred)
        
            train1['prediction'] = train_pred 
            test1['prediction'] = test_pred
            oot1['prediction'] = oot_pred
            
            train2 = train1[features_no_tech+['prediction', self.date_column, self.it, ]].copy()
            test2 = test1[features_no_tech+['prediction', self.date_column, self.it]].copy()
            oot2 = oot1[features_no_tech+['prediction', self.date_column, self.it]].copy()
            
            scores, vector_by_month = self.receive_all_vector_and_score_by_month(train2, oot2, 'prediction')
            oot_by_month1 = scores.reset_index()
            oot_by_month1['T'] = oot_by_month1['index']+1
            oot_by_month1['score_train'] = r2_train
            oot_by_month1['score_test'] = r2_test
            
            vector_by_month1 = vector_by_month.reset_index()
            vector_by_month1['T'] = vector_by_month1['index']+1
            vector_by_month1.drop('index', axis = 1, inplace = True)
            vector_by_month1['score_train'] = r2_train
            vector_by_month1['score_test'] = r2_test
            
            
        if self.reject_inf and self.task == 'binary':
            return oot_by_month1, vector_by_month1, oot_by_month1_reject, vector_by_month1_reject
        else:
            return oot_by_month1, vector_by_month1, oot_by_month1, vector_by_month1
    
    def get_one_sample_all_prediction_fit_model(self, num_period, periods):
        
        period = periods[num_period]
        
        train, test, oot = self.receive_new_all_train_test_oot(period)
        
        if self.task == 'binary':
            mpp_prediction = self.mpp.predict(data=oot, 
                                          by_month=True,
                                          preprocessed=False)
        
        train_new, test_new, oot_new = self.turn_variables_all(train, test, oot)
        
        oot_by_month, vector_by_month, oot_by_month_reject, vector_by_month_reject = self.one_period_prediction_fit_model(train_new,
                                                                                                                            test_new,
                                                                                                                            oot_new)
        
        oot_by_month['type'] = 'True'
        oot_by_month['num_period'] = num_period
        oot_by_month = pd.merge(oot_by_month, mpp_prediction[[self.date_column, 'F-score', '1_rate', 
                                                                  'G-score', 'G2-score', 'G3-score']], 
                                   on = self.date_column, how = 'left')
        
        return oot_by_month, vector_by_month, oot_by_month_reject, vector_by_month_reject
    
    def get_one_sample_shuffle_prediction_fit_model(self, all_periods, num_rep):
                        
        period = self.sample_shuffle_periods(all_periods, num_rep)
        
        train, test, oot = self.receive_new_shuffle_train_test_oot(period, num_rep)
        
        if self.task == 'binary':
            mpp_prediction = self.mpp.predict(data=oot, 
                                          by_month=True,
                                          preprocessed=False)
        
        train_new, test_new, oot_new = self.turn_variables_all(train, test, oot)
        
        oot_by_month, vector_by_month, oot_by_month_reject, vector_by_month_reject = self.one_period_prediction_fit_model(train_new,
                                                                                                                          test_new,
                                                                                                                          oot_new)
        oot_by_month['type'] = 'Shuffle'
        oot_by_month['num_period'] = num_rep
        oot_by_month = pd.merge(oot_by_month, mpp_prediction[[self.date_column, 'F-score', '1_rate', 
                                                                  'G-score', 'G2-score', 'G3-score']], 
                                   on = self.date_column, how = 'left')
        
        return oot_by_month, vector_by_month, oot_by_month_reject, vector_by_month_reject
    
    def one_period_prediction(self, train, test, oot):
        
        train_pr, test_pr, oot_pr, attribute_list_model, features_no_tech = self.preprocessing(train, test, oot)

        params_to_fit = self.model.get_params()
        model_to_fit = self.pass_model(**params_to_fit)

        fit_model = model_to_fit.fit(train_pr[features_no_tech], train_pr[self.it])
        
        if self.task == 'binary':
            
            good_bad_dict = self.prep_dict['good_bad_dict']
            train_pred = model_to_fit.predict_proba(train_pr[features_no_tech])[:, 1]
            test_pred = model_to_fit.predict_proba(test_pr[features_no_tech])[:, 1]
            oot_pred = model_to_fit.predict_proba(oot_pr[features_no_tech])[:, 1]
            
            train_pr['prediction'] = train_pred
            test_pr['prediction'] = test_pred
            oot_pr['prediction'] = oot_pred
            
            Gini_train = roc_auc_score(train_pr[self.it], train_pred)*2-1
            Gini_test = roc_auc_score(test_pr[self.it], test_pred)*2-1
            
            oot_by_month = self.by_month_gini(oot_pr.copy(), 'prediction')
            oot_by_month['score_train'] = Gini_train
            oot_by_month['score_test'] = Gini_test
            
            if self.reject_inf:
                cut_f = self.receive_threshold(train_pr[self.it], train_pred)
                
                train2_reject = train_pr.loc[train_pr['prediction'] > cut_f].reset_index(drop=True)
                test2_reject = test_pr.loc[test_pr['prediction'] > cut_f].reset_index(drop=True)
                oot2_reject = oot_pr.loc[oot_pr['prediction'] > cut_f].reset_index(drop=True)
                
                Gini_train_reject = metrics.roc_auc_score(train2_reject[self.it], train2_reject['prediction'])*2-1 
                Gini_test_reject = metrics.roc_auc_score(test2_reject[self.it], test2_reject['prediction'])*2-1 
                
                scores_reject = self.by_month_gini(oot2_reject.copy(), 'prediction')
                oot_by_month_reject = scores_reject.reset_index()
                oot_by_month_reject['T'] = oot_by_month_reject['index']+1
                oot_by_month_reject['score_train'] = Gini_train
                oot_by_month_reject['score_test'] = Gini_test
                oot_by_month_reject['score_train_reject'] = Gini_train_reject
                oot_by_month_reject['score_test_reject'] = Gini_test_reject
            
        else:
            train_pred = model_to_fit.predict(train_pr[features_no_tech])
            test_pred = model_to_fit.predict(test_pr[features_no_tech])
            oot_pred = model_to_fit.predict(oot_pr[features_no_tech])
            
            train_pr['prediction'] = train_pred
            test_pr['prediction'] = test_pred
            oot_pr['prediction'] = oot_pred
            
            r2_train = metrics.r2_score(train_pr[self.it], train_pred)
            r2_test = metrics.r2_score(test_pr[self.it], test_pred)
            
            oot_by_month = self.by_month_r2(oot_pr, 'prediction')
            oot_by_month['score_train'] = r2_train
            oot_by_month['score_test'] = r2_test
            
        oot_by_month1 = oot_by_month.reset_index()
        oot_by_month1['T'] = oot_by_month1['index']+1
        
        if self.reject_inf and self.task == 'binary':
            return oot_by_month1, oot_by_month_reject
        else:
            return oot_by_month1, oot_by_month1
        
    def get_one_sample_all_prediction(self, num_period, periods):
        
        period = periods[num_period]
        
        train, test, oot = self.receive_new_all_train_test_oot(period)
        
        if self.task == 'binary':
            mpp_prediction = self.mpp.predict(data=oot, 
                                          by_month=True,
                                          preprocessed=False)
        
        train_new, test_new, oot_new = self.turn_variables_all(train, test, oot)
        
        oot_by_month, oot_by_month_reject = self.one_period_prediction(train_new, test_new, oot_new)
        
        oot_by_month['type'] = 'True'
        oot_by_month['num_period'] = num_period
        
        oot_by_month = pd.merge(oot_by_month, mpp_prediction[[self.date_column, 'F-score', '1_rate', 
                                                                  'G-score', 'G2-score', 'G3-score']], 
                                   on = self.date_column, how = 'left')
        
        return oot_by_month, oot_by_month_reject
    
    def get_one_sample_shuffle_prediction(self, all_periods, num_rep, fit_model = True):
                
        period = self.sample_shuffle_periods(all_periods, num_rep)
        
        train, test, oot = self.receive_new_shuffle_train_test_oot(period, num_rep)
        
        if self.task == 'binary':
            mpp_prediction = self.mpp.predict(data=oot, 
                                          by_month=True,
                                          preprocessed=False)
        
        train_new, test_new, oot_new = self.turn_variables_all(train, test, oot)
        
        oot_by_month, oot_by_month_reject = self.one_period_prediction(train_new, test_new, oot_new)
        
        oot_by_month['type'] = 'Shuffle'
        oot_by_month['num_period'] = num_rep
        oot_by_month = pd.merge(oot_by_month, mpp_prediction[[self.date_column, 'F-score', '1_rate', 
                                                                  'G-score', 'G2-score', 'G3-score']], 
                                   on = self.date_column, how = 'left')
        
        return oot_by_month, oot_by_month_reject
        
    def calculate_psi_for_one_period(self, train, test, N = 50, encode = False, 
                                       groupped_s = 'N number obs', n_jobs = 3, verbose = 0):
            
        Statistics = []
        k_test = 0

        def one_col_psi(o):
            nonlocal k_test
            nonlocal Statistics
            k_test = k_test+1

            if train[columns[o]].dtype == 'object' or (type(self.category_list) != type(None) and columns[o] in self.category_list and encode == True):

                x = train[columns[o]].copy()
                y = test[columns[o]].copy()

                everything = pd.concat([x, y])
                encoder = LabelEncoder().fit(everything.fillna('MISSING'))                    

                x_label = pd.Series(encoder.transform(x.fillna('MISSING')))
                y_label = pd.Series(encoder.transform(y.fillna('MISSING')))

                KS = stats.ks_2samp(x_label, y_label)

                Statistics.append([columns[o], 
                                   get_PSI_stat(x_label, y_label, N, category = True, encode = False, 
                                                groupped_s = groupped_s)*100])
                gc.collect()

            elif type(self.category_list) != type(None) and columns[o] in self.category_list and encode == False:

                x = train[columns[o]].copy()
                y = test[columns[o]].copy()

                everything = pd.concat([x, y])

                x_label = x.fillna(everything.min()-1)
                y_label = y.fillna(everything.min()-1)

                KS = stats.ks_2samp(x_label, y_label)

                Statistics.append([columns[o], get_PSI_stat(x_label, y_label, N, category = True, 
                                                            encode = False, groupped_s = groupped_s)*100])
                gc.collect()


            else:
                x = train[columns[o]].copy()
                y = test[columns[o]].copy()

                everything = pd.concat([x, y])
                minimum = everything.min()-1

                x = x.fillna(minimum)
                y = y.fillna(minimum)

                KS = stats.ks_2samp(x, y)

                Statistics.append([columns[o], 
                                 get_PSI_stat(x, y, N, category = False, encode = False, groupped_s = groupped_s)*100])

        gc.collect()

        columns = test.columns.to_list()

        for i in self.technical_vars:
            if i in columns:
                columns.remove(i)

        if self.it in columns:
            columns.remove(it)

        parallel = Parallel(n_jobs=n_jobs, require='sharedmem', verbose = verbose)
        with parallel:
            par_res = parallel((delayed(one_col_psi)(o) for o in range(len(columns))))

        labels = ['variable', 'psi']
        psi_check = pd.DataFrame.from_records(Statistics, columns = labels)
        psi_check['one'] =1
        psi_check1 = psi_check.pivot(index='one', columns='variable', values='psi').reset_index(drop = True)
        psi_check1.columns = [col+'_PSI' for col in psi_check1.columns.values]

        return psi_check1

    def receive_attribute_vectors_by_prediction(self, data, prediction, number_cuts = 3):
        
        data_c = data.copy()

        data_c['group'] = pd.qcut(data_c[prediction], number_cuts, duplicates = 'drop')

        unique_sorted = sorted(data_c['group'].unique())

        all_fin_vects = []

        for group, name in zip(unique_sorted, ['small_pred', 'medium_pred', 'big_pred']):
            data1 = data_c.loc[data_c['group'] == group].copy()
            data1.drop(['group', prediction], axis = 1, inplace = True)

            vect = self.receive_attribute_vector(data1)
            vect.columns = [i+'_'+name for i in vect.columns]

            all_fin_vects.append(vect)

        all_fin_vector_all = pd.concat(all_fin_vects, axis=1)

        return all_fin_vector_all
        
    def receive_all_vector(self, train, test, prediction):
        
        attribute_vector = self.receive_attribute_vector(test)

        psi_check = self.calculate_psi_for_one_period(train, test, N = 50, encode = False,
                                           groupped_s = 'N number obs', n_jobs = self.n_jobs, verbose = 0)

        attribute_vector_by_prediction = self.receive_attribute_vectors_by_prediction(test, prediction, 
                                                                                        number_cuts = 3)

        result_vector_cols = attribute_vector.columns.to_list() + psi_check.columns.to_list() + attribute_vector_by_prediction.columns.to_list()

        result_vector = pd.concat([attribute_vector, psi_check, attribute_vector_by_prediction], axis = 1, 
                                  ignore_index = True)

        result_vector.columns = [i for i in result_vector_cols]

        return result_vector
    
    def by_month_score(self, model, X, y):
        
        if self.task == 'binary':
            scores = self.by_month_gini(model, X, y)
            
        elif self.task == 'numeric':
            scores = self.by_month_r2(model, X, y)
            
        return scores
    
    def receive_all_vector_and_score_by_month(self, train, test, prediction):
    
        train.reset_index(inplace = True, drop = True)
        test.reset_index(inplace = True, drop = True)
#         target = y.name
#         y_new = y.reset_index()[self.it]
        
        all_vectors = []        
        scores = []
        
        if self.task == 'binary':
            train_score = metrics.roc_auc_score(train[self.it], train[prediction])*2-1
        elif self.task == 'numeric':
            train_score = metrics.r2_score(train[self.it], train[prediction])

        for i in sorted(test[self.date_column].unique()):
            
            test1 = test.loc[test[self.date_column] == i].copy().reset_index(drop = True).drop(self.date_column, 
                                                                                               axis = 1)

            y_month = test1[self.it].copy()
            pr = test1[prediction].copy()
            
            test1.drop(self.it, inplace = True, axis = True)
            
            if self.task == 'binary':
                
                if y_month.sum()>0:
                    score = metrics.roc_auc_score(y_month, pr)*2-1
                else:
                    score = np.nan
                
            elif self.task == 'numeric':
                score = metrics.r2_score(y_month, pr)
                
            scores.append([i, len(y_month), score])   
            
            if self.fit_model == True:
                all_vector_i = self.receive_all_vector(train, test1, prediction)

                all_vectors.append(all_vector_i)

        col_names = [self.date_column, 'number', 'score']
        scores = pd.DataFrame.from_records(scores, columns = col_names)
        
        all_vectors_df = pd.concat(all_vectors, ignore_index = True)
        
        return scores, all_vectors_df
    
    def plot_scores_prediction(self, results_by_month):
        
        fig, axes = plt.subplots(2,
                         1, figsize = (5, 5), dpi = 100, 
                         sharex = False, constrained_layout = True)
        seab1 = sns.lineplot(x=results_by_month["T"], y=results_by_month["score"], ax = axes[0], 
                             color = 'blue')
    
    def stratified_split(self, data, target, list_of_vars_for_strat, sort_by_var, size_of_test, drop_technical, 
                           random_state):
    
        """
        Стратифицированно бьет данные на трейн и тест.
        Подаваемые параметры:

        data - передаваемые данные
        target - название таргетной переменной
        list_of_vars_for_strat - список переменных, по которым производится стратификация
        sort_by_var - переменная, по которой производится группировка (id клиентов/транзакций и пр). В этой переменной 
        не должно быть пропусков, но если пропуски заполнить одним значением, метод "решит", что это один и тот же клиент. 
        Поэтому если есть пропуски, то их следует заполнять уникальными, не встреченными ранее в данных значениями
        size_of_test - размер тестовой выборки
        drop_technical - удалить ли "технические" переменные (лист переменных, по которым делается стратификация, 
        группирующая переменная)

        """

        max_target = data.groupby(sort_by_var).aggregate({target: 'max'})
        max_target = max_target.reset_index()

        data = pd.merge(data, max_target, on = sort_by_var, suffixes = ["", "_max"])

        target1 = target+"_max"

        if len(list_of_vars_for_strat) == 0:
            list_of_vars_for_strat = [target1]
        if target in list_of_vars_for_strat:
            list_of_vars_for_strat.remove(target)
            list_of_vars_for_strat.append(target1)
        else:
            list_of_vars_for_strat.append(target1)

        for i in list_of_vars_for_strat:
            if i == list_of_vars_for_strat[0]:
                data['For_stratify'] = data[i].astype('str')
            else:
                data['For_stratify'] += data[i].astype('str')

        data_nodup = data[[sort_by_var, 'For_stratify', target1]].drop_duplicates(subset = sort_by_var)


        train, test, target_train, target_test = train_test_split(data_nodup, data_nodup[target1], 
                                                    test_size = size_of_test, random_state = random_state)

        X_train = data[data[sort_by_var].isin(train[sort_by_var])].copy()
        train_index = X_train.index
        y_train = data.iloc[train_index][target].copy()
        X_test = data[data[sort_by_var].isin(test[sort_by_var])].copy()
        test_index = X_test.index
        y_test = data.iloc[test_index][target].copy()
        
        if drop_technical == True:

            X_train.drop(list_of_vars_for_strat, axis = 1, inplace = True)
            X_train.drop(sort_by_var, axis = 1, inplace = True)

            X_test.drop(list_of_vars_for_strat, axis = 1, inplace = True)
            X_test.drop(sort_by_var, axis = 1, inplace = True)

        else:
            X_train.drop(target1, axis = 1, inplace = True)
            X_test.drop(target1, axis = 1, inplace = True)

        X_train.drop(target, axis = 1, inplace = True)
        X_train.drop('For_stratify', axis = 1, inplace = True)

        X_test.drop(target, axis = 1, inplace = True)
        X_test.drop('For_stratify', axis = 1, inplace = True)   

        return X_train, X_test, y_train, y_test
        
    def fit_and_graph(self):
        
        # Получить общий список месяцев 
        
        all_periods = self.receive_train_period()
                
        # Получить прогнозы для всех периодов без бутстрэпа
        train_periods = self.sample_all_periods(all_periods)
        
        self.all_periods = all_periods
        self.train_periods = train_periods
        
        print('number of examples = ', len(train_periods))
        
        parallel = Parallel(n_jobs=self.n_jobs, require='sharedmem', verbose = self.verbose)  
        
        if self.fit_model == True:
            
            with parallel:
                results = parallel((delayed(self.get_one_sample_all_prediction_fit_model)(i, 
                                        train_periods) for i in np.arange(len(train_periods))))
            
            oot_by_month_list = [result[0] for result in results]
            vector_by_month_list = [result[1] for result in results]
            
            oot_by_month_list_reject = [result[2] for result in results]
            vector_by_month_list_reject = [result[3] for result in results]
            
            self.oot_by_month_df = pd.concat(oot_by_month_list, ignore_index = True)
            self.vector_by_month_df = pd.concat(vector_by_month_list, ignore_index = True)
            
            if self.reject_inf:
                self.oot_by_month_df_reject = pd.concat(oot_by_month_list_reject, ignore_index = True)
                self.vector_by_month_df_reject = pd.concat(vector_by_month_list_reject, ignore_index = True)

            if self.n_reps > 0:
                print('number of examples shuffle = ', self.n_reps)
                with parallel:
                    results_shuffle = parallel((delayed(self.get_one_sample_shuffle_prediction_fit_model)(all_periods,
                                                               num_rep) for num_rep in np.arange(self.n_reps)))

                oot_shuffle_by_month_list = [result[0] for result in results_shuffle]
                vector_shuffle_by_month_list = [result[1] for result in results_shuffle]
                
                oot_shuffle_by_month_list_reject = [result[2] for result in results_shuffle]
                vector_shuffle_by_month_list_reject = [result[3] for result in results_shuffle]
                
                oot_by_month_list2 = oot_by_month_list + oot_shuffle_by_month_list
                vector_by_month_list2 = vector_by_month_list + vector_shuffle_by_month_list
                oot_by_month_list2_reject = oot_by_month_list_reject + oot_shuffle_by_month_list_reject
                vector_by_month_list2_reject = vector_by_month_list_reject + vector_shuffle_by_month_list_reject
                
                self.oot_by_month_df2 = pd.concat(oot_by_month_list2, ignore_index = True)
                self.vector_by_month_df2 = pd.concat(vector_by_month_list2, ignore_index = True)
                
                if self.reject_inf:
                    self.oot_by_month_df_reject = pd.concat(oot_by_month_list2_reject, ignore_index = True)
                    self.vector_by_month_df_reject = pd.concat(vector_by_month_list2_reject, ignore_index = True)
                    outputs = (self.oot_by_month_df, self.oot_by_month_df2, 
                               self.vector_by_month_df, self.vector_by_month_df2,
                               self.oot_by_month_df_reject, self.vector_by_month_df_reject,
                               self.oot_by_month_df2_reject, self.vector_by_month_df2_reject)
                else:
                    outputs = (self.oot_by_month_df, self.oot_by_month_df2, self.vector_by_month_df, self.vector_by_month_df2)
                
            else:
                if self.reject_inf:
                    outputs = (self.oot_by_month_df, self.vector_by_month_df, 
                               self.oot_by_month_df_reject, self.vector_by_month_df_reject)
                else:
                    outputs = (self.oot_by_month_df, self.vector_by_month_df)
            
        elif self.fit_model == False:
        
            with parallel:
                results = parallel((delayed(self.get_one_sample_all_prediction)(i, 
                                        train_periods) for i in np.arange(len(train_periods))))
            oot_by_month_list = [result[0] for result in results]
            oot_by_month_list_reject = [result[1] for result in results]
            self.oot_by_month_df = pd.concat(oot_by_month_list, ignore_index = True)

            if self.n_reps > 0:
                print('number of examples shuffle = ', self.n_reps)
                with parallel:
                    results_shuffle = parallel((delayed(self.get_one_sample_shuffle_prediction)(all_periods,
                                                               num_rep) for num_rep in np.arange(self.n_reps)))

                oot_shuffle_by_month_list = [result[0] for result in results_shuffle]
                oot_shuffle_by_month_list_reject = [result[1] for result in results_shuffle]
                    
                oot_by_month_list2 = oot_by_month_list + oot_shuffle_by_month_list
                self.oot_by_month_df2 = pd.concat(oot_by_month_list2, ignore_index = True)
                
                if self.reject_inf:
                    self.oot_by_month_df_reject = pd.concat(oot_by_month_list_reject, ignore_index = True)
                    oot_by_month_list2_reject = oot_by_month_list_reject + oot_shuffle_by_month_list_reject
                    self.oot_by_month_df2_reject = pd.concat(oot_by_month_list2_reject, ignore_index = True)
                    outputs = (self.oot_by_month_df, self.oot_by_month_df2,
                               self.oot_by_month_df_reject, self.oot_by_month_df2_reject)
                else:
                    outputs = (self.oot_by_month_df, self.oot_by_month_df2)
            else:
                if self.reject_inf:
                    self.oot_by_month_df_reject = pd.concat(oot_by_month_list_reject, ignore_index = True)
                    outputs = (self.oot_by_month_df, self.oot_by_month_df_reject)
                else:
                    outputs = self.oot_by_month_df
        
        self.plot_ginis()
        
        return outputs
    
    def one_plot(self, data, name, valid = None):
        res_1 = data.groupby(by='T')['score'].aggregate(['mean', lambda x: np.percentile(x, 5),
                                                         lambda x: np.percentile(
                                                             x, 95),
                                                         ]).reset_index().rename(columns={'<lambda_0>': 'perc_5',
                                                                                          '<lambda_1>': 'perc_95'})
        x_1 = res_1['T'].to_list()
        y_1 = res_1['mean'].to_list()
        y_upper_1 = res_1['perc_95'].to_list()
        y_lower_1 = res_1['perc_5'].to_list()

        if type(valid)==type(None):
            fig1 = go.Figure([
                go.Scatter(
                    x=x_1,
                    y=y_1,
                    line=dict(color='rgb(0, 0, 250)'),
                    mode='lines'),
                go.Scatter(
                    x=x_1+x_1[::-1],  # x, then x reversed
                    y=y_upper_1+y_lower_1[::-1],  # upper, then lower reversed
                    fill='toself',
                    fillcolor='rgba(0, 0, 250,0.2)',
                    line=dict(color='rgba(0, 0, 250,0)'),
                    hoverinfo="skip",
                    showlegend=False)
            ])

            fig1.update_layout(
                title={'text': str(''.join([name])),
                       'x': 0.5,
                       'xanchor': 'center'},
                autosize=False,
                width=700,
                height=500,
                margin=dict(l=0, r=50, b=30, t=50, pad=4),
                hovermode='x',
                xaxis={'title': 'Month', 'type': 'category'},
                yaxis={'title': 'Metric value', 'range': [0, 1]},
                legend=dict(font=dict(size=16), bgcolor="#e5ecf6"))
        else:
            fig1 = go.Figure([
                go.Scatter(
                    x=x_1,
                    y=y_1,
                    line=dict(color='rgb(0, 0, 250)'),
                    mode='lines'),
                go.Scatter(
                    x=x_1+x_1[::-1],  # x, then x reversed
                    y=y_upper_1+y_lower_1[::-1],  # upper, then lower reversed
                    fill='toself',
                    fillcolor='rgba(0, 0, 250,0.2)',
                    line=dict(color='rgba(0, 0, 250,0)'),
                    hoverinfo="skip",
                    showlegend=False),
                go.Scatter(
                    x=valid['T'].to_list(),  
                    y=valid['score'].to_list(), 
                    line=dict(color='rgb(222, 0, 111,0)'),
                    hoverinfo="skip",
                    showlegend=False)
            ])

            fig1.update_layout(
                title={'text': str(''.join([name])),
                       'x': 0.5,
                       'xanchor': 'center'},
                autosize=False,
                width=700,
                height=500,
                margin=dict(l=0, r=50, b=30, t=50, pad=4),
                hovermode='x',
                xaxis={'title': 'Month', 'type': 'category'},
                yaxis={'title': 'Metric value', 'range': [0, 1]},
                legend=dict(font=dict(size=16), bgcolor="#e5ecf6"))

        fig1.show()
    
    def plot_ginis(self, valid=None):
        
        self.one_plot(self.oot_by_month_df, 
                      'Прогноз деградации модели без перемешивания данных:', valid)
        
        if self.reject_inf:
            self.one_plot(self.oot_by_month_df_reject, 
                          'Прогноз деградации модели без перемешивания данных with reject:', valid)
        
        if self.n_reps > 0:
            self.one_plot(self.oot_by_month_df2, 
                          'Прогноз деградации модели с перемешиванием данных:', valid)
              
            if self.reject_inf:
                self.one_plot(self.oot_by_month_df2_reject, 
                              'Прогноз деградации модели без перемешивания данных with reject:', valid)
                
        return self
    
    def calculate_r2_for_several_vars(self, train, y_train, technical_vars, n_jobs = 3, verbose = 5):
    
        cols = []
        r2_train_list = []
        corr_list = []
        corr_pvalue_list = []

        def calculate_r2_for_one_var(col):

            nonlocal r2_train_list
            nonlocal corr_list
            nonlocal corr_pvalue_list

            model = LinearRegression()

            model_fit = model.fit(pd.DataFrame(train[col]), y_train)
            train_pred = model_fit.predict(pd.DataFrame(train[col]))

            r2_train = metrics.r2_score(y_train, train_pred)
            corr = pearsonr(train[col], y_train)[0]
            corr_pvalue = pearsonr(train[col], y_train)[1]

            cols.append(col)
            r2_train_list.append(r2_train)
            corr_list.append(corr)
            corr_pvalue_list.append(corr_pvalue)

        columns = train.columns.to_list()

        for i in technical_vars:
            if i in columns:
                columns.remove(i)

        parallel = Parallel(n_jobs=n_jobs, require='sharedmem', verbose = verbose)
        with parallel:
            par_res = parallel((delayed(calculate_r2_for_one_var)(col) for col in columns))


        scores = pd.DataFrame()
        scores['variable'] = cols
        scores['score'] = r2_train_list
        scores['corr'] = corr_list
        scores['pvalue'] = corr_pvalue_list

        return scores
    
    def fit_model_by_mpp(self):
        
        if self.n_reps > 0:
            all_f = self.oot_by_month_df2.copy()
            all_f['num_period_to_str'] = self.oot_by_month_df2['num_period'].astype('str')+'_'+self.oot_by_month_df2['type']
            
        else:
            all_f = self.oot_by_month_df2.copy()
            all_f['num_period_to_str'] = self.oot_by_month_df['num_period'].astype('str')+'_'+self.oot_by_month_df['type']
        
        X_train, X_test, y_train, y_test = self.stratified_split(all_f, 'score', [], 'num_period_to_str', 
                                                                   size_of_test = 0.1, drop_technical = True, 
                                                                   random_state = 421)
        
        mpp_columns = ['number', 'score_test', 'F-score', '1_rate', 'G-score', 'G2-score', 'G3-score']
        
        pr = preprocessing.StandardScaler()
        pr.fit(X_train[mpp_columns])
        X_train2 = pd.DataFrame(pr.transform(X_train[mpp_columns]), columns = mpp_columns)
        X_test2 = pd.DataFrame(pr.transform(X_test[mpp_columns]), columns = mpp_columns)
        
        model_fin = LinearRegression()
        print(X_train2, y_train)
        model_fit = model_fin.fit(X_train2, y_train)
        prediction_train = model_fit.predict(X_train2)
        prediction_test = model_fit.predict(X_test2)
        
        print('R2 train for model prediction score =', metrics.r2_score(y_train, prediction_train))
        print('R2 test for model prediction score =', metrics.r2_score(y_test, prediction_test))
        
        self.r2_model_mpp = model_fit
        self.pr_mpp = pr
        self.mpp_columns = mpp_columns
        
        return self.mpp_columns, self.pr_mpp, self.r2_model_mpp
    
    def fit_model_by_results(self):
        
        if self.n_reps > 0:
            all_f = self.vector_by_month_df2
            all_f['score'] = self.oot_by_month_df2['score']
            all_f['num_period'] = self.oot_by_month_df2['num_period'].astype('str')+'_'+self.oot_by_month_df2['type']
            
        else:
            all_f = self.vector_by_month_df
            all_f['score'] = self.oot_by_month_df['score']
            all_f['num_period'] = self.oot_by_month_df['num_period'].astype('str')+'_'+self.oot_by_month_df['type']
            
        X_train, X_test, y_train, y_test = self.stratified_split(all_f, 'score', [], 'num_period', 
                                                                   size_of_test = 0.1, drop_technical = True, 
                                                                   random_state = 421)
        
        r2_vector = self.calculate_r2_for_several_vars(X_train, y_train, ['num_period'], n_jobs = self.n_jobs, 
                                                  verbose = self.verbose)
        
        sel_cols = r2_vector.loc[(r2_vector['score'] > 0.05) & (r2_vector['pvalue'] <= 0.05), 'variable'].to_list()
        X_train1 = X_train[sel_cols].copy()
        corr = X_train1.corr()
        result_tab, non_doubles, col_doubles, double_dic = find_doubles_corr(X_train1, sel_cols, 
                                                                     corr, definition = r2_vector, 
                                                                     lvl = 0.6, light_unstable = None)
        
        fin_colls =non_doubles+['T', 'score_test']
        
        pr = preprocessing.StandardScaler()
        pr.fit(X_train[fin_colls])
        X_train2 = pd.DataFrame(pr.transform(X_train[fin_colls]), columns = fin_colls)
        X_test2 = pd.DataFrame(pr.transform(X_test[fin_colls]), columns = fin_colls)
        
        model_fin = LinearRegression()
        model_fit = model_fin.fit(X_train2, y_train)
        prediction_train = model_fit.predict(X_train2)
        prediction_test = model_fit.predict(X_test2)
        
        print('R2 train for model prediction score =', metrics.r2_score(y_train, prediction_train))
        print('R2 test for model prediction score =', metrics.r2_score(y_test, prediction_test))
        
        self.r2_vector = r2_vector
        self.corr = corr
        self.fin_colls = fin_colls
        self.r2_model = model_fit
        self.pr = pr
        
        return self.r2_vector, self.corr, self.fin_colls, self.pr, self.r2_model 
    
    def save_tables(self):
        
        if self.fit_model == True:
            
            if self.n_reps > 0:
                self.oot_by_month_df.to_csv(self.gini_degradation_path+'oot_by_month_only_true.csv', sep = '&')
                self.oot_by_month_df2.to_csv(self.gini_degradation_path+'oot_by_month_with_shuffle.csv', sep = '&')
                self.vector_by_month_df.to_csv(self.gini_degradation_path+'vector_by_month_only_true.csv', sep = '&')
                self.vector_by_month_df2.to_csv(self.gini_degradation_path+'vector_by_month_with_shuffle.csv', sep = '&')
                if self.reject_inf:
                    self.oot_by_month_df_reject.to_csv(self.gini_degradation_path+'oot_by_month_only_true_reject.csv', sep = '&')
                    self.oot_by_month_df2_reject.to_csv(self.gini_degradation_path+'oot_by_month_with_shuffle_reject.csv', sep = '&')
                    self.vector_by_month_df_reject.to_csv(self.gini_degradation_path+'vector_by_month_only_true_reject.csv', sep = '&')
                    self.vector_by_month_df2_reject.to_csv(self.gini_degradation_path+'vector_by_month_with_shuffle_reject.csv', sep = '&')
            else:
                self.oot_by_month_df.to_csv(self.gini_degradation_path+'oot_by_month_only_true.csv', sep = '&')
                self.vector_by_month_df.to_csv(self.gini_degradation_path+'vector_by_month_only_true.csv', sep = '&')
                if self.reject_inf:
                    self.oot_by_month_df_reject.to_csv(self.gini_degradation_path+'oot_by_month_only_true_reject.csv', sep = '&')
                    self.vector_by_month_df_reject.to_csv(self.gini_degradation_path+'vector_by_month_only_true_reject.csv', sep = '&')
                
            self.r2_vector.to_csv(self.gini_degradation_path+'r2_vector.csv', sep = '&')
            self.corr.to_csv(self.gini_degradation_path+'corr_matrix.csv', sep = '&')
            joblib.dump(self.fin_colls, self.gini_degradation_path+'selected_columns.dat')
            joblib.dump(self.pr, self.gini_degradation_path+'r2_preprocessing.dat')
            joblib.dump(self.r2_model, self.gini_degradation_path+'r2_model.dat')
        
        else:
            if self.n_reps > 0:
                self.oot_by_month_df.to_csv(self.gini_degradation_path+'oot_by_month_only_true.csv', sep = '&')
                self.oot_by_month_df2.to_csv(self.gini_degradation_path+'oot_by_month_with_shuffle.csv', sep = '&')
                if self.reject_inf:
                    self.oot_by_month_df_reject.to_csv(self.gini_degradation_path+'oot_by_month_only_true_reject.csv', sep = '&')
                    self.oot_by_month_df2_reject.to_csv(self.gini_degradation_path+'oot_by_month_with_shuffle_reject.csv', sep = '&')
            else:
                self.oot_by_month_df.to_csv(self.gini_degradation_path+'oot_by_month_only_true.csv', sep = '&')
                if self.reject_inf:
                    self.oot_by_month_df_reject.to_csv(self.gini_degradation_path+'oot_by_month_only_true_reject.csv', sep = '&')
        
        #if self.task == 'binary':
            #joblib.dump(self.r2_model_mpp, self.gini_degradation_path+'r2_preprocessing_mpp.dat')
            #joblib.dump(self.pr_mpp, self.gini_degradation_path+'r2_model_mpp.dat')
            #joblib.dump(self.mpp_columns, self.gini_degradation_path+'mpp_columns.dat')
        
        return self
    
    def fit(self):
        
        results = self.fit_and_graph()
            
        saved = self.save_tables()
        
        return results
    
    def predict_compare(self, valid_sample):
        
        valid_sample.columns = [i.upper() for i in valid_sample.columns]
        
        attribute_list_model = self.prep_dict['attribute_list_model']
        selected_features_df = self.prep_dict['selected_features_df']
        technical_vars = self.prep_dict['technical_vars']
        low_outlier = self.prep_dict['low_outlier']
        high_outlier = self.prep_dict['high_outlier']
        categorical_features_df = self.prep_dict['categorical_features_df']
        mediana = self.prep_dict['mediana']
        attribute_list_model = self.prep_dict['attribute_list_model']
        yeo_johnson = self.prep_dict['yeo_johnson']
        preproc = self.prep_dict['preproc']
        spline_split_points = self.spline_split_points
        
        if 'right_border' in list(self.prep_dict.keys()):
            right_border =  self.prep_dict['right_border']
        else:
            right_border = 'right_border'
        if 'left_border' in list(self.prep_dict.keys()):
            left_border = self.prep_dict['left_border']
        else:
            left_border = 'left_border'
        
        
        # Получаем лист разработанных признаков.
        features = selected_features_df['new variable'].to_list()
        # Удаляем служебные переменные из features.
        features_no_tech = [x for x in features if x not in technical_vars]
        # 'count_dist' == 2 соответствует категориальным переменным.
        if self.old_model == False:
            categories = self.category_list
        else:
            categories = []
            
        non_outliers = attribute_list_model.loc[attribute_list_model[right_border] == attribute_list_model[left_border], 
                                                    'attribute'].to_list()
        # cols_outliers - колонки, проходящие проверку на выбросы.
        cols_outliers = set(features_no_tech) - set(non_outliers)        

        
        y_train = self.train[self.it]
        y_test = self.test[self.it]
        y_valid = valid_sample[self.it]
        
        train_turned = self.turn_variables_with_values(data=self.train,
                                                 rules_list=selected_features_df) 
        test_turned = self.turn_variables_with_values(data=self.test,
                                                 rules_list=selected_features_df) 
        valid_turned = self.turn_variables_with_values(data=valid_sample,
                                                 rules_list=selected_features_df) 
        
        if self.it not in technical_vars:
            technical_vars.append(self.it)
        
        train_prepr = data_preprocessing_test(train_turned, y_train, technical_vars, categories,
                                               drop_technical = True,
                                               attribute_list = attribute_list_model, 
                                               var_col = 'attribute',
                                               median = 'val_mediana',
                                               high_outlier = right_border, 
                                               low_outlier = left_border, scale = preproc, 
                                               yeo_johnson = yeo_johnson, cols_outlier = cols_outliers)

        test_prepr = data_preprocessing_test(test_turned, y_test, technical_vars, categories,
                                               drop_technical = True,
                                               attribute_list = attribute_list_model, 
                                               var_col = 'attribute',
                                               median = 'val_mediana',
                                               high_outlier = right_border, 
                                               low_outlier = left_border, scale = preproc, 
                                               yeo_johnson = yeo_johnson, cols_outlier = cols_outliers)

        valid_prepr = data_preprocessing_test(valid_turned, y_valid, technical_vars, categories,
                                               drop_technical = True,
                                               attribute_list = attribute_list_model, 
                                               var_col = 'attribute',
                                               median = 'val_mediana',
                                               high_outlier = right_border, 
                                               low_outlier = left_border, scale = preproc, 
                                               yeo_johnson = yeo_johnson, cols_outlier = cols_outliers)

        if type(spline_split_points) is not type(None):
            train_prepr = for_splines_test(train_prepr, self.spline_split_points)
            test_prepr = for_splines_test(test_prepr, self.spline_split_points)
            valid_prepr = for_splines_test(valid_prepr, self.spline_split_points)
            features_no_tech = valid_prepr.columns
           

        if self.task == 'binary':
            preds_train = self.model.predict_proba(train_prepr)[:, 1]
            preds_test = self.model.predict_proba(test_prepr)[:, 1]
            preds_valid = self.model.predict_proba(valid_prepr)[:, 1]

            score_train = roc_auc_score(y_train, preds_train)*2-1
            score_test = roc_auc_score(y_test, preds_test)*2-1
            score_valid = roc_auc_score(y_valid, preds_valid)*2-1
        
            mpp_pred_valid = self.mpp.predict(data=valid_sample, 
                                              by_month=True,
                                              preprocessed=False)

        
        elif self.task == 'numeric':
            preds_train = self.model.predict(train_prepr)
            preds_test = self.model.predict(test_prepr)
            preds_valid = self.model.predict(valid_prepr)

            score_train = metrics.r2_score(y_train, preds_train)
            score_test = metrics.r2_score(y_test, preds_test)
            score_valid = metrics.r2_score(y_valid, preds_valid)
        
        train_turned1 = train_turned.copy()
        test_turned1 = test_turned.copy()
        valid_turned1 = valid_turned.copy()
        
        train_turned1['prediction'] = preds_train
        test_turned1['prediction'] = preds_test
        valid_turned1['prediction'] = preds_valid

        if self.it not in train_turned1.columns.to_list():
            train_turned1[self.it] = y_train
            test_turned1[self.it] = y_test
            valid_turned1[self.it] = y_valid
            
        if self.fit_model == True:
            scores_valid, vector_by_month_valid = self.receive_all_vector_and_score_by_month(train_turned1, 
                                                                                               valid_turned1, 
                                                                                               'prediction')
            scores_valid['score_train'] = score_train
            scores_valid['score_test'] = score_test
            scores_valid1 = scores_valid.reset_index()
            scores_valid1['T'] = scores_valid1['index']+1
            vector_by_month_valid['T'] = scores_valid1['T']
            vector_by_month_valid['score_test'] = score_test
        
            valid_t = pd.DataFrame(self.pr.transform(vector_by_month_valid[self.fin_colls]), columns = self.fin_colls)
            prediction_valid = self.r2_model.predict(valid_t)
            scores_valid1['prediction'] = prediction_valid
            scores_valid1['prediction_corrected'] = scores_valid1['prediction'] - 0.04
            
            if self.task == 'binary':            
                scores_valid1 = pd.merge(scores_valid1, mpp_pred_valid[[self.date_column, 'F-score', '1_rate', 
                                                                      'G-score', 'G2-score', 'G3-score']], 
                                                                       on = self.date_column, how = 'left')
                valid_t_mpp = pd.DataFrame(self.pr_mpp.transform(scores_valid1[self.mpp_columns]), 
                                                                       columns = self.mpp_columns)
                prediction_valid_mpp = self.r2_model_mpp.predict(valid_t_mpp)
                scores_valid1['prediction_MPP'] = prediction_valid_mpp-0.04
            
            return scores_valid1, vector_by_month_valid
        
        else:
            valid_prepr[self.date_column] = valid_turned[self.date_column]
            valid_prepr[self.it] = valid_turned[self.it]
            
            if self.task == 'binary':

                valid_prepr['prediction'] = self.model.predict_proba(valid_prepr[features_no_tech])[:, 1]
                
                scores_valid = self.by_month_gini(valid_prepr, 'prediction')
                scores_valid['score_train'] = score_train
                scores_valid['score_test'] = score_test
                scores_valid1 = scores_valid.reset_index()
                scores_valid1['T'] = scores_valid1['index']+1
            else:
                valid_prepr['prediction'] = self.model.predict(valid_prepr[features_no_tech])
                
                scores_valid = self.by_month_r2(valid_prepr, 'prediction')
                scores_valid['score_train'] = score_train
                scores_valid['score_test'] = score_test
                scores_valid1 = scores_valid.reset_index()
                scores_valid1['T'] = scores_valid1['index']+1
        
        if self.task == 'binary':            
            scores_valid1 = pd.merge(scores_valid1, mpp_pred_valid[[self.date_column, 'F-score', '1_rate', 
                                                                  'G-score', 'G2-score', 'G3-score']], 
                                                                   on = self.date_column, how = 'left')
            valid_t_mpp = pd.DataFrame(self.pr_mpp.transform(scores_valid1[self.mpp_columns]), columns = self.mpp_columns)
            prediction_valid_mpp = self.r2_model_mpp.predict(valid_t_mpp)
            scores_valid1['prediction_MPP'] = prediction_valid_mpp-0.04
        
        self.plot_ginis(valid=scores_valid1)
        
        return scores_valid1
    
def utility(fpr, tpr, tp_cost, fp_cost, fn_cost, tn_cost, good_rate):
    return tp_cost*tpr*good_rate+fn_cost*(1-tpr)*good_rate+fp_cost*fpr*(1-good_rate)+tn_cost*(1-fpr)*(1-good_rate)

def indifference_curve(X, y, model, tp_cost, fp_cost, fn_cost, tn_cost, good_rate, money_target=1):
    from sklearn.metrics import roc_curve, plot_roc_curve

    s=((1-good_rate)*(tn_cost-fp_cost))/(good_rate*(tp_cost-fn_cost))

    plot_roc_curve(model, X, y)
    x_axes=[0,1]
    y_axes=[1-1*s,1]
    if not money_target:
        x_axes=[0,s]
        y_axes=[0,1]
    return plt.plot(x_axes,y_axes)
