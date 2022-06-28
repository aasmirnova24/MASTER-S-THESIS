from operator import pos
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as po
from IPython.display import display

po.init_notebook_mode()
import copy
import itertools
import math
import sys
import warnings

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
import scikitplot as skplt
import seaborn as sns
import win32api
import win32job
import winerror
from IPython.display import Image
from scipy.stats import norm
from sklearn import metrics, tree
from sklearn.cluster import KMeans
from sklearn.metrics import (auc, average_precision_score, roc_auc_score,
                             roc_curve, recall_score, precision_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml_pipeline import hist_bad_rate



# При импорте модуля автоматически устанавливается ограничение 250ГБ на использование RAM
# Чтобы установить свой лимит:
# pipeline_utils.limit_memory(N * 1024 * 1024 * 1024) # N - память в ГБ


def mu(n_first_objects=15, dir_=dir(), globals_=globals()):
    '''
    MU = memory usage. Выводит список объектов в выполняемом .ipynb файле в 
    соответствии с их размерами. 
    Выводится только потребление памяти непосредственно приписываемое объекту, 
    без учета потребления памяти объектов, на которые он ссылается (shallow memory usage). 
    
    Вызов в ipynb-файле:
    
    from restict_memory import mu
    mu(n_first_objects=15, dir_=dir(), globals_=globals())
    
    Параметры:
    ---------
    n_first_objects: int
        Число отображаемых объектов
    dir_ : 
        Результат вызова функции dir() в ipynb файле
    globals:
        Результат вызова функции globals() в ipynb файле
    
    Возвращает:
    ----------
    Возвращает названия n_first_objects самых больших объектов в выполняемом 
    .ipynb файле вместе с их размерами
    '''
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    sizes = sorted([(x, round(sys.getsizeof(globals_.get(x))/(1024*1024),2)) for x in dir_ if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
    sizes = pd.DataFrame(data=[x[1] for x in sizes], index=[x[0] for x in sizes], columns=['Size, MB'])
    print(', '.join(list(sizes.head(n_first_objects).index)))
    return sizes.head(n_first_objects)


g_hjob = None

def create_job(job_name='', breakaway='silent'):
    hjob = win32job.CreateJobObject(None, job_name)
    if breakaway:
        info = win32job.QueryInformationJobObject(hjob,
                    win32job.JobObjectExtendedLimitInformation)
        if breakaway == 'silent':
            info['BasicLimitInformation']['LimitFlags'] |= (
                win32job.JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)
        else:
            info['BasicLimitInformation']['LimitFlags'] |= (
                win32job.JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        win32job.SetInformationJobObject(hjob,
            win32job.JobObjectExtendedLimitInformation, info)
    return hjob


def assign_job(hjob):
    global g_hjob
    hprocess = win32api.GetCurrentProcess()
    try:
        win32job.AssignProcessToJobObject(hjob, hprocess)
        g_hjob = hjob
    except win32job.error as e:
        if (e.winerror != winerror.ERROR_ACCESS_DENIED or
            sys.getwindowsversion() >= (6, 2) or
            not win32job.IsProcessInJob(hprocess, None)):
            raise
        warnings.warn('The process is already in a job. Nested jobs are not '
            'supported prior to Windows 8.')


def limit_memory(memory_limit):
    if g_hjob is None:
        return
    info = win32job.QueryInformationJobObject(g_hjob,
                win32job.JobObjectExtendedLimitInformation)
    info['ProcessMemoryLimit'] = memory_limit
    info['BasicLimitInformation']['LimitFlags'] |= (
        win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY)
    win32job.SetInformationJobObject(g_hjob,
        win32job.JobObjectExtendedLimitInformation, info)

assign_job(create_job())
memory_limit = 250 * 1024 * 1024 * 1024 # 400 GB
limit_memory(memory_limit)


def display_nicely(df,
                   maxrows=500,
                   floats=3):
    '''
    Показывает все строки (до maxrows) и столбцы датафрейма с числом знаков 
    после запятой равным floats.
    '''
    format_str = ''.join(['{:,.', str(floats), 'f}'])
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None, 
                           'display.float_format', format_str.format):
        display(df.iloc[:maxrows, :])
    return


def check_nans(df):
    '''Проверяет наличие NAN в колонках датасета'''
    a = 0
    for label, content in df.iteritems():
        if content.isnull().values.any():
            print(f'{label}: NANs!')
            a=1
    if a == 0:
        print('No NANs.')
    return


def get_pretty_dates(dates_vector=None):
    '''
    Переводит даты из формата '201905' в "May'19".

    Параметры:
    ----------
    dates_vector : pd.Series
        Pandas - серия с вектором дат.

    Возвращает:
    -----------
    Словарь формата {201905: "May'19", 201906: "Jun'19"}.
    '''
    months = sorted(dates_vector.astype('int').unique())
    month_dict = {'01' : 'Jan',
                  '02' : 'Feb',
                  '03' : 'Mar',
                  '04' : 'Apr',
                  '05' : 'May',
                  '06' : 'Jun',
                  '07' : 'Jul',
                  '08' : 'Aug',
                  '09' : 'Sep',
                  '10' : 'Oct',
                  '11' : 'Nov',
                  '12' : 'Dec'}

    pretty_dates ={x : str(month_dict[str(x)[-2:]] + '\'' + str(x)[2:-2]) for x in months}
    return pretty_dates


def drop_unnamed(df):

    '''
    Находит и удаляет колонки типа 'Unnamed: 0'.

    Параметры:
    ---------
    data: pandas DataFrame
        Датафрейм с данными

    Возвращает:
    ----------
    Датасет без колонок типа 'Unnamed: 0'. Переводит все названия колонок в верхний регистр.
    '''
    df.columns = list(map(str.upper, df.columns))
    unnamed_list = df.loc[:, df.columns.str.contains('^UNNAMED')].columns.tolist()
    
    if unnamed_list:
        df.drop(unnamed_list, axis=1, inplace=True)
        print ('Функцией drop_unnamed удалены колонки:', unnamed_list)
    return df


def append_for_deletion(all_columns=None,
                        all_columns_bin=None,
                        deleted=None,
                        list_to_delete=[],
                        reason='',
                        overwrite=False,
                        necessary_fields=[],
                        ignore_whitelist = False,
                        whitelist=pd.Series(dtype=str)
                        ):

    '''
       Собирает серию формата 'переменная - список причин удаления' для 
       хранения списка удаленных переменных и сборки воронки.
       Позволяет полностью удалить и перезаписать изменения, внесенные на любом 
       этапе отбора.

       Параметры:
       ---------
       data : pd.DataFrame
            Исходные данные. Используется для получения списка колонок, 
            сам датафрейм в функции не меняется.
       deleted : None или pd.Series
            Серия с переменными на удаление. Если функция применяется впервые 
            оставить deleted=None.
       list_to_delete : list
            Лист переменных, добавляемых в серию на данном этапе.
       reason : str
            Причина удаления добавляемых переменных.
       overwrite : boolean
            При overwrite=True в случае если проверка по данной причине уже 
            проводилась, функция удаляет результы прошлого отбора по 
            данной причине и записывает результаты нового.
       necessary_fields: list
            Список необходимых служебных признаков. Переменные из этого списка 
            не будут добавлены на удаление.
       whitelist: pandas Series
            Серия с whitelist переменными.

       Возвращает:
       Pandas - серию формата 'переменная - список причин' с переменными на 
       удаление.

    '''
    list_to_delete = list(map(str.upper, list_to_delete))
    # Убираем из листа на удаление, колонки, которых изначально нет в датасете.
    if all_columns_bin is not None:
        all_columns_bin = list(map(str.upper, all_columns_bin))
        list_to_delete = [x for x in list_to_delete if x in all_columns_bin]
    else:
        all_columns = list(map(str.upper, all_columns))
        list_to_delete = [x for x in list_to_delete if x in all_columns]
    
    # Если серии с переменными на удаление еще нет - создаем.
    if deleted is None:
        del_feat = pd.Series(name='deleted', dtype='object')
        deletion_reasons = []
    # Если серия есть - делаем копию.
    else:
        del_feat = deleted.copy(deep=True)
        deletion_reasons = [item for sublist in del_feat.values for item in sublist]
    # Проверяем является ли проверка повторной
    if (reason in deletion_reasons) and (overwrite == False):
        raise ValueError(f'Проверка на {reason} уже проводилась. Используйте флаг overwrite = True, чтобы удалить результаты предыдущей проверки и сохранить новые.')

    elif (reason in deletion_reasons) and (overwrite == True):
        # Удаляем результаты предыдущей проверки при overwrite == True:
        for feature in del_feat.index:
            if reason in del_feat[feature]:
                del_feat[feature] = tuple([x for x in del_feat[feature] if x != reason])
                # del_feat[feature].remove(reason)
                if not del_feat[feature]:
                    del_feat.drop(feature, inplace=True)

    # Записываем переменную и причину удаления
    i = 0
    j = ''
    for feature in [x for x in list_to_delete if x not in necessary_fields]:
        if feature in whitelist.index:
            if ignore_whitelist == False:
                print(f'Whitelist-признак {feature} не прошел отбор. Переменная не удалена.')
                whitelist[feature] = tuple([*whitelist[feature], reason])
            elif ignore_whitelist == True:
                print(f'Whitelist-признак {feature} не прошел отбор. Переменная будет удалена (ignore_whitelist == True).')
                whitelist[feature] = tuple([*whitelist[feature], reason+'(WHITELIST_IGNORED)'])
            if '' in whitelist[feature]:
                whitelist[feature] = tuple(x for x in whitelist[feature] if x != '')
            j='\n'

        if (feature not in whitelist.index) or (ignore_whitelist == True):
            if feature in del_feat.index:
                del_feat[feature] = tuple([*del_feat[feature], reason])
            else:
                del_feat[feature] = tuple([reason])
                i+=1

    if all_columns_bin is not None:
        left = len(set(all_columns_bin)-set((del_feat.index)))
    else:
        left = len(set(all_columns)-set(del_feat.index))

    print(f'{j}{len(list_to_delete)} переменных записано на удаление, из них новых: {i}.')
    print(f'После удаления в датасете останется {left} переменных (включая necessary_fields).')
    return del_feat, whitelist


def remove_from_deletion(deleted=None, reason=None):
    '''
       Удаляет изменения, внесенные по обозначенной причине из списка на удаление.
    '''
    del_feat = deleted.copy(deep=True)
    for feature in del_feat.index:
        if reason in del_feat[feature]:
            # del_feat[feature].remove(reason)
            del_feat[feature] = tuple([x for x in del_feat[feature] if x != reason])
            if not del_feat[feature]:
                del_feat.drop(feature, inplace=True)
    return del_feat


def funnel_func(deleted=None, 
                all_columns=[], 
                all_columns_bin=[], 
                necessary_fields=None,
                binning_step_number = None,
                variables_desc=None, 
                changes = None,
                plots=False,
                collapse = False,
                stages = {
                        'Валидность и заполненность': ['Blacklist', 'Все пропуски','Одно значение','>97% пропусков','Пропуски и одно значение','1% = 99%','Empty time ranges'],
                        'Стабильность': ['Unstable'],
                        'Таргет корреляция': ['Stats with target'],
                        'Матрица корреляций': ['Correlations'],
                        'Биннинг категориальных': ['Биннинг'],
                        'Валидация после биннинга': ['1% = 99% BIN'],
                        'Таргет корреляция после биннинга': ['Stats with target BIN'],
                        'Матрица корреляций после биннинга': ['Correlations BIN'],
                        'Two-forest': ['2forests'],
                        'VIF':['vif'],
                        'Boruta-Shap' : ['Boruta-Shap', 'Boruta', 'Shap'],
                        'p-value': ['p-value'],
                        'Whitelist': ['whitelist']}):
    '''
    Собирает отчет по воронке, выводит график.
    
    Параметры:
    ---------
    deleted: pd.Series
        Результат применения функции append_for_deletion. 
    all_columns: list
        Исходный список колонок.
    binned_list: list
        Список переменных после биннинга.
    variables_desc: 
        Датафрейм с иснформацией об источниках признаков. Ключевые поля - 
        'Переменная', 'Источник'.
    plots:boolean
        Выводить ли графики воронки.
    
    Возвращает:
    ----------
    Датафрейм со статистиками по удаленным и добавленным на биннинге признакам.
    '''
    if all_columns is not None:
        all_columns = [x for x in all_columns if x not in necessary_fields]
    if all_columns_bin is not None:
        all_columns_bin = [x for x in all_columns_bin if x not in necessary_fields]

    variables_desc = variables_desc.copy()
    variables_desc.drop_duplicates(inplace=True, subset=['Переменная'], keep='last')
    variables_desc.set_index('Переменная', inplace=True)
    features = list(deleted.index)

    # fun_1 - расширенная версия серии deleted с указанием источника
    fun_1 = pd.DataFrame(data = deleted.values, columns=['reasons'], index=features)
    fun_1['Reason'] = [deleted[ind][0] for ind in features]
    
    # Достаем переменные для которых нет данных об источнике в датафрейме variables_desc (aka var_info)
    for x in [x for x in [*all_columns, *all_columns_bin] if x not in variables_desc.index]:
        # Если переменная получена в результате биннига - подтягиваем название исходной переменной
        if changes is not None:
            if changes.loc[changes['new variable'] == x, 'genuine variable'].values:
                variables_desc.loc[x, 'Источник'] = variables_desc.loc[changes.loc[changes['new variable'] == x, 'genuine variable'].values[0], 'Источник']
            else:
                variables_desc.loc[x, 'Источник'] = 'Other'
        else:
            variables_desc.loc[x, 'Источник'] = 'Other'
        
    fun_1['Источник'] = [variables_desc.loc[feature, 'Источник'] for feature in features]
    steps = list(fun_1['Reason'].unique())

    # fun_2 - датафрейм с воронкой
    fun_2 = pd.DataFrame(columns=['Шаг', 'Было', 'Удалено', 'Осталось'])
    
    columns = all_columns
    source_vc = pd.Series(index=all_columns, data=[variables_desc.loc[feature, 'Источник'] for feature in all_columns]).value_counts()
    sources = list(source_vc.index)
    fun_2.loc[0, 'Шаг'] = 'Data'
    fun_2.loc[0, 'Было'] = len(columns)
    fun_2.loc[0, 'Удалено'] = 0
    fun_2.loc[0, 'Осталось'] = len(columns)
    for source in source_vc.index:
        fun_2.loc[0, source + ' (Было)'] = source_vc[source]
        fun_2.loc[0, source + ' (Удалено)'] = 0
        fun_2.loc[0, source + ' (Осталось)'] = source_vc[source]

    if (all_columns_bin) and (binning_step_number is not None):
        steps.insert(binning_step_number-1, 'Биннинг')
    for i, step in enumerate(steps):
        fun_2.loc[i+1, 'Шаг'] = step
        fun_2.loc[i+1, 'Было'] = len(columns)
        if step == 'Биннинг':
            columns = all_columns_bin
            source_vc = pd.Series(index=all_columns_bin, data=[variables_desc.loc[feature, 'Источник'] for feature in all_columns_bin]).value_counts()
            sources.extend(list(source_vc.index))
            sources = set(sources)
        del_step = [x for x in features if step in deleted[x]]
        columns = [x for x in columns if x not in del_step]
        fun_2.loc[i+1, 'Осталось'] = len(columns)
        fun_2.loc[i+1, 'Удалено'] = fun_2.loc[i+1, 'Было'] - fun_2.loc[i+1, 'Осталось']
        
        # fun 4 - серия 
        fun_4 = fun_1[['Reason', 'Источник']].loc[fun_1['Reason'] == step].groupby(['Источник']).count().iloc[:,0]
        
        for source in source_vc.index:
            if source not in fun_4.index:
                fun_4[source] = 0
            fun_2.loc[i+1, source + ' (Было)'] = int(source_vc[source])
            fun_2.loc[i+1, source + ' (Удалено)'] = int(fun_4[source])
            source_vc[source] = source_vc[source] - int(fun_4[source])
            fun_2.loc[i+1, source + ' (Осталось)'] = source_vc[source]
        fun_2.fillna(0, inplace=True)

    if collapse == True:
        # Схлопываем воронку в соответствии со стадиями в stages
        # fun_3 - схлопнутый fun_2
        fun_3 = pd.DataFrame(index = range(len(stages)+1), columns = fun_2.columns)
        fun_3.iloc[0, :] = fun_2.iloc[0, :].copy()

        for i, stage in enumerate(stages.keys()):
            smm = 0
            fun_3.loc[i+1,'Было'] = fun_2.loc[fun_2['Шаг'].isin(stages[stage]), 'Было'].max()
            fun_3.loc[i+1,'Удалено'] = fun_2.loc[fun_2['Шаг'].isin(stages[stage]), 'Удалено'].sum()
            fun_3.loc[i+1,'Осталось'] = fun_2.loc[fun_2['Шаг'].isin(stages[stage]), 'Осталось'].min()
            fun_3.loc[i+1,'Шаг'] = stage

            for source in sources:# source_vc.index:
                fun_3.loc[i+1, source + ' (Было)'] = fun_2.loc[fun_2['Шаг'].isin(stages[stage]), source + ' (Было)'].max()
                fun_3.loc[i+1, source + ' (Удалено)'] = fun_2.loc[fun_2['Шаг'].isin(stages[stage]), source + ' (Удалено)'].sum()
                fun_3.loc[i+1, source + ' (Осталось)'] = fun_2.loc[fun_2['Шаг'].isin(stages[stage]), source + ' (Осталось)'].min()
                smm = smm + fun_3.loc[i+1, source + ' (Удалено)']
            # print(stage, smm)
            fun_3.dropna(axis=0, subset=['Было', 'Осталось'], how='all', inplace=True)
            fun_3.fillna(0, inplace=True)

    else:
        fun_3 = fun_2

    fun_3['Удалено %'] = (fun_3['Удалено']/fun_3['Было']*100).round(0)
    fun_3[[ 'Было', 'Удалено', 'Удалено %', 'Осталось']] = fun_3[[ 'Было', 'Удалено', 'Удалено %', 'Осталось']].astype(int)
    # display(fun_3)
    if plots == True:
        fig = go.Figure()
        for source in source_vc.index:
            fig.add_trace(go.Funnel(
                name = source,
                orientation = "h",
                y = fun_3['Шаг'],
                x = fun_3[source + ' (Осталось)'],
                textposition = "inside"
            ))
        fig.update_layout(title="Осталось")

        fig.show()

        fig = go.Figure()
        for source in source_vc.index:

            fig.add_trace(go.Funnel(
                name = source,
                orientation = "h",
                y = fun_3['Шаг'],
                x = fun_3[source + ' (Удалено)'],
                textposition = "inside"
                        ))
        fig.update_layout(title="Удалено")
        fig.show()

    return fun_3[['Шаг', 'Было', 'Удалено', 'Удалено %', 'Осталось']]


def show_sources(features, changes, variables_desc):

    '''
    Принимает список переменных и выводит барплот с числом переменных в данном 
    списке для каждого из имеющихся источников.

    Параметры:
    ---------
    features: iterable
        Список переменных.
    changes: pandas DataFrame
        Датафрейм со списком новых и исходных переменных. Ключевые поля - 
        'new variable', 'genuine variable'.
    variables_desc: pandas DataFrame
        Датафрейм с иснформацией об источниках признаков. Ключевые поля - 
        'Переменная', 'Источник'.

    Возвращает:
    ----------
    Показывает график, возвращает датафрейм с источниками переменных и датафрейм 
    с числом переменных по источнику.
    '''
    
    variables_desc = variables_desc.copy()
    variables_desc.set_index('Переменная', inplace=True)

    # Датафрейм fun_1 - расширенная версия серии deleted с указанием источника
    sources_df = pd.DataFrame(columns=['Источник'], index=features)
    
    # Достаем переменные для которых нет данных об источнике в датафрейме variables_desc (aka var_info)
    for x in [x for x in features if x not in variables_desc.index]:
        # Если переменная получена в результате биннига - подтягиваем название исходной переменной
        if changes is not None:
            if changes.loc[changes['new variable'] == x, 'genuine variable'].values:
                variables_desc.loc[x, 'Источник'] = variables_desc.loc[changes.loc[changes['new variable'] == x, 'genuine variable'].values[0], 'Источник']
            else:
                variables_desc.loc[x, 'Источник'] = 'Other'
        else:
            variables_desc.loc[x, 'Источник'] = 'Other'
            
    sources_df['Источник'] = [variables_desc.loc[feature, 'Источник'] for feature in features]
    sources_df.reset_index(inplace=True)
    counts = sources_df.groupby('Источник').count()
    f = plt.figure(figsize=(10, 6))
    plt.barh(list(counts.index), counts.iloc[:,0].values)

    for i, v in enumerate(counts.iloc[:,0].values):
        plt.text(v , i , ' '+str(v),  va='center', fontweight='bold')
    
    plt.show()
    return sources_df, counts


def time_ranges_plot(dfs_dict, date_column):
    '''
    Показывает график с временными диапазонами в приведенных датасетах.

    Параметры:
    ---------

    dfs_dict : dict
        Словарь датафреймов.

    comment : str

    Возвращает:
    ----------
    Выводит график.

    '''
    sns.set()
    plt.figure(figsize = (16, 1))

    for df in dfs_dict.keys():
        months = sorted(dfs_dict[df][date_column].astype('int').unique())
        pretty_dates = get_pretty_dates(dfs_dict[df][date_column])
        if len(months) > 1:
            plt.plot([pretty_dates[x] for x in months], [df for x in months], lw=5)
        else:
            plt.scatter([pretty_dates[x] for x in months], [df for x in months], linewidths=5, marker='_')

        plt.tick_params(labelsize=14, axis = 'y')
        plt.title('Рассматриваемые временные диапазоны:', fontdict={'fontsize':15})
    plt.show()
    return


def count_data(dfs_dict=None,
               bad_class=0,
               target=None,
               task='binary'):
    '''
    Считает число наблюдений и Good/Bad rate

    Параметры:
    ----------
    dfs_dict : dict
        Словарь с датафреймами данных вида {'Train': X_train, 
                                            'Test': X_test, 
                                            'OOT': oot}.
    bad_class: 0 или 1
        Какой класс считать нежелательным. Используется при task == 'binary'.
    target: str
        Целевая переменная.
    task: 'multiclass', 'binary' или 'numeric'
        Тип задачи.

    Возвращает:
    ---------
    Датафрейм со статистиками.
    
    '''
    if task == 'binary':
        if bad_class == 0:
            good_class = 1
        elif bad_class == 1:
            good_class = 0
        else:
            raise ValueError("Значение bad_class должно быть 0 или 1.")

        count_df = pd.DataFrame(columns=['#','Bad', 'Good', 'Good rate', 'Bad rate'], index=['All', *list(dfs_dict.keys())])

        for dataset_key in dfs_dict.keys():
            count_df.loc[dataset_key, 'shape'] = ' x '.join([str(x) for x in dfs_dict[dataset_key].shape])

            vc = dfs_dict[dataset_key][target].value_counts()
            count_df.loc[dataset_key, '#'] = len(dfs_dict[dataset_key])
            count_df.loc[dataset_key, 'Bad'] = vc.loc[bad_class]
            count_df.loc[dataset_key, 'Good'] = vc.loc[good_class]
            count_df.loc[dataset_key, 'Good rate'] = count_df.loc[dataset_key,'Good']/count_df.loc[dataset_key,'#']
            count_df.loc[dataset_key, 'Bad rate'] =  count_df.loc[dataset_key,'Bad']/count_df.loc[dataset_key,'#']

        count_df.loc['All', 'Bad'] = count_df.loc[:, 'Bad'].sum()
        count_df.loc['All', 'Good'] = count_df.loc[:, 'Good'].sum()
        count_df.loc['All', '#'] = count_df.loc[:, '#'].sum()
        count_df.loc['All', 'Bad rate'] = count_df.loc['All', 'Bad']/count_df.loc['All', '#']
        count_df.loc['All', 'Good rate'] = count_df.loc['All', 'Good']/count_df.loc['All', '#']
        count_df.fillna(' ', inplace=True)
        display(count_df.style.format("{:.2%}", subset=pd.IndexSlice[:, ['Good rate', 'Bad rate']]))

    elif task == 'numeric':
        count_df = pd.DataFrame(columns=['#', 'min', 'mean', 'max'], index=['All', *list(dfs_dict.keys())])
        sum_num = 0
        for dataset_key in dfs_dict.keys():
            count_df.loc[dataset_key, 'shape'] = ' x '.join([str(x) for x in dfs_dict[dataset_key].shape])
            count_df.loc[dataset_key, '#'] = len(dfs_dict[dataset_key])
            count_df.loc[dataset_key, 'min'] = dfs_dict[dataset_key][target].min()
            count_df.loc[dataset_key, 'mean'] = dfs_dict[dataset_key][target].mean()
            count_df.loc[dataset_key, 'max'] = dfs_dict[dataset_key][target].max()
            sum_num += dfs_dict[dataset_key][target].sum()
        count_df.loc['All', '#'] = count_df.loc[dfs_dict.keys(), '#'].sum()
        count_df.loc['All', 'min'] = count_df.loc[dfs_dict.keys(), 'min'].min()
        count_df.loc['All', 'mean'] = count_df.loc[dfs_dict.keys(), 'mean'].mean()
        count_df.loc['All', 'max'] = count_df.loc[dfs_dict.keys(), 'max'].max()
        display(count_df.fillna(' '))


    elif task == 'multiclass':
        unique_target = sorted(set([y for x in [list(dfs_dict[dataset_key][target].unique()) for dataset_key in dfs_dict.keys()] for y in x]))
        columns_num = ['#' + str(x) for x in unique_target]
        columns_rate = [str(x) + ' rate' for x in unique_target]
        count_df = pd.DataFrame(columns = ['#', *columns_num, *columns_rate] , index=['All', *list(dfs_dict.keys())])

        for dataset_key in dfs_dict.keys():
            count_df.loc[dataset_key, '#'] = len(dfs_dict[dataset_key])
            count_df.loc[dataset_key, 'shape'] = ' x '.join([str(x) for x in dfs_dict[dataset_key].shape])

            vc = dfs_dict[dataset_key][target].value_counts()
            for target_value in sorted(dfs_dict[dataset_key][target].unique()):
                count_df.loc[dataset_key, '#' + str(target_value)] = vc[target_value]
                count_df.loc[dataset_key, str(target_value) + ' rate'] = vc[target_value]/vc.sum()

        for target_value in unique_target:
            count_df.loc['All', '#' + str(target_value)] = count_df.loc[dfs_dict.keys(), '#' + str(target_value)].sum()
        for target_value in unique_target:
            count_df.loc['All', str(target_value) + ' rate'] = count_df.loc['All', '#' + str(target_value)]/count_df.loc['All', columns_num].sum()
        count_df.loc[dfs_dict.keys(), '#'] = count_df.loc[dfs_dict.keys(), '#'].astype('int')
        count_df.loc['All', '#'] = count_df.loc[dfs_dict.keys(), '#'].sum()

        def form(val):
            if val == -999:
                fr = '-'
            else:
                fmt = '{:.' + str(2) + '%}'
                fr = fmt.format(val)
            return fr

        def form2(val):
            if val == -999:
                fr = 0
            else:
                fr = '{:d}'.format(val)
            return fr
        count_df['shape'].fillna('', inplace=True)

        vmin = count_df[columns_num].min().min()
        vmax = count_df[columns_num].max().max() + 100

        display(count_df.fillna(-999).style
                        .background_gradient(cmap='Blues', subset=columns_num, vmin=vmin, vmax=vmax)
                        .format(form, subset=pd.IndexSlice[:, columns_rate])
                        .format(form2, subset=pd.IndexSlice[:, columns_num]))

    return


def model_stats(clf, data_dict, task='binary', round_to=2, multioutput_n=False):
    '''
    Считает датафрейм с метриками для модели
    Для пороговых метрик порог стандартный.

    Параметры:
    ----------
    clf : model
        Обученный классификатор
    data_dict : dict
        Словарь формата {'Train': [X_train, y_train], 'Test': [X_test, y_test]}
    task: str
        Тип задачи binary или multiclass
    round_to: int
        Округление результатов, число знаков после запятой

    Возвращает:
    ----------
    Показывает и возвращает датафрейм с метриками.
    '''

    def form(val):
        if val == -999:
            fr = '-'
        else:
            fmt = '{:.' + str(round_to) + '%}'
            fr = fmt.format(val)
        return fr

    df_keys =  list(data_dict.keys())

    if task == 'binary':
        mod_st = pd.DataFrame(columns = df_keys, index=['APS', 'ROC AUC', 'Gini'])

        if not sorted(data_dict[df_keys[0]][1].unique()) == [0,1]:
            vc = data_dict[df_keys[0]][1].value_counts()
            pos_label = vc.loc[vc == vc.iloc[-1]].index[0]
            print(f'установлен pos_label для recall и precision score = {pos_label}')
        else:
            pos_label=1

        for df_key in df_keys:
            if multioutput_n is not False:
                y_pred = clf.predict(data_dict[df_key][0])[:, multioutput_n]
                y_pred_proba = clf.predict_proba(data_dict[df_key][0])[multioutput_n][:,1]
            else:
                y_pred = clf.predict(data_dict[df_key][0])
                y_pred_proba = clf.predict_proba(data_dict[df_key][0])[:,1]
            
            if len(data_dict[df_key][1].unique()) == 1:
                print(f'В таргете датасета {df_key} все значения одинаковые, метрики не рассчитаны.')
            else:
                mod_st.loc['Recall', df_key] = recall_score(data_dict[df_key][1], y_pred, pos_label=pos_label)
                mod_st.loc['Precision', df_key] = precision_score(data_dict[df_key][1], y_pred, pos_label=pos_label, zero_division=0)
                mod_st.loc['APS', df_key] = average_precision_score(data_dict[df_key][1], y_pred_proba, pos_label=pos_label)
                mod_st.loc['ROC AUC', df_key] = roc_auc_score(data_dict[df_key][1], y_pred_proba)
                mod_st.loc['Gini', df_key] = 2*mod_st.loc['ROC AUC', df_key]-1
            
        # mod_st = mod_st.loc[['Recall', 'Precision','Gini'], :]
        vmin = mod_st.min().min() - 0.2
        vmax = mod_st.max().max() + 0.2

        display(mod_st.fillna(-999).style
                .background_gradient(cmap='Greens', vmin=vmin, vmax=vmax) 
                .format(form))

    if task == 'multiclass':
        try:
            if multioutput_n is not False:
                unique_target = clf.classes_[multioutput_n]
            else:
                unique_target = clf.classes_
        except:
            unique_target = sorted([set([y for x in [list(pd.Series(data_dict[df_key][1]).unique()) for df_key in df_keys] for y in x])])
        
        iterables = [[*unique_target, 'Macro', 'Weighted'], df_keys]
        mod_st = pd.DataFrame(columns=['Gini', 'Recall', 'Precision', '#'],
                              index=pd.MultiIndex.from_product(iterables))
       
        for df_key in df_keys:
            if multioutput_n is not False:
                y_pred_proba = clf.predict_proba(data_dict[df_key][0])[multioutput_n]
                y_pred = clf.predict(data_dict[df_key][0])[:, multioutput_n]

            else:
                y_pred_proba = clf.predict_proba(data_dict[df_key][0])
                y_pred = clf.predict(data_dict[df_key][0])

            for i, target_value in enumerate(unique_target):
                y_ovr_true = pd.Series(index=data_dict[df_key][1].index)
                y_ovr_true.loc[data_dict[df_key][1] == target_value] = 1
                y_ovr_true.loc[data_dict[df_key][1] != target_value] = 0

                y_ovr_pred = pd.Series(index=range(len(y_pred)))
                y_ovr_pred.loc[y_pred == target_value] = 1
                y_ovr_pred.loc[y_pred != target_value] = 0

                y_ovr_proba = [row[i] for row in y_pred_proba]

                if len(y_ovr_true.unique()) == 2:

                    # mod_st.loc[(str(target_value), [df_key]), 'APS'] = average_precision_score(y_ovr_true, y_ovr_pred)
                    mod_st.loc[(target_value, [df_key]), 'Gini'] = 2*roc_auc_score(y_ovr_true, y_ovr_proba)-1
                    mod_st.loc[(target_value, [df_key]), 'Recall'] = recall_score(y_ovr_true, y_ovr_pred)
                    mod_st.loc[(target_value, [df_key]), 'Precision'] = precision_score(y_ovr_true, y_ovr_pred, zero_division=0)

                else:
                    print(f'В таргете датасета {df_key} для класса {target_value} все значения одинаковые, Метрики не определены.')

                mod_st.loc[(target_value, [df_key]), '#'] = len(data_dict[df_key][1].loc[data_dict[df_key][1]==target_value])
            
            try:
                mod_st.loc[('Macro', [df_key]), 'Gini'] = 2*roc_auc_score(data_dict[df_key][1], y_pred_proba, average="macro", multi_class="ovr", labels = unique_target)-1
                mod_st.loc[('Macro', [df_key]), 'Recall'] = recall_score(data_dict[df_key][1], y_pred, average="macro")
                mod_st.loc[('Macro', [df_key]), 'Precision'] = precision_score(data_dict[df_key][1], y_pred, average="macro", zero_division=0)

                mod_st.loc[('Weighted', [df_key]), 'Gini'] = 2*roc_auc_score(data_dict[df_key][1], y_pred_proba, average="weighted", multi_class="ovr", labels = unique_target)-1
                mod_st.loc[('Weighted', [df_key]), 'Recall'] = recall_score(data_dict[df_key][1], y_pred, average="weighted")
                mod_st.loc[('Weighted', [df_key]), 'Precision'] = precision_score(data_dict[df_key][1], y_pred, average="weighted", zero_division=0)


            except:
                print(f'В таргете датасета {df_key} все значения одинаковы, Метрики не определены.')

            mod_st.loc[('Macro', [df_key]), '#'] = len(data_dict[df_key][1])
            mod_st.loc[('Weighted', [df_key]), '#'] = len(data_dict[df_key][1])


        vmin = mod_st.loc[:, ['Gini', 'Recall', 'Precision']].min().min() - 0.2
        vmax = mod_st.loc[:, ['Gini', 'Recall', 'Precision']].max().max() + 0.2

        display(mod_st.fillna(-999).style
                .background_gradient(cmap='Greens', vmin=vmin, vmax=vmax, subset=['Gini', 'Recall', 'Precision'])
                .background_gradient(cmap='Reds_r', vmin=0, vmax=30, subset=['#']) 
                .format(form, subset=pd.IndexSlice[:, ['Gini', 'Recall', 'Precision']]))

    return mod_st


def show_descriptions(columns, var_dict,var_desc):
    '''
    Показывает описания переменных

    columns : list

    var_dict : dict

    var_desc : dict
    '''
    for i, x in enumerate(columns, start=1):
        try:
            print (i, x, '.'*(30-len(x)-math.floor(math.log(i, 10))), var_desc[x].strip().capitalize())
        except:
            try:
                print (i, x, '.'*(30-len(x)-math.floor(math.log(i, 10))), var_desc[var_dict[x]].strip().capitalize())
            except:
                print (i, x, '.'*(30-len(x)-math.floor(math.log(i, 10))))
    return


def roc_lift_gain(model, data, figsize=(18, 16)):
    matplotlib.rc_file_defaults()

    f = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 4)
    ax1 = f.add_subplot(gs[:2, :2])
    gini_dict = {}
    for dataset in data.keys():
        
        fpr, tpr, _ = roc_curve(data[dataset][1], model.predict_proba(data[dataset][0])[:, 1])
        gini = 2 * auc(fpr, tpr) - 1
        gini_dict[dataset] = gini
        ax1.plot(fpr, tpr, label = dataset + ', GINI = ' + '{:.2%}'.format(gini))

        plt.legend(loc= 'lower right', fontsize=20)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.title('ROC curve', fontsize=15)

    for i, dataset in enumerate([x for x in data.keys() if x != 'Train']):
        f.add_subplot(gs[0, i+2])
        skplt.metrics.plot_lift_curve(data[dataset][1], model.predict_proba(data[dataset][0]), title='Lift Curve ' + dataset, ax=plt.gca())

        plt.legend(loc=1)
        f.add_subplot(gs[1, i+2])
        skplt.metrics.plot_cumulative_gain(data[dataset][1], model.predict_proba(data[dataset][0]), title ='Cummulative Gains Curve ' + dataset, ax=plt.gca())
   
    f.tight_layout()
    plt.show()
    return


def binary_hists(y_true=None, y_test_pred=None, dict_target=None, target=None, figsize=(10,7)):

    matplotlib.rc_file_defaults()

    f = plt.figure(figsize=figsize)
    ax1 = f.add_subplot(2,2,1)
    ax2 = f.add_subplot(2,2,2)
    ax3 = f.add_subplot(2,2,3)
    ax4 = f.add_subplot(2,2,4)

    test_pred = pd.DataFrame(y_test_pred)
    test_pred.rename(columns = {0:'Prediction'}, inplace = True)

    # Hist 0
    test_pred.loc[list(y_true == 0), 'Prediction'].hist(bins = 500, color = 'green',edgecolor='none', alpha = 0.8, ax=ax1)
    ax1.set_title('Histogram ' + dict_target[0].capitalize() +  '(0)')

    # Hist 1
    test_pred.loc[list(y_true == 1), 'Prediction'].hist(bins = 500, color = 'deeppink',edgecolor='none', alpha = 0.8, ax=ax2)
    ax2.set_title('Histogram '+ dict_target[1].capitalize() + '(1)')

    # Bin hist
    sns.distplot(test_pred.loc[list(y_true == 0), 'Prediction'], norm_hist = True, hist = True,
                 hist_kws={'alpha':.75}, kde_kws={'linewidth':3}, color="g", label=dict_target[0].capitalize(), ax=ax3)
    sns.distplot(test_pred.loc[list(y_true == 1), 'Prediction'], norm_hist = True, hist = True,
                 hist_kws={'alpha':.35}, kde_kws={'linewidth':3}, color="deeppink", label=dict_target[1].capitalize(), ax=ax3)
    ax3.set_title('Распределение Good/ Bad')
    ax3.legend()
    
    # Model PD/Goodrate
    hbr_trans = hist_bad_rate(test_pred['Prediction'], y_true, N=10, method = 'pieces', fillnas=None)
    hbr_trans['x'] = [str(i) for i in hbr_trans.index.categories]
    x_ticks = [c for i, c in enumerate(hbr_trans['x'].values) if i%15==0]
    hbr_trans = hbr_trans.reset_index(drop=True)
    hbr_trans['pers'] = hbr_trans['Prediction']/sum(hbr_trans['Prediction'])
    c = 'Prediction'
    cc = c + '_mean'
    hbr_trans['xx'] = [str(round(i, 3)) for i in hbr_trans[cc]]
    ax4.plot(hbr_trans['xx'], hbr_trans[target], 'o-', label=target, color = 'deeppink')
    plt.ylabel((dict_target[1]+'rate').capitalize(), fontsize=13)
    plt.xlabel('Model PD', fontsize=13)
    plt.twinx()
    plt.bar(hbr_trans['xx'], hbr_trans['pers'], color='darkgray', alpha=0.5, label='Sample Rate', )
    x_ticks = hbr_trans['xx'].to_list()
    plt.xticks(x_ticks)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def bootscores_concat(mtrcs, boot_scores, rnd=1):
    scores_concated = pd.DataFrame()
    for mtr in mtrcs:
        temp_scores = (pd.concat([x[mtr] for x in boot_scores.values()], axis=1, keys = ['_'.join([mtr, x]) for x in boot_scores.keys()])*100).round(rnd)
        deltas = list(itertools.combinations(temp_scores.columns, 2))
        for delta in deltas:
            temp_scores[''.join(['\u0394', mtr, ' ', delta[0][len(mtr)+1:], '-', delta[1][len(mtr)+1:]])] = temp_scores[delta[0]] - temp_scores[delta[1]]
        scores_concated = pd.concat([scores_concated, temp_scores], axis=1)
    scores_concated = scores_concated[sorted(scores_concated.columns)]
    return scores_concated


def capitalize_keys(d):
        result = {}
        for key, value in d.items():
            upper_key = key.upper()
            result[upper_key] = value
        return result


def by_month(data=None,
             date_column=None,
             target=None,
             target_dict=None,
             proba = 'calib_scores'):
    '''

    data: dict
          e.g. {'test': data_called} 
    date_column: str    
          e.g. 'month_call_rev', даты в формате #yyyymm, колонка с датами - числовая, например 202103 

    target: str

    target_dict={'good': 1, 'bad': 0},

    proba=str, e.g. 'score'

    Возвращает:
    ---------
    Датафрейм со значениями gini по месяцам
    '''

    data = capitalize_keys(data)
    months = sorted(set([y for x in [list(data[x][date_column].unique()) for x in data.keys()] for y in x]))
    scores = pd.DataFrame(index=months, columns = ['Month','Number', 'Good rate', 'Bad rate'])
    number_columns = []
    for dataset_key in data.keys():
        data[dataset_key][date_column] = data[dataset_key][date_column].round(0).astype(int)
        months_ = data[dataset_key][date_column].unique()
        
        for month in months_:
            data_month = data[dataset_key].loc[data[dataset_key][date_column] == float(month)].copy()
            scores.loc[month,  dataset_key] = round(roc_auc_score(data_month[target], data_month[proba])*2-1, 6)
            if dataset_key in ['TEST', 'OOT']:
                number_columns.append(dataset_key)
                scores.loc[month,  'Good rate'] = len(data_month.loc[data_month[target] == target_dict['good']])/len(data_month)
                scores.loc[month,  'Bad rate'] = len(data_month.loc[data_month[target] == target_dict['bad']])/len(data_month)
                scores.loc[month,  'Number'] = int(len(data_month))

    if ('OOT' in data.keys()) and ('TEST' in data.keys()):
        oot_lenght = len(data['OOT'][date_column].unique())
        scores.loc[months[-(oot_lenght+1)], 'OOT'] = scores.loc[months[-(oot_lenght+1)], 'TEST']

    number_columns = list(set(number_columns))
    number_columns = ', '.join(number_columns)
    scores['Month'] = ['.'.join([str(x)[-2:], str(x)[:-2]]) for x in scores.index]
    not_a_month = [x for x in scores.columns if x != 'Month']
    scores[not_a_month] = scores[not_a_month].astype('float').fillna('').round(6)
    return scores


def master_scale(app_data=None, target=None, pred_proba='pred_proba', how='tree', params={}, path=''):
    
    def binomial_test(buckets):
        buckets['k*']=norm.ppf(0.025)*np.sqrt(buckets['count']*buckets['median']*(1-buckets['median']))+buckets['count']*buckets['median']
        buckets['PD *']=buckets['k*']/buckets['count']
        a = (1-math.e**(-35*buckets['median']))/(1-math.e**(-35))
        buckets['R']=0.03*a+0.16*(1-a)
        buckets['t'] =norm.ppf(buckets['median'])
        buckets['Q'] =norm.cdf((np.sqrt(buckets.R)*norm.ppf(0.975)+buckets.t)/(np.sqrt(1-buckets.R)))
        buckets['v1'] = (buckets['Q']*(1-buckets['Q']))/norm.pdf((np.sqrt(buckets['R'])*norm.cdf(0.025)-buckets['t'])/(np.sqrt(1-buckets['R'])))
        buckets['v2'] = ((2*buckets['R']-1)*norm.cdf(0.025)-buckets['t']*np.sqrt(buckets['R']))/(np.sqrt(buckets['R']*(1-buckets['R'])))
        buckets['Q_correlated'] = buckets['Q']+(1/(2*buckets['count']))*(2*buckets['Q']-1+buckets['v1']*buckets['v2'])
        buckets['Q_correlated_H0'] = buckets['DR']>buckets['Q_correlated']
        buckets['binom_test_H0'] = buckets['DR']>buckets['median']
        return buckets

    if how == 'tree':
        
        y_ms = app_data[target]
        train_ms, test_ms, y_train_ms, y_test_ms = train_test_split(app_data[pred_proba].values.reshape(-1, 1), y_ms, test_size=0.3, random_state=42) 
        master_scale_tree = DecisionTreeClassifier(**params)
        master_scale_tree.fit(train_ms, y_train_ms)

        y_test_pred_ms = master_scale_tree.predict(test_ms)
        y_train_pred_ms = master_scale_tree.predict(train_ms)

        print('Train:\n')
        print('Accuracy =', round(metrics.accuracy_score(y_train_ms, y_train_pred_ms), 4))
        print('Precision =', round(metrics.precision_score(y_train_ms, y_train_pred_ms), 4))
        print('Recall =', round(metrics.recall_score(y_train_ms, y_train_pred_ms), 4))

        print('\nTest:\n')
        print('Accuracy =', round(metrics.accuracy_score(y_test_ms, y_test_pred_ms), 4))
        print('Precision =', round(metrics.precision_score(y_test_ms, y_test_pred_ms), 4))
        print('Recall =', round(metrics.recall_score(y_test_ms, y_test_pred_ms), 4))

        dot_data_3 = tree.export_graphviz(master_scale_tree,
                                          out_file=os.path.join(path, "master_scale_tree.dot"),
                                          feature_names=[pred_proba],
                                          class_names=['0', '1'],
                                          filled=True,
                                          rounded=True,
                                          special_characters=True,
                                          leaves_parallel=False)


        graph = pydotplus.graphviz.graph_from_dot_file(os.path.join(path, "master_scale_tree.dot"))
        display(Image(graph.create_png()))
        
        X_1 = pd.DataFrame(app_data[pred_proba].values.reshape(-1, 1))
        pred = master_scale_tree.predict_proba(app_data[pred_proba].values.reshape(-1, 1))[:, 1]

        X_1.rename(columns = {0: pred_proba}, inplace = True)

        X_1['class'] = pred
        buckets = pd.pivot_table(X_1, pred_proba, index = 'class', aggfunc = {pred_proba: ['mean', 'median', 'min', 'max', 'count']}) #считаем описательную статистику для каждого бакета 
        buckets = buckets.sort_values('mean') 
        buckets.reset_index(inplace=True)
        buckets['median_k+1'] = buckets['median'].shift(-1)
        buckets['median_k+1'].fillna(1, inplace=True)
        buckets['PD_k_upper']=np.sqrt(buckets['median']*buckets['median_k+1'])
        buckets['PD_k_lower']=buckets['PD_k_upper'].shift(1)
        buckets['PD_k_lower'].fillna(0, inplace=True)
        buckets['bucket_i']=buckets.index

        buckets = buckets[['class', 'bucket_i', 'count', 'min', 'max', 'median', 'mean', 'median_k+1', 'PD_k_lower', 'PD_k_upper']]
        app_data['class'] = pred
        app_data_ = app_data.merge(buckets, on='class')
        BR = app_data_.groupby('bucket_i')[target].agg(['count', 'sum'])
        BR['DR']=BR['sum']/BR['count']

        buckets['DR']=BR['DR']

        fig, ax1 = plt.subplots(figsize=(4, 4))
        BR['DR'].plot(kind='line', color='r', marker='d', secondary_y=True)
        BR['count'].plot(kind='bar')

        BR_data_gp_median = app_data_.groupby('bucket_i')[pred_proba].agg(['median'])
        BR['median']=BR_data_gp_median['median']
        BR['median_k+1'] = BR['median'].shift(-1)
        BR['median_k+1'].fillna(1, inplace=True)
        BR['PD_k_upper']=np.sqrt(BR['median']*BR['median_k+1'])
        BR['PD_k_lower']=BR['PD_k_upper'].shift(1)
        BR['PD_k_lower'].fillna(0, inplace=True)
        plt.show()
        bin_test = binomial_test(BR)

        return bin_test
    
    elif how == 'kmeans':
        
        #clustering
        km = KMeans(**params)
        
        # fit & predict clusters
        print('Обучаю k-means кластеризатор.')
        # fit & predict clusters
        app_data['cluster'] = km.fit_predict(app_data[pred_proba].values.reshape(-1,1))
        # cluster's centroids
        print('Центры кластеров:')
        print(km.cluster_centers_)

        clusters = pd.DataFrame(km.cluster_centers_, columns=['median']).sort_values('median')
        clusters['cluster']=clusters.index
        clusters.reset_index(inplace=True, drop=True)
        clusters['cluster_sc']=clusters.index
        clusters['median_k+1'] = clusters['median'].shift(-1).fillna(1)
        clusters['PD_k_upper']=np.sqrt(clusters['median']*clusters['median_k+1'])
        clusters['PD_k_lower']=clusters['PD_k_upper'].shift(1).fillna(0)

        app_data = app_data.merge(clusters, on='cluster')
        display(app_data)
        BR_k = app_data.groupby('cluster_sc')[target].agg(['count', 'sum']) 
        BR_k['DR']=BR_k['sum']/BR_k['count']

        fig, ax1 = plt.subplots(figsize=(10, 10))
        BR_k['DR'].plot(kind='line', color='r', marker='d', secondary_y=True)
        BR_k['count'].plot(kind='bar')
        plt.show()

        BR_k['median']=app_data.groupby('cluster_sc')[pred_proba].agg(['median'])['median']
        BR_k['median_k+1'] = BR_k['median'].shift(-1).fillna(1)
        BR_k['PD_k_upper']=np.sqrt(BR_k['median']*BR_k['median_k+1'])
        BR_k['PD_k_lower']=BR_k['PD_k_upper'].shift(1).fillna(0)

        clusters = binomial_test(BR_k)
        return clusters
    else:
        raise ValueError("Параметр 'how' должен быть 'tree' или 'kmeans'.")


# Фнукция преобровазания pd.Series формата "yyyymm" в формат "%b %y"
def set_pretty_dates(self, to_return='list'):

    '''
    self (pd.Series) - колонка с даьами формата "yyyymm"
    
    to_return (str) - текстовая переменная, с вариантами "list" или "dict". 
    Возвращать ли лист с датами или словарь в старом формате.
    '''
    
    # Проверяем, что нам дали pd.Series
    if isinstance(self, pd.Series):
        
        # Преобразуем формат в datetime
        col_modified = pd.to_datetime(self, format='%Y%m')

        # Преобразуем datetime в формат "%b %y"
        col_modified_m_y = col_modified.dt.strftime('%b %y')

        # Если возрвращать лист
        if to_return == 'list':
            
            # Переводим значения отформатированной колонки в лист
            fin_list = col_modified_m_y.values.tolist()

            return fin_list

        # Если возвращать словарь
        elif to_return == 'dict':
            
            # Переводим значения исходной колонки в лист
            fin_keys = self.values.tolist()

            # Переводим значения отформатированной колонки в лист
            fin_values = col_modified_m_y.values.tolist()
            
            # Сшиваем словарь
            fin_dict = dict(zip(fin_keys, fin_values))

            return fin_dict
    else:
        print('Допусается только формат pd.Series. Пожалуйста, проверьте передоваемую функции перменную.')