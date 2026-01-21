"""
All functions in this file are used to generate the configuration dictionary
to initialize the IndiMap object in the main function of analyze.py.
All dataset specific changes or adjustments are performed here.
"""

from math import nan
import numpy as np
import pandas as pd
default_config = {
    'bootstrap_iterations': 1000,
    'map_confusion': False,
}


def determine_condition(row):
    return_values = {
        ('Low', 'accuracy focus'): 0,
        ('Low', 'speed focus'): 1,
        ('High', 'accuracy focus'): 0,
        ('High', 'speed focus'): 1
    }
    return return_values[(row['noise'], row['sat'])]


def get_human_on_mnist():
    path = 'dataset/mnist/human.csv'
    data = pd.read_csv(path)
    data['noise'] = ['Low' if x == 'easy' else 'High' for x in data.noise]
    data['resp'] = data['response']
    data['acc'] = data['correct']
    data['rt'] = data['resp_rt']
    data['conf'] = data['confidence']
    data['subj'] = data['subject']
    data = data.groupby(['mnist_index', 'sat', 'noise', 'subject', 'repeat']).mean(numeric_only=True).reset_index()
    data['cond'] = data.apply(determine_condition, axis = 1)
    return data


def get_rtnet_on_mnist():
    human_data = get_human_on_mnist()
    model_data = pd.read_csv('dataset/mnist/rtnet.csv')
    model_data.mnist_index = model_data.mnist_index + 1
    model_data['stim'] = model_data['true label']
    model_data['resp'] = model_data['choice']
    model_data['acc'] = model_data['correct']
    model_data['conf'] = model_data['confidence diff']
    model_data['inst'] = model_data['model']
    model_data['cond'] = [1 if x == 'speed focus' else 0 for x in model_data.sat]
    model_data = model_data.groupby(['mnist_index', 'cond', 'noise', 'inst', 'reps']).mean(numeric_only=True).reset_index()
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': 'IndiMap_results/mnist_rtnet',
        'graph_path': 'IndiMap_plots/mnist_rtnet',
    }
    return {**default_config, **config}


def get_alexnet_on_mnist():
    human_data = get_human_on_mnist()
    model_data = pd.read_csv('dataset/mnist/alexnet.csv')
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['conf'] = model_data['top2diff_conf']
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': 'IndiMap_results/mnist_alexnet',
        'graph_path': 'IndiMap_plots/mnist_alexnet',
    }
    return {**default_config, **config}


def get_alexnet_on_mnist_control(n):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/alexnet_control/control_inst_{n}.csv')
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['conf'] = model_data['top2diff_conf']
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/mnist_alexnet_control/inst_{n}',
        'graph_path': f'IndiMap_plots/mnist_alexnet_control/inst_{n}',
    }
    return {**default_config, **config}


def get_resnet18_on_mnist():
    human_data = get_human_on_mnist()
    model_data = pd.read_csv('dataset/mnist/resnet18.csv')
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['conf'] = model_data['top2diff_conf']
    model_data['cond'] = [1 if x == 'speed focus' else 0 for x in model_data.sat]
    model_data = model_data.groupby(['mnist_index', 'cond', 'noise', 'inst']).mean(numeric_only=True).reset_index()
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': 'IndiMap_results/mnist_resnet18',
        'graph_path': 'IndiMap_plots/mnist_resnet18',
    }
    return {**default_config, **config}


def exclude_RT_outliers(data):
    """
    Exclude RT outliers using Tukey's interquartile criterion.
    Follow RTNet paper procedure.
    """

    subjs = data['subject_ID'].unique()
    blurs = data['blur'].unique()
    removal_array = []
    for subj in subjs:
        for blur in blurs:
            filter_data = data[(data['subject_ID'] == subj) & (data['blur'] == blur)]
            q1 = filter_data['p_rt'].quantile(0.25)
            q3 = filter_data['p_rt'].quantile(0.75)
            iqr = q3 - q1
            lb = q1 - 1.5 * iqr
            ub = q3 + 1.5 * iqr
            outliers = filter_data[(filter_data['p_rt'] < lb) | (filter_data['p_rt'] > ub)].index
            data.loc[outliers, 'p_rt'] = np.nan
            removal_array.append(len(outliers) / len(filter_data))
    # print(f"Mean: {np.mean(removal_array)}, Min: {np.min(removal_array)}, Max: {np.max(removal_array)}")
    return data


def get_human_on_ecoset10():
    path = 'dataset/ecoset10/human.csv'
    data = pd.read_csv(path)
    data = exclude_RT_outliers(data)
    image_to_stim = data[['image_index', 'stim']].drop_duplicates().sort_values('image_index')
    stim_to_id = {stim: i for i, stim in enumerate(image_to_stim['stim'].unique())}
    data['subj'] = data['subject_ID']
    data['stim'] = data['stim'].map(stim_to_id)
    data['resp'] = data['resp'].map(stim_to_id)
    data['rt'] = data['p_rt']
    data = data.groupby(['image_index', 'blur', 'subj', 'reps']).mean(numeric_only=True).reset_index()
    data['cond'] = [1 for x in range(len(data))]
    return data


def get_rtnet_on_ecoset10():
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv('dataset/ecoset10/rtnet.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data['resp'] = model_data['resp'].astype(int)
    model_data = model_data.groupby(['image_index', 'cond', 'blur', 'inst', 'rep'], as_index=False).mean(numeric_only=True)

    config = {
        'task_name': 'ecoset10',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': 'IndiMap_results/ecoset10_rtnet',
        'graph_path': 'IndiMap_plots/ecoset10_rtnet',
    }
    return {**default_config, **config}


def get_alexnet_on_ecoset10():
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv('dataset/ecoset10/alexnet.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data = model_data.groupby(['image_index', 'blur', 'inst']).mean(numeric_only=True).reset_index()

    config = {
        'task_name': 'ecoset10',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': 'IndiMap_results/ecoset10_alexnet',
        'graph_path': 'IndiMap_plots/ecoset10_alexnet',
    }
    return {**default_config, **config}


def get_resnet18_on_ecoset10():
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv('dataset/ecoset10/resnet18.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data = model_data.groupby(['image_index', 'blur', 'inst']).mean(numeric_only=True).reset_index()

    config = {
        'task_name': 'ecoset10',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': 'IndiMap_results/ecoset10_resnet18',
        'graph_path': 'IndiMap_plots/ecoset10_resnet18',
    }
    return {**default_config, **config}


def get_human_on_imagenet():
    path = 'dataset/imagenet16/human.csv'
    data = pd.read_csv(path)
    data = data.groupby(['image_index', 'blur', 'subj']).mean(numeric_only=True).reset_index()
    data['cond'] = [1 for x in range(len(data))]
    return data


def get_alexnet_on_imagenet():
    human_data = get_human_on_imagenet()
    model_data = pd.read_csv('dataset/imagenet16/alexnet2.csv')
    model_data = model_data[model_data.match == 'acc']
    model_data['conf'] = model_data['top2diff_conf']
    model_data['cond'] = [1 for x in range(len(model_data))]

    config = {
        'task_name': 'imagenet16',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': 'IndiMap_results/imagenet16_alexnet',
        'graph_path': 'IndiMap_plots/imagenet16_alexnet',
    }
    return {**default_config, **config}


def get_resnet18_on_imagenet():
    human_data = get_human_on_imagenet()
    model_data = pd.read_csv('dataset/imagenet16/resnet18.csv')
    model_data = model_data[model_data.match == 'acc']
    model_data['conf'] = model_data['top2diff_conf']
    model_data['cond'] = [1 for x in range(len(model_data))]

    config = {
        'task_name': 'imagenet16',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': 'IndiMap_results/imagenet16_resnet18',
        'graph_path': 'IndiMap_plots/imagenet16_resnet18',
    }
    return {**default_config, **config}


def get_resnet_on_imagenet_n(n):
    human_data = get_human_on_imagenet()
    model_data = pd.read_csv(f'dataset/imagenet16/resnet18_2_{n}.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for _ in range(len(model_data))]

    config = {
        'task_name': 'imagenet16',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/imagenet16_resnet18_{n}',
        'graph_path': f'IndiMap_plots/imagenet16_resnet18_{n}',
    }
    return {**default_config, **config}
