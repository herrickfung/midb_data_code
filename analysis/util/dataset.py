"""
All functions in this file are used to generate the configuration dictionary
to initialize the IndiMap object in the main function of analyze.py.
All dataset specific changes or adjustments are performed here.
"""

import numpy as np
import pandas as pd
default_config = {
    'bootstrap_iterations': 1000,
    'map_confusion': False,
    'map_category': True,
}

# subject by stim_label mapping
cate_map_config = {
    'bootstrap_iterations': 50,
    'map_confusion': False,
    'map_category': False,
}

def determine_condition(row):
    return_values = {
        ('Low', 'accuracy focus'): 0,
        ('Low', 'speed focus'): 1,
        ('High', 'accuracy focus'): 0,
        ('High', 'speed focus'): 1
    }
    return return_values[(row['noise'], row['sat'])]


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


def _remove_group_average(data, id_col, group_col, condition_col, variables):
    """
    Remove the average-performer effect from the response matrix, prior to
    the IndiMap mapping stage: subtract, separately for each
    group_col/condition combination, the average response across
    subjects/instances (id_col) from each individual's own response.

    The across-subject/instance average is computed on each individual's
    own (id_col, group_col, condition_col)-level mean first, then averaged
    unweighted across individuals - so subjects/instances that happen to
    contribute more trials/reps/noise-levels within a group/condition
    don't get overweighted in the reference average.
    """
    data = data.copy()
    present_vars = [v for v in variables if v in data.columns]

    individual_avg = (
        data.groupby([id_col, group_col, condition_col])[present_vars]
        .mean()
        .reset_index()
    )
    group_avg = (
        individual_avg.groupby([group_col, condition_col])[present_vars]
        .mean()
        .reset_index()
        .rename(columns={v: f'{v}_group_avg' for v in present_vars})
    )

    data = data.merge(group_avg, on=[group_col, condition_col], how='left')
    for v in present_vars:
        data[v] = data[v] - data[f'{v}_group_avg']
    data = data.drop(columns=[f'{v}_group_avg' for v in present_vars])
    return data


def remove_category_average(data, id_col, category_col, condition_col, variables):
    """ Category-level variant of _remove_group_average: category_col is
    the stimulus category (e.g. 'stim'), so every trial in a category gets
    the same category/condition-level average subtracted. """
    return _remove_group_average(data, id_col, category_col, condition_col, variables)


def remove_item_average(data, id_col, item_col, condition_col, variables):
    """ Item-level variant of _remove_group_average: item_col is the
    individual item/image index (e.g. 'mnist_index'/'image_index'), so
    each item gets its own item/condition-level average subtracted -
    finer-grained than remove_category_average. """
    return _remove_group_average(data, id_col, item_col, condition_col, variables)


def get_human_on_mnist(variant: str = 'standard'):
    path = 'dataset/mnist/standard/human.csv'
    data = pd.read_csv(path)
    data['noise'] = ['Low' if x == 'easy' else 'High' for x in data.noise]
    data['resp'] = data['response']
    data['acc'] = data['correct']
    data['rt'] = data['resp_rt']
    data['conf'] = data['confidence']
    data['subj'] = data['subject']
    if variant == 'category':
        data = data.groupby(['stim', 'sat', 'noise', 'subject', 'repeat']).mean(numeric_only=True).reset_index()
    else:
        data = data.groupby(['mnist_index', 'sat', 'noise', 'subject', 'repeat']).mean(numeric_only=True).reset_index()
    data['cond'] = data.apply(determine_condition, axis = 1)
    return data


def get_human_on_ecoset10(variant: str = 'standard'):
    path = 'dataset/ecoset10/standard/human.csv'
    data = pd.read_csv(path)
    data = exclude_RT_outliers(data)
    image_to_stim = data[['image_index', 'stim']].drop_duplicates().sort_values('image_index')
    stim_to_id = {stim: i for i, stim in enumerate(image_to_stim['stim'].unique())}
    data['subj'] = data['subject_ID']
    data['stim'] = data['stim'].map(stim_to_id)
    data['resp'] = data['resp'].map(stim_to_id)
    data['rt'] = data['p_rt']
    if variant == 'category':
        data = data.groupby(['stim', 'blur', 'subj', 'reps']).mean(numeric_only=True).reset_index()
    else:
        data = data.groupby(['image_index', 'blur', 'subj', 'reps']).mean(numeric_only=True).reset_index()
    data['cond'] = [1 for x in range(len(data))]
    return data


def get_rtnet_on_mnist(variant: str = 'standard'):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/{variant}/rtnet.csv')
    model_data.mnist_index = model_data.mnist_index + 1

    if variant == 'standard':
        model_data['stim'] = model_data['true label']
        model_data['resp'] = model_data['choice']
        model_data['acc'] = model_data['correct']
        model_data['conf'] = model_data['confidence diff']
        model_data['inst'] = model_data['model']
        model_data['cond'] = [1 if x == 'speed focus' else 0 for x in model_data.sat]
        model_data = model_data.groupby(['mnist_index', 'cond', 'noise', 'inst', 'reps']).mean(numeric_only=True).reset_index()
        model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan
    elif variant == 'untrained':
        model_data['inst'] = model_data['instance']
        model_data['cond'] = [1 if x == 3 else 6 for x in model_data.threshold]
        model_data = model_data.groupby(['mnist_index', 'cond', 'noise', 'inst', 'reps']).mean(numeric_only=True).reset_index()

    config = {
        'task_name': 'mnist',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}/mnist_rtnet',
        'graph_path': f'IndiMap_plots/{variant}/mnist_rtnet',
    }
    return {**default_config, **config}


def get_rtnet_on_mnist_control(n):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/control/rtnet/control_inst_{n}.csv')
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['mnist_index'] = model_data.mnist_index + 1
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/control/mnist_rtnet/inst_{n}',
        'graph_path': f'IndiMap_plots/control/mnist_rtnet/inst_{n}',
    }
    return {**default_config, **config}


def get_rtnet_on_mnist_accuracy_control(sd):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/accuracy_control/rtnet/mean{str(sd)}sd.csv')
    model_data['inst'] = model_data['instance']
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['mnist_index'] = model_data.mnist_index + 1
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/accuracy_control/mnist_rtnet/{str(sd)}sd',
        'graph_path': f'IndiMap_plots/accuracy_control/mnist_rtnet/{str(sd)}sd',
    }
    return {**default_config, **config}


def get_alexnet_on_mnist(variant: str = 'standard'):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/{variant}/alexnet.csv')
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['conf'] = model_data['top2diff_conf']
    if variant == 'standard':
        model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}/mnist_alexnet',
        'graph_path': f'IndiMap_plots/{variant}/mnist_alexnet',
    }
    return {**default_config, **config}


def get_alexnet_on_mnist_control(n):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/control/alexnet/control_inst_{n}.csv')
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
        'output_path': f'IndiMap_results/control/mnist_alexnet/inst_{n}',
        'graph_path': f'IndiMap_plots/control/mnist_alexnet/inst_{n}',
    }
    return {**default_config, **config}


def get_alexnet_on_mnist_accuracy_control(sd):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/accuracy_control/alexnet/mean{str(sd)}sd.csv')
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
        'output_path': f'IndiMap_results/accuracy_control/mnist_alexnet/{str(sd)}sd',
        'graph_path': f'IndiMap_plots/accuracy_control/mnist_alexnet/{str(sd)}sd',
    }
    return {**default_config, **config}


def get_resnet18_on_mnist(variant: str = 'standard'):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/{variant}/resnet18.csv')
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['conf'] = model_data['top2diff_conf']
    if variant == 'standard':
        model_data['cond'] = [1 if x == 'speed focus' else 0 for x in model_data.sat]
    else:
        model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data = model_data.groupby(['mnist_index', 'cond', 'noise', 'inst']).mean(numeric_only=True).reset_index()
    if variant == 'standard':
        model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}/mnist_resnet18',
        'graph_path': f'IndiMap_plots/{variant}/mnist_resnet18',
    }
    return {**default_config, **config}


def get_resnet18_on_mnist_control(n):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/control/resnet18/control_inst_{n}.csv')
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['conf'] = model_data['top2diff_conf']
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/control/mnist_resnet18/inst_{n}',
        'graph_path': f'IndiMap_plots/control/mnist_resnet18/inst_{n}',
    }
    return {**default_config, **config}


def get_resnet18_on_mnist_accuracy_control(sd):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/accuracy_control/resnet18/mean{str(sd)}sd.csv')
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['conf'] = model_data['top2diff_conf']
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/accuracy_control/mnist_resnet18/{str(sd)}sd',
        'graph_path': f'IndiMap_plots/accuracy_control/mnist_resnet18/{str(sd)}sd',
    }
    return {**default_config, **config}


def get_rtnet_on_ecoset10(variant: str = 'standard'):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv(f'dataset/ecoset10/{variant}/rtnet.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data['resp'] = model_data['resp'].astype(int)
    if variant == 'standard':
        model_data = model_data.groupby(['image_index', 'cond', 'blur', 'inst', 'rep'], as_index=False).mean(numeric_only=True)
    else:
        model_data = model_data.groupby(['image_index', 'cond', 'inst', 'rep'], as_index=False).mean(numeric_only=True)

    config = {
        'task_name': 'ecoset10',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}/ecoset10_rtnet',
        'graph_path': f'IndiMap_plots/{variant}/ecoset10_rtnet',
    }
    return {**default_config, **config}


def get_rtnet_on_ecoset10_control(n):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv(f'dataset/ecoset10/control/rtnet/control_inst_{n}.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data['resp'] = model_data['resp'].astype(int)

    config = {
        'task_name': 'ecoset10',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/control/ecoset10_rtnet/inst_{n}',
        'graph_path': f'IndiMap_plots/control/ecoset10_rtnet/inst_{n}',
    }
    return {**default_config, **config}


def get_alexnet_on_ecoset10(variant: str = 'standard'):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv(f'dataset/ecoset10/{variant}/alexnet.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    if variant == 'standard':
        model_data = model_data.groupby(['image_index', 'blur', 'inst']).mean(numeric_only=True).reset_index()
    else:
        model_data = model_data.groupby(['image_index', 'inst']).mean(numeric_only=True).reset_index()
    
    config = {
        'task_name': 'ecoset10',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}/ecoset10_alexnet',
        'graph_path': f'IndiMap_plots/{variant}/ecoset10_alexnet',
    }
    return {**default_config, **config}


def get_alexnet_on_ecoset10_control(n):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv(f'dataset/ecoset10/control/alexnet/control_inst_{n}.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]

    config = {
        'task_name': 'ecoset10',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/control/ecoset10_alexnet/inst_{n}',
        'graph_path': f'IndiMap_plots/control/ecoset10_alexnet/inst_{n}',
    }
    return {**default_config, **config}


def get_resnet18_on_ecoset10(variant: str = 'standard'):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv(f'dataset/ecoset10/{variant}/resnet18.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    if variant == 'standard':
        model_data = model_data.groupby(['image_index', 'blur', 'inst']).mean(numeric_only=True).reset_index()
    else:
        model_data = model_data.groupby(['image_index', 'inst']).mean(numeric_only=True).reset_index()

    config = {
        'task_name': 'ecoset10',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}/ecoset10_resnet18',
        'graph_path': f'IndiMap_plots/{variant}/ecoset10_resnet18',
    }
    return {**default_config, **config}


def get_resnet18_on_ecoset10_control(n):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv(f'dataset/ecoset10/control/resnet18/control_inst_{n}.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]

    config = {
        'task_name': 'ecoset10',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/control/ecoset10_resnet18/inst_{n}',
        'graph_path': f'IndiMap_plots/control/ecoset10_resnet18/inst_{n}',
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
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/imagenet16_resnet18_{n}',
        'graph_path': f'IndiMap_plots/imagenet16_resnet18_{n}',
    }
    return {**default_config, **config}

def get_rtnet_on_mnist_category_map():
    human_data = get_human_on_mnist('category')
    model_data = pd.read_csv(f'dataset/mnist/standard/rtnet.csv')
    model_data.mnist_index = model_data.mnist_index + 1

    model_data['stim'] = model_data['true label']
    model_data['resp'] = model_data['choice']
    model_data['acc'] = model_data['correct']
    model_data['conf'] = model_data['confidence diff']
    model_data['inst'] = model_data['model']
    model_data['cond'] = [1 if x == 'speed focus' else 0 for x in model_data.sat]
    model_data = model_data.groupby(['stim', 'cond', 'noise', 'inst', 'reps']).mean(numeric_only=True).reset_index()
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'stim',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/category/mnist_rtnet',
        'graph_path': f'IndiMap_plots/category/mnist_rtnet',
    }
    return {**cate_map_config, **config}


def get_alexnet_on_mnist_category_map():
    human_data = get_human_on_mnist('category')
    model_data = pd.read_csv(f'dataset/mnist/standard/alexnet.csv')
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['conf'] = model_data['top2diff_conf']
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan
    model_data = model_data.groupby(['stim', 'cond', 'noise', 'inst']).mean(numeric_only=True).reset_index()

    config = {
        'task_name': 'mnist',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'stim',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/category/mnist_alexnet',
        'graph_path': f'IndiMap_plots/category/mnist_alexnet',
    }
    return {**cate_map_config, **config}


def get_resnet18_on_mnist_category_map():
    human_data = get_human_on_mnist('category')
    model_data = pd.read_csv(f'dataset/mnist/standard/resnet18.csv')
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['conf'] = model_data['top2diff_conf']
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan
    model_data = model_data.groupby(['stim', 'cond', 'noise', 'inst']).mean(numeric_only=True).reset_index()

    config = {
        'task_name': 'mnist',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'stim',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/category/mnist_resnet18',
        'graph_path': f'IndiMap_plots/category/mnist_resnet18',
    }
    return {**cate_map_config, **config}


def get_rtnet_on_ecoset10_category_map():
    human_data = get_human_on_ecoset10('category')
    model_data = pd.read_csv(f'dataset/ecoset10/standard/rtnet.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data['resp'] = model_data['resp'].astype(int)
    model_data = model_data.groupby(['stim', 'cond', 'blur', 'inst', 'rep'], as_index=False).mean(numeric_only=True)

    config = {
        'task_name': 'ecoset10',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'stim',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/category/ecoset10_rtnet',
        'graph_path': f'IndiMap_plots/category/ecoset10_rtnet',
    }
    return {**cate_map_config, **config}


def get_alexnet_on_ecoset10_category_map():
    human_data = get_human_on_ecoset10('category')
    model_data = pd.read_csv(f'dataset/ecoset10/standard/alexnet.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data = model_data.groupby(['stim', 'cond', 'blur', 'inst']).mean(numeric_only=True).reset_index()

    config = {
        'task_name': 'ecoset10',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'stim',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/category/ecoset10_alexnet',
        'graph_path': f'IndiMap_plots/category/ecoset10_alexnet',
    }
    return {**cate_map_config, **config}


def get_resnet18_on_ecoset10_category_map():
    human_data = get_human_on_ecoset10('category')
    model_data = pd.read_csv(f'dataset/ecoset10/standard/resnet18.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data = model_data.groupby(['stim', 'cond', 'blur', 'inst']).mean(numeric_only=True).reset_index()

    config = {
        'task_name': 'ecoset10',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'stim',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/category/ecoset10_resnet18',
        'graph_path': f'IndiMap_plots/category/ecoset10_resnet18',
    }
    return {**cate_map_config, **config}


# demeaning getters (standard/uncontrolled trained models, both MNIST and
# ecoset10). Mapping stays trial-by-trial at the item level
# (map_together='mnist_index'/'image_index', default_config), exactly like
# the plain get_{model}_on_{task}(variant='standard') getters - the only
# addition is a demeaning step applied to both subj_data and inst_data
# *before* that item-level mapping: variant='category' removes the
# average-performer category-level effect (remove_category_average,
# grouped by 'stim'+'cond'); variant='item' removes the average-performer
# item-level effect instead (remove_item_average, grouped by the item
# index+'cond' - finer-grained than category). See those functions'
# docstrings for exactly what is subtracted.
def get_rtnet_on_mnist_demeaning(variant='category'):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv('dataset/mnist/standard/rtnet.csv')
    model_data.mnist_index = model_data.mnist_index + 1
    model_data['stim'] = model_data['true label']
    model_data['resp'] = model_data['choice']
    model_data['acc'] = model_data['correct']
    model_data['conf'] = model_data['confidence diff']
    model_data['inst'] = model_data['model']
    model_data['cond'] = [1 if x == 'speed focus' else 0 for x in model_data.sat]
    model_data = model_data.groupby(['mnist_index', 'cond', 'noise', 'inst', 'reps']).mean(numeric_only=True).reset_index()
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    group_col = 'stim' if variant == 'category' else 'mnist_index'
    demean = remove_category_average if variant == 'category' else remove_item_average
    human_data = demean(human_data, 'subj', group_col, 'cond', ['acc', 'conf', 'rt'])
    model_data = demean(model_data, 'inst', group_col, 'cond', ['acc', 'conf', 'rt'])

    config = {
        'task_name': 'mnist',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}_demeaning/mnist_rtnet',
        'graph_path': f'IndiMap_plots/{variant}_demeaning/mnist_rtnet',
    }
    return {**default_config, **config}


def get_alexnet_on_mnist_demeaning(variant='category'):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv('dataset/mnist/standard/alexnet.csv')
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['conf'] = model_data['top2diff_conf']
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    group_col = 'stim' if variant == 'category' else 'mnist_index'
    demean = remove_category_average if variant == 'category' else remove_item_average
    human_data = demean(human_data, 'subj', group_col, 'cond', ['acc', 'conf', 'rt'])
    model_data = demean(model_data, 'inst', group_col, 'cond', ['acc', 'conf', 'rt'])

    config = {
        'task_name': 'mnist',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}_demeaning/mnist_alexnet',
        'graph_path': f'IndiMap_plots/{variant}_demeaning/mnist_alexnet',
    }
    return {**default_config, **config}


def get_resnet18_on_mnist_demeaning(variant='category'):
    human_data = get_human_on_mnist()
    model_data = pd.read_csv('dataset/mnist/standard/resnet18.csv')
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['conf'] = model_data['top2diff_conf']
    model_data['cond'] = [1 if x == 'speed focus' else 0 for x in model_data.sat]
    model_data = model_data.groupby(['mnist_index', 'cond', 'noise', 'inst']).mean(numeric_only=True).reset_index()
    model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    group_col = 'stim' if variant == 'category' else 'mnist_index'
    demean = remove_category_average if variant == 'category' else remove_item_average
    human_data = demean(human_data, 'subj', group_col, 'cond', ['acc', 'conf', 'rt'])
    model_data = demean(model_data, 'inst', group_col, 'cond', ['acc', 'conf', 'rt'])

    config = {
        'task_name': 'mnist',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}_demeaning/mnist_resnet18',
        'graph_path': f'IndiMap_plots/{variant}_demeaning/mnist_resnet18',
    }
    return {**default_config, **config}


def get_rtnet_on_ecoset10_demeaning(variant='category'):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv('dataset/ecoset10/standard/rtnet.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data['resp'] = model_data['resp'].astype(int)
    model_data = model_data.groupby(['image_index', 'cond', 'blur', 'inst', 'rep'], as_index=False).mean(numeric_only=True)

    group_col = 'stim' if variant == 'category' else 'image_index'
    demean = remove_category_average if variant == 'category' else remove_item_average
    human_data = demean(human_data, 'subj', group_col, 'cond', ['acc', 'conf', 'rt'])
    model_data = demean(model_data, 'inst', group_col, 'cond', ['acc', 'conf', 'rt'])

    config = {
        'task_name': 'ecoset10',
        'model_name': 'rtnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}_demeaning/ecoset10_rtnet',
        'graph_path': f'IndiMap_plots/{variant}_demeaning/ecoset10_rtnet',
    }
    return {**default_config, **config}


def get_alexnet_on_ecoset10_demeaning(variant='category'):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv('dataset/ecoset10/standard/alexnet.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data = model_data.groupby(['image_index', 'blur', 'inst']).mean(numeric_only=True).reset_index()

    group_col = 'stim' if variant == 'category' else 'image_index'
    demean = remove_category_average if variant == 'category' else remove_item_average
    human_data = demean(human_data, 'subj', group_col, 'cond', ['acc', 'conf', 'rt'])
    model_data = demean(model_data, 'inst', group_col, 'cond', ['acc', 'conf', 'rt'])

    config = {
        'task_name': 'ecoset10',
        'model_name': 'alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}_demeaning/ecoset10_alexnet',
        'graph_path': f'IndiMap_plots/{variant}_demeaning/ecoset10_alexnet',
    }
    return {**default_config, **config}


def get_resnet18_on_ecoset10_demeaning(variant='category'):
    human_data = get_human_on_ecoset10()
    model_data = pd.read_csv('dataset/ecoset10/standard/resnet18.csv')
    model_data['conf'] = model_data['conf_top2diff']
    model_data['cond'] = [1 for x in range(len(model_data))]
    model_data = model_data.groupby(['image_index', 'blur', 'inst']).mean(numeric_only=True).reset_index()

    group_col = 'stim' if variant == 'category' else 'image_index'
    demean = remove_category_average if variant == 'category' else remove_item_average
    human_data = demean(human_data, 'subj', group_col, 'cond', ['acc', 'conf', 'rt'])
    model_data = demean(model_data, 'inst', group_col, 'cond', ['acc', 'conf', 'rt'])

    config = {
        'task_name': 'ecoset10',
        'model_name': 'resnet18',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'image_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}_demeaning/ecoset10_resnet18',
        'graph_path': f'IndiMap_plots/{variant}_demeaning/ecoset10_resnet18',
    }
    return {**default_config, **config}