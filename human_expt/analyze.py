'''
analysis script
create this mainly for checking accuracy & confidence
'''

run_merge = False
run_process = False

from matplotlib import rcParams
import matplotlib.pyplot as plt
import merge_data
import numpy as np
import pandas as pd
import pathlib
import warnings
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
rcParams['font.family'] = 'CMU Sans Serif'


def manage_path():
    # input
    current_path = pathlib.Path(__file__).parent.absolute()
    data_path = current_path / 'merged_raw/data.csv'

    # output
    result_path = current_path / 'results'
    pro_data_path = current_path / 'results/data.csv'
    graph_path = result_path / 'graph'

    result_path.mkdir(exist_ok=True)
    graph_path.mkdir(exist_ok=True)
    return data_path, pro_data_path, graph_path


def read(datafile):
    data = pd.read_csv(datafile, delimiter = ',')
    return data


def process(data):
    subjs = data.subject_ID.unique()
    blurs = data.blur.unique()
    n_subjs = len(subjs)
    n_blurs = len(blurs)
    n_params = 6

    pro_data = np.empty(shape = (n_subjs, n_blurs, n_params))
    for i, subj in enumerate(subjs):
        subj_data = data[data.subject_ID == subj]
        for j, blur in enumerate(blurs):
            subj_cond_data = subj_data[data.blur == blur]

            subj_no = subj
            stim_cond = blur
            acc = subj_cond_data.acc.sum() / len(subj_cond_data)
            conf_mean = subj_cond_data.conf.mean()
            percept_rt = subj_cond_data.p_rt.mean()
            conf_rt = subj_cond_data.c_rt.mean()

            # package output
            pro_data[i, j, :] = (subj_no, stim_cond, acc, conf_mean,
                                 percept_rt, conf_rt,
                                 )

    # convert to pd.DataFrame
    pro_data = pro_data.reshape(n_subjs * n_blurs, n_params)
    column_name = ['subj_no', 'blur', 'acc',
                   'conf', 'p_rt', 'c_rt',
                   ]
    pro_data = pd.DataFrame(pro_data, columns=column_name)
    return pro_data


def exclude_subj(pro_data, raw_data):
    # exclude 2.5 away subjects and range 0.2 to 1
    subject_acc = pro_data.groupby('subj_no')['acc'].mean()
    mean_acc = subject_acc.mean()
    std_acc = subject_acc.std()
    lower_bound = max(mean_acc - 2.5 * std_acc, 0.25)
    upper_bound = mean_acc + 2.5 * std_acc
    exclusion = subject_acc[(subject_acc <= lower_bound) | (subject_acc >= upper_bound)].index
    pro_data = pro_data[~pro_data['subj_no'].isin(exclusion)]
    raw_data = raw_data[~raw_data['subject_ID'].isin(exclusion)]

    # renumber subject number
    subjs = pro_data['subj_no'].unique()
    subj_map = {old: new for new, old in enumerate(subjs, start=1)}

    # Map the old 'subj_no' to the new numbering
    pro_data['subj_no'] = pro_data['subj_no'].map(subj_map)
    raw_data['subject_ID'] = raw_data['subject_ID'].map(subj_map)

    print("---------------------------------------------------------------------")
    print("Exclusion Info:")
    print(f"{len(exclusion)} subjects excluded.")
    print(f"Analyzing {pro_data['subj_no'].nunique()} subjects ......")
    print("---------------------------------------------------------------------")

    return pro_data, raw_data


def plot_average(data, path, plot_param):
    subjs = data.subj_no.unique()
    blurs = sorted(data.blur.unique(), reverse=True)
    cmap = plt.get_cmap('Dark2')
    xticks = ['High', 'Low']

    plt.clf()
    plt.figure(figsize=(2, 2))
    for i, blur in enumerate(blurs):
        plot_data = data[data.blur==blur]

        if 'rt' in plot_param:
            plot_data[plot_param] = plot_data[plot_param] / 1000
        plt.bar(i, plot_data[plot_param].mean(),
                yerr=plot_data[plot_param].sem(),
                color=cmap(i), alpha=0.5
                )
        plt.scatter([i-0.25 for x in range(len(plot_data))], plot_data[plot_param],
                    s=5, alpha=0.8, color=cmap(i)
                    )

    plt.xticks([0,1], xticks, fontsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.xlabel('Blur Level', fontsize=8, fontweight='bold')
    plt.ylabel(plot_param, fontsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(plot_param, fontweight='bold', fontsize=10)
    figname = path / f"{plot_param}.png"
    plt.tight_layout()
    plt.savefig(figname, dpi=384)


def graph(data, path):
    plot_average(data, path, 'acc')
    plot_average(data, path, 'conf')
    plot_average(data, path, 'p_rt')
    plot_average(data, path, 'c_rt')

def main():
    in_data_path, out_pro_path, graph_path = manage_path()
    if run_process:
        if run_merge:
            raw_data = merge_data.merge_raw_data()

        print("\nProcessing individuals ......")
        data = read(in_data_path)
        pro_data = process(data)
        pro_data, raw_data = exclude_subj(pro_data, raw_data)
        pro_data.to_csv(out_pro_path, sep=',', index=False)
        raw_data.to_csv(in_data_path, sep=',', index=False)
        print('--------------------------------------------------------------------------------')
        print('Processing Completed.')
    else:
        pro_data = pd.read_csv(out_pro_path)
        print("Processed data read from " + str(out_pro_path))

    graph(pro_data, graph_path)

    print("--------------------------------------------------------------------------------")
    print('ALL DONE')


if __name__ == "__main__":
    main()
