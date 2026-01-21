'''
this script merge all individual data files and group all together into one
'''

import numpy as np
import json
import os
import pathlib
import pandas as pd


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = ' ' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")


def manage_path():
    # input
    current_path = pathlib.Path(__file__).parent.absolute()
    data_path = current_path / 'data'
    parti_data_path = [x for x in data_path.iterdir() if x.with_suffix('.csv')]

    # output
    merged_raw_path = current_path / 'merged_raw'
    outputfile_path = merged_raw_path / 'data.csv'

    try:
        os.mkdir(merged_raw_path)
    except FileExistsError:
        print('result directory already existed ......')

    return parti_data_path, outputfile_path


def read(datafile, subj_no):
    # this function read and clean up the file and return a dataframe ready for process
    data = pd.read_csv(datafile, delimiter = ',')

    data['subject_ID'] = subj_no+1
    subj_info = data[data['trial_type'] == 'survey'].response
    subj_info = json.loads(subj_info.values[0])
    subj_age = int(subj_info.get("P0_Q1"))
    subj_sex = subj_info.get("P1_Q0")

    # clean up and remove uncessary data
    need_data = ['stimulus', 'perceptual', 'confidence']
    data = data[data['curr_triallist'] == 2]
    data = data[data['stimulus'].isin(need_data)]

    data['trial_no'] = data['trial_no'].astype(int)
    total_no_of_trial = data['trial_no'].max()+1
    percept_rt = []
    conf_resp = []
    conf_rt = []

    # go into each trial, get value and append to array for later
    for i in range(0, total_no_of_trial):
        trial_data = data[data['trial_no'] == i]
        percept_rt.append(trial_data[trial_data['stimulus'] == 'perceptual'].rt.values[0])
        conf_resp.append(trial_data[trial_data['stimulus'] == 'confidence'].response.values[0])
        conf_rt.append(trial_data[trial_data['stimulus'] == 'confidence'].rt.values[0])

    data['trial_no'] = data['trial_no'] + 1
    # remove more unnecessary data
    data = data.drop(data[data['stimulus'] == 'stimulus'].index)
    data = data.drop(data[data['stimulus'] == 'confidence'].index)
    data = data.dropna(axis = 1)

    data['age'] = subj_age
    data['sex'] = subj_sex
    data['acc'] = data['correct'].astype(int)
    data['conf'] = conf_resp
    data['p_rt'] = percept_rt
    data['c_rt'] = conf_rt

    data['acc'] = data['acc'].astype(int)
    data['p_rt'] = data['p_rt'].astype(float)
    data['conf'] = data['conf'].astype(int)
    data['c_rt'] = data['c_rt'].astype(float)

    data['stim'] = data['category']
    data['resp'] = data['response']
    data['blur'] = data['blur'].astype(int)
    data['reps'] = data['reps'].astype(int)
    data['image_index'] = data['image_index'].astype(int)

    data = data.drop(['trial_type', 'trial_index', 'time_elapsed', 'internal_node_id', 'stimulus', 'curr_triallist', 'correct'], axis = 1)
    data.reset_index()

    # reorder to make it look better
    data = data[['subject_ID', 'age', 'sex', 'trial_no', 'image_index', 'stim', 'blur', 'reps', 'resp', 'acc', 'conf', 'p_rt', 'c_rt' ]]
    return data


def merge_raw_data():
    in_data_path, out_merged_raw_path = manage_path()
    print("\nProcessing and Merging data files ......")
    merged = pd.DataFrame({})
    for file in in_data_path:
        data = read(file, in_data_path.index(file))
        merged = pd.concat([merged, data])
        progress_bar(in_data_path.index(file)+1, len(in_data_path))
    print('--------------------------------------------------------------------------------')
    print(str(len(in_data_path)) + " files processed completed.")

    # setup output file
    merged.to_csv(out_merged_raw_path, sep=',', index=False)
    print("--------------------------------------------------------------------------------")
    print(f'data ready at {out_merged_raw_path}')

    return merged

