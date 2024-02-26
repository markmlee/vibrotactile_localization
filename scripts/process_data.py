import microphone
import microphone_utils
# from franka_motion import FrankaMotion
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sys



def load_data_and_plot():
    #create instance of microphone class
    devicelist=[0,1,11,12,13,14]
    number_of_mics = len(devicelist)
    fs = 22050
    channels_in = 1
    save_path_data = "/home/iam-lab/audio_localization/audio_datacollection/data/franka_init_test_6mic/"

    total_trial_count = 100


    mics_to_plot = [0] #index 0,1,2,3,4,5 -> mic 0,1,11,12,13,14

    #load all wav files from dataset (all mic, all trials)
    mics_all_trials = microphone_utils.load_wav_files_from_dataset(devicelist, total_trial_count, save_path_data)
    # print(f"size of mics_all_trials: {len(mics_all_trials)}") #--> 6 mics

    data_list = []
    #get only the indices of interest from mics_to_plot from all trials
    for i in mics_to_plot:
        data_list.append(mics_all_trials[i])
    
    #plot envelope ratio
    # microphone_utils.plot_envelope_ratio(data_list)


    #plot recorded wav files
    microphone_utils.plot_time_domain(data_list, fs)

    #plot spectrogram of data_list[0]
    # microphone_utils.plot_spectrogram(data_list[0], fs)

def main():
    print(f" ------ starting script ------  ")

    #create instance of microphone class
    devicelist=[0,1,11,12,13,14]
    number_of_mics = len(devicelist)
    channels_in = 1
    save_path_data = "/home/iam-lab/audio_localization/audio_datacollection/data/franka_init_test_6mic/"

    total_trial_count = 100

    #load all wav files from dataset (all mic, all trials)
    mics_all_trials = microphone_utils.load_wav_files_as_dataset(devicelist, total_trial_count, save_path_data)
    print(f"size of mics_all_trials: {len(mics_all_trials)}") #--> 100 trials

    #load all labels from dataset (all trials)
    labels_all_trials = microphone_utils.load_labels_from_dataset(save_path_data, total_trial_count)

    # print(f"labels size of all trials: {len(labels_all_trials)}") #--> 100 
    # labels for trial 0: [0. 0.]

    # get list of [X,Y] from dataset
    XY_list = microphone_utils.preprocess_data(mics_all_trials, labels_all_trials)

    print(f"size of XY_list: {len(XY_list)}") #--> 100 trials

    



if __name__ == '__main__':
    main()