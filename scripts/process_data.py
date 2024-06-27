import microphone
import microphone_utils
# from franka_motion import FrankaMotion
import argparse

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import seaborn as sns 
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset

import microphone_utils as mic_utils



def load_data_and_plot():
    devicelist=[0,1,11,12,13,14]
    number_of_mics = len(devicelist)
    fs = 44100
    channels_in = 1
    save_path_data = "/home/mark/audio_learning_project/data/test_generalization/stick_location/"

    total_trial_count = 5


    # -----------------------------------------------------------
    #plot single recorded wav files
    # mics_to_plot = [3] #index 0,1,2,3,4,5 -> mic 0,1,11,12,13,14

    # #load all wav files from dataset (all mic, all trials)
    # mics_all_trials = microphone_utils.load_wav_files_from_dataset(devicelist, total_trial_count, save_path_data)
    # # print(f"size of mics_all_trials: {len(mics_all_trials)}") #--> 6 mics

    # data_list = []
    # #get only the indices of interest from mics_to_plot from all trials
    # for i in mics_to_plot:
    #     data_list.append(mics_all_trials[i])
    
    # #plot envelope ratio
    # # microphone_utils.plot_envelope_ratio(data_list)
    
    # microphone_utils.plot_time_domain(data_list, fs)
        
    # -----------------------------------------------------------
    #plot all 6 wav files
        
    mics_to_plot = [0] #index 0,1,2,3,4,5 -> mic 0,1,11,12,13,14

    #load all wav files from dataset (all mic, all trials)
    mics_all_trials = microphone_utils.load_wav_files_from_dataset(devicelist, total_trial_count, save_path_data)
    # print(f"size of mics_all_trials: {len(mics_all_trials)}") #--> 6 mics

    data_list = []
    #get only the indices of interest from mics_to_plot from all trials
    for i in mics_to_plot:
        data_list.append(mics_all_trials[i])

    microphone_utils.grid_plot_time_domain(data_list, fs)
        
    # -----------------------------------------------------------

def visualize_amplitude_across_mic(dataset_distance_micAmplitude):
    """
    Visualize how amplitude decreases/increase by distance 
    input: (N_trials * repeated trials, 7) where 0th column is distance and 1-7th column is max amplitude of each trial for all mic
    output: plot curve of amplitude vs distance
    """

    num_trials, num_mic = dataset_distance_micAmplitude.shape
    num_mic = num_mic - 1 #exclude distance column
    
    # Create 2 subplots in a 2x1 format
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot amplitude vs distance for each mic using seaborn package
    for i in range(num_mic):
        # Create a DataFrame for the current mic
        df = pd.DataFrame({
            'Distance': dataset_distance_micAmplitude[:,0],
            'Normalized Amplitude': dataset_distance_micAmplitude[:, i+1]
        })

        # Determine which subplot to use based on mic number
        ax = axs[0] if i > 2 else axs[1]

        # Create a line plot for the current mic on the correct subplot
        sns.lineplot(data=df, x='Distance', y='Normalized Amplitude', label=f'Mic {i}', ax=ax, alpha=0.7, linestyle='-', linewidth=2.5)
    # Set labels and title for each subplot
    axs[0].set_xlabel('Distance (cm)')
    axs[0].set_ylabel('Normalized Amplitude')
    axs[0].set_title('Amplitude vs Distance for Mic 3,4,5')
    axs[0].legend()

    axs[1].set_xlabel('Distance (cm)')
    axs[1].set_ylabel('Normalized Amplitude')
    axs[1].set_title('Amplitude vs Distance for Mic 0,1,2')
    axs[1].legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_timeshift_across_mic(dataset_distance_micAmplitude):
    """
    Visualize how time of arrival decreases/increase by distance 
    input: (N_trials * repeated trials, 2) where 0th column is distance and 1th column is time difference
    output: plot curve of delta Time vs distance
    """

    num_trials, num_timediffs = dataset_distance_micAmplitude.shape
    num_timediffs = num_timediffs - 1 #exclude distance column

    mic_labels = ['Mic 3,0', 'Mic 4,1', 'Mic 5,2']

    # get the mean y-values at x=10
    y_at_10 = [0,0,0]

    # Create 3 subplots in a 3x1 format
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Plot time diff vs distance for each pair of mic (3,0), (4,1), (5,2) using seaborn package
    for i in range(num_timediffs):
        # Create a DataFrame for the current mic
        df = pd.DataFrame({
            'Distance': dataset_distance_micAmplitude[:,0],
            'Time Difference (ms)': dataset_distance_micAmplitude[:, i+1]
        })

        # Determine which subplot to use based on mic number
        ax = axs[0] if i == 0 else axs[1] if i == 1 else axs[2]

        # Create a line plot for the current mic on the correct subplot
        sns.lineplot(data=df, x='Distance', y='Time Difference (ms)', label=mic_labels[i], ax=ax, alpha=0.7, linestyle='-', linewidth=2.5)


        # Add a vertical dashed line at x=10
        # ax.axvline(10, color='r', linestyle='--')
        # Add a horizontal dashed line at the interpolated y-value
        # ax.axhline(y_at_10[i], color='r', linestyle='--')

    # Set labels and title for each subplot
    axs[0].title.set_text('Time Difference of Arrival vs Distance Hit on Cylinder')
    axs[0].set_xlabel('Distance (cm)')
    axs[0].set_ylabel('Time Difference (ms)')
    axs[0].legend()

    axs[1].set_xlabel('Distance (cm)')
    axs[1].set_ylabel('Time Difference (ms)')
    axs[1].legend()

    axs[2].set_xlabel('Distance (cm)')
    axs[2].set_ylabel('Time Difference (ms)')
    axs[2].legend()

    

    # Show the plot
    plt.tight_layout()
    plt.show()



def visualize_timeshift_across_mic_rings(dataset_distance_micAmplitude):
    """
    Visualize how time of arrival decreases/increase by distance 
    input: (N_trials * repeated trials, 2) where 0th column is distance and 1th column is time difference
    output: plot curve of delta Time vs distance
    """

    num_trials, num_timediffs = dataset_distance_micAmplitude.shape
    num_timediffs = num_timediffs - 1 #exclude distance column

    mic_labels = ['Mic 3,4', 'Mic 3,5', 'Mic 0,1', 'Mic 0,2']
    
    
    # Create 3 subplots in a 2x2 format
    fig, axs = plt.subplots(4,1, figsize=(10, 10))

    # Plot time diff vs distance for each pair of mic (3,0), (4,1), (5,2) using seaborn package
    for i in range(num_timediffs):
        # Create a DataFrame for the current mic
        df = pd.DataFrame({
            'Distance': dataset_distance_micAmplitude[:,0],
            'Time Difference (ms)': dataset_distance_micAmplitude[:, i+1]
        })

        # Determine which subplot to use based on mic number
        ax = axs[0] if i == 0 else axs[1] if i == 1 else axs[2] if i == 2 else axs[3]

        # Create a line plot for the current mic on the correct subplot
        sns.lineplot(data=df, x='Distance', y='Time Difference (ms)', label=mic_labels[i], ax=ax, alpha=0.7, linestyle='-', linewidth=2.5)


        # Add a vertical dashed line at x=10
        ax.axvline(10, color='r', linestyle='--')

        # Add a horizontal dashed line at y=0
        ax.axhline(0, color='r', linestyle='--')

    # Set labels and title for each subplot
    axs[0].title.set_text('Time Difference of Arrival vs Distance Hit on Cylinder')
    axs[0].set_xlabel('Distance (cm)')
    axs[0].set_ylabel('Time Difference (ms)')
    axs[0].legend()

    axs[1].set_xlabel('Distance (cm)')
    axs[1].set_ylabel('Time Difference (ms)')
    axs[1].legend()

    axs[2].set_xlabel('Distance (cm)')
    axs[2].set_ylabel('Time Difference (ms)')
    axs[2].legend()

    axs[3].set_xlabel('Distance (cm)')
    axs[3].set_ylabel('Time Difference (ms)')
    axs[3].legend()
    

    # Show the plot
    plt.tight_layout()
    plt.show()

def load_data_for_amplitude():
    save_path_data = "/home/mark/audio_learning_project/data/franka_2D_localization_full_UMC/"
    total_trial_count = 50 #50 distance x 15 radian intervals x 5 repeat counts = 3750 trials
    devicelist=[2]
    number_of_mics = 6
    fs = 44100

    # Define the ranges
    ranges0 = [(0, 50), (750, 800), (1500, 1550), (2250, 2300), (3000, 3050)] #trial ranges that are the same regions of the dataset
    ranges1 = [(50, 100), (800, 850), (1550, 1600), (2300, 2350), (3050, 3100)] #--> direct hit over the mic (drastic peak)
    ranges2 = [(100, 150), (850, 900), (1600, 1650), (2350, 2400), (3100, 3150)] 
    ranges3 = [(150, 200), (900, 950), (1650, 1700), (2400, 2450), (3150, 3200)] 
    ranges4 = [(200, 250), (950, 1000), (1700, 1750), (2450, 2500), (3200, 3250)]
    ranges5 = [(250, 300), (1000,1050), (1750,1800), (2500,2550), (3250,3300)] #-->-44 deg
    ranges6 = [(300, 350), (1050,1100), (1800,1850), (2550,2600), (3300,3350)]
    ranges7 = [(350, 400), (1100,1150), (1850,1900), (2600,2650), (3350,3400)]
    ranges8 = [(400, 450), (1150,1200), (1900,1950), (2650,2700), (3400,3450)]
    ranges9 = [(450, 500), (1200,1250), (1950,2000), (2700,2750), (3450,3500)]
    ranges10 = [(500, 550), (1250,1300), (2000,2050), (2750,2800), (3500,3550)]
    ranges11 = [(550, 600), (1300,1350), (2050,2100), (2800,2850), (3550,3600)]
    ranges12 = [(600, 650), (1350,1400), (2100,2150), (2850,2900), (3600,3650)]
    ranges13 = [(650, 700), (1400,1450), (2150,2200), (2900,2950), (3650,3700)]
    ranges14 = [(700, 750), (1450,1500), (2200,2250), (2950,3000), (3700,3750)]

    # ranges = ranges0 + ranges1 + ranges2 + ranges3 + ranges4 + ranges5 + ranges6 + ranges7 + ranges8 + ranges9 + ranges10 + ranges11 + ranges12 + ranges13 + ranges14
    ranges = ranges0
    num_trials = 50

    dataset_distance_micAmplitude = np.empty((0, 7))

    # Load the wav files for each range
    for trial_start,trial_end in ranges:
        mics_all_trials, mic_max_amplitude_all_trials, _ = microphone_utils.load_wav_files_from_dataset_sections(devicelist, trial_start, trial_end, save_path_data)

        #convert list to np
        mic_max_amplitude_all_trials = np.array(mic_max_amplitude_all_trials)
        #squeeze the 1st dimension
        mic_max_amplitude_all_trials = np.squeeze(mic_max_amplitude_all_trials)

        print(f"mic_max_amplitude_all_trials shape: {mic_max_amplitude_all_trials.shape}") #--> (N trials, 6 mics)
        #get entire max for each mic from all trials
        mic_max_amplitude = np.max(mic_max_amplitude_all_trials, axis=0)
        # print(f"mic_max_amplitude shape: {mic_max_amplitude.shape}") #--> (6 mics

        #normalize amplitude for each mic by dividing by max amplitude
        mic_max_amplitude_all_trials_normalized = mic_max_amplitude_all_trials / mic_max_amplitude

        #print shape of mic_max_amplitude_all_trials_normalized
        print(f"mic_max_amplitude_all_trials_normalized shape: {mic_max_amplitude_all_trials_normalized.shape}") #--> (N trials, 6 mics)

        #evenly space distance from 0 to 20 by num_trials
        distance = np.linspace(0, 20, num_trials)

        #concatenate distance to mic_max_amplitude_all_trials_normalized at the 0th column
        mic_max_amplitude_all_trials_normalized = np.concatenate([distance[:, np.newaxis], mic_max_amplitude_all_trials_normalized], axis=1)
        print(f"mic_max_amplitude_all_trials_normalized shape: {mic_max_amplitude_all_trials_normalized.shape}") #--> (N trials, 7 mics)

        dataset_distance_micAmplitude = np.vstack((dataset_distance_micAmplitude, mic_max_amplitude_all_trials_normalized))

    print(f"dataset_distance_micAmplitude shape: {dataset_distance_micAmplitude.shape}") #--> (N trials, 7 mics)
    
    visualize_amplitude_across_mic(dataset_distance_micAmplitude)

    data_list = []
    #get only the indices of interest from mics_to_plot from all trials
    for i in range(number_of_mics):
        data_list.append(mics_all_trials[0][i])

    microphone_utils.grid_plot_time_domain(data_list, fs)

def load_data_for_timeshift():
    save_path_data = "/home/mark/audio_learning_project/data/franka_2D_localization_full_UMC/"
    total_trial_count = 50 #50 distance x 15 radian intervals x 5 repeat counts = 3750 trials
    devicelist=[2]
    number_of_mics = 6
    fs = 44100

    # Define the ranges
    ranges0 = [(0, 50), (750, 800), (1500, 1550), (2250, 2300), (3000, 3050)] #trial ranges that are the same regions of the dataset
    ranges1 = [(50, 100), (800, 850), (1550, 1600), (2300, 2350), (3050, 3100)] #--> direct hit over the mic (drastic peak)
    ranges2 = [(100, 150), (850, 900), (1600, 1650), (2350, 2400), (3100, 3150)] 
    ranges3 = [(150, 200), (900, 950), (1650, 1700), (2400, 2450), (3150, 3200)] 
    ranges4 = [(200, 250), (950, 1000), (1700, 1750), (2450, 2500), (3200, 3250)]
    ranges5 = [(250, 300), (1000,1050), (1750,1800), (2500,2550), (3250,3300)] #-->-44 deg
    ranges6 = [(300, 350), (1050,1100), (1800,1850), (2550,2600), (3300,3350)]
    ranges7 = [(350, 400), (1100,1150), (1850,1900), (2600,2650), (3350,3400)]
    ranges8 = [(400, 450), (1150,1200), (1900,1950), (2650,2700), (3400,3450)]
    ranges9 = [(450, 500), (1200,1250), (1950,2000), (2700,2750), (3450,3500)]
    ranges10 = [(500, 550), (1250,1300), (2000,2050), (2750,2800), (3500,3550)]
    ranges11 = [(550, 600), (1300,1350), (2050,2100), (2800,2850), (3550,3600)]
    ranges12 = [(600, 650), (1350,1400), (2100,2150), (2850,2900), (3600,3650)]
    ranges13 = [(650, 700), (1400,1450), (2150,2200), (2900,2950), (3650,3700)]
    ranges14 = [(700, 750), (1450,1500), (2200,2250), (2950,3000), (3700,3750)]

    # ranges = ranges0 + ranges1 + ranges2 + ranges3 + ranges4 + ranges5 + ranges6 + ranges7 + ranges8 + ranges9 + ranges10 + ranges11 + ranges12 + ranges13 + ranges14
    ranges = ranges5
    num_trials = 50

    dataset_distance_micAmplitude = np.empty((0, 4))

    # Load the wav files for each range
    np.set_printoptions(precision=2)
    for trial_start,trial_end in ranges:
        mics_all_trials, mic_max_amplitude_all_trials, mic_max_index_all_trials = microphone_utils.load_wav_files_from_dataset_sections(devicelist, trial_start, trial_end, save_path_data)

        #convert list to np
        mic_max_index_all_trials = np.array(mic_max_index_all_trials)
        #squeeze the 1st dimension
        mic_max_index_all_trials = np.squeeze(mic_max_index_all_trials)

        print(f" mic_max_index_all_trials shape: {(mic_max_index_all_trials.shape)}") #--> (N trials, 6 mics)

        #get the difference between (mic0,3)
        mic_max_index_diff03 = mic_max_index_all_trials[:,3] - mic_max_index_all_trials[:,0]
        #fs = 44100, so divide by fs to get time in seconds
        mic_max_index_diff03 = mic_max_index_diff03 / fs * 1000 #convert to ms
        #remove outliers where mic_max_index_diff03 absolute value is greater than 100
        mic_max_index_diff03 = np.where(np.abs(mic_max_index_diff03) > 100, 0, mic_max_index_diff03)
        print(f"mic_max_index_diff03 {mic_max_index_diff03}")

        


        #get the difference between (mic1,4)
        mic_max_index_diff14 = mic_max_index_all_trials[:,4] - mic_max_index_all_trials[:,1]
        #fs = 44100, so divide by fs to get time in seconds
        mic_max_index_diff14 = mic_max_index_diff14 / fs * 1000
        #remove outliers where mic_max_index_diff03 absolute value is greater than 100
        mic_max_index_diff14 = np.where(np.abs(mic_max_index_diff14) > 100, 0, mic_max_index_diff14)
        print(f"mic_max_index_diff14 {mic_max_index_diff14}")

        


        #get the difference between (mic2,5)
        mic_max_index_diff25 = mic_max_index_all_trials[:,5] - mic_max_index_all_trials[:,1]
        #fs = 44100, so divide by fs to get time in seconds
        mic_max_index_diff25 = mic_max_index_diff25 / fs * 1000
        #remove outliers where mic_max_index_diff03 absolute value is greater than 100
        mic_max_index_diff25 = np.where(np.abs(mic_max_index_diff25) > 100, 0, mic_max_index_diff25)
        print(f"mic_max_index_diff25 {mic_max_index_diff25}")

        


        #evenly space distance from 0 to 20 by num_trials
        distance = np.linspace(0, 20, num_trials)

        #reshape mic_max_index_diff03 to (N trials, 1)
        mic_max_index_diff03 = mic_max_index_diff03.reshape(-1,1)
        mic_max_index_diff14 = mic_max_index_diff14.reshape(-1,1)
        mic_max_index_diff25 = mic_max_index_diff25.reshape(-1,1)

        #concat into (N tirals, 3)
        mic_max_index_diff03 = np.concatenate([mic_max_index_diff03, mic_max_index_diff14, mic_max_index_diff25], axis=1)

        print(f"mic_max_index_diff03 shape before: {mic_max_index_diff03.shape}") #--> (N trials, 3)

        #concatenate distance to mic_max_amplitude_all_trials_normalized at the 0th column
        mic_max_index_diff03 = np.concatenate([distance[:, np.newaxis], mic_max_index_diff03], axis=1)

        print(f"mic_max_index_diff03 shape: {mic_max_index_diff03.shape}") #--> (N trials, 4)

        dataset_distance_micAmplitude = np.vstack((dataset_distance_micAmplitude, mic_max_index_diff03))

        
         
    print(f"dataset_distance_micAmplitude shape: {dataset_distance_micAmplitude.shape}") #--> (N trials, 4 mics)
    
    visualize_timeshift_across_mic(dataset_distance_micAmplitude)

def load_data_for_timeshift_across_ring():
    save_path_data = "/home/mark/audio_learning_project/data/franka_2D_localization_full_UMC/"
    total_trial_count = 50 #50 distance x 15 radian intervals x 5 repeat counts = 3750 trials
    devicelist=[2]
    number_of_mics = 6
    fs = 44100

    # Define the ranges
    ranges0 = [(0, 50), (750, 800), (1500, 1550), (2250, 2300), (3000, 3050)] #trial ranges that are the same regions of the dataset
    ranges1 = [(50, 100), (800, 850), (1550, 1600), (2300, 2350), (3050, 3100)] #--> direct hit over the mic (drastic peak)
    ranges2 = [(100, 150), (850, 900), (1600, 1650), (2350, 2400), (3100, 3150)] 
    ranges3 = [(150, 200), (900, 950), (1650, 1700), (2400, 2450), (3150, 3200)] 
    ranges4 = [(200, 250), (950, 1000), (1700, 1750), (2450, 2500), (3200, 3250)]
    ranges5 = [(250, 300), (1000,1050), (1750,1800), (2500,2550), (3250,3300)] #-->-44 deg
    ranges6 = [(300, 350), (1050,1100), (1800,1850), (2550,2600), (3300,3350)]
    ranges7 = [(350, 400), (1100,1150), (1850,1900), (2600,2650), (3350,3400)]
    ranges8 = [(400, 450), (1150,1200), (1900,1950), (2650,2700), (3400,3450)]
    ranges9 = [(450, 500), (1200,1250), (1950,2000), (2700,2750), (3450,3500)]
    ranges10 = [(500, 550), (1250,1300), (2000,2050), (2750,2800), (3500,3550)]
    ranges11 = [(550, 600), (1300,1350), (2050,2100), (2800,2850), (3550,3600)]
    ranges12 = [(600, 650), (1350,1400), (2100,2150), (2850,2900), (3600,3650)]
    ranges13 = [(650, 700), (1400,1450), (2150,2200), (2900,2950), (3650,3700)]
    ranges14 = [(700, 750), (1450,1500), (2200,2250), (2950,3000), (3700,3750)]

    # ranges = ranges0 + ranges1 + ranges2 + ranges3 + ranges4 + ranges5 + ranges6 + ranges7 + ranges8 + ranges9 + ranges10 + ranges11 + ranges12 + ranges13 + ranges14
    ranges = ranges2
    num_trials = 50

    dataset_distance_micAmplitude = np.empty((0, 5))

    # Load the wav files for each range
    np.set_printoptions(precision=2)
    for trial_start,trial_end in ranges:
        mics_all_trials, mic_max_amplitude_all_trials, mic_max_index_all_trials = microphone_utils.load_wav_files_from_dataset_sections(devicelist, trial_start, trial_end, save_path_data)

        #convert list to np
        mic_max_index_all_trials = np.array(mic_max_index_all_trials)
        #squeeze the 1st dimension
        mic_max_index_all_trials = np.squeeze(mic_max_index_all_trials)

        print(f" mic_max_index_all_trials shape: {(mic_max_index_all_trials.shape)}") #--> (N trials, 6 mics)

        #get the difference between (mic0,3)
        mic_max_index_diff34 = mic_max_index_all_trials[:,3] - mic_max_index_all_trials[:,4]
        #fs = 44100, so divide by fs to get time in seconds
        mic_max_index_diff34 = mic_max_index_diff34 / fs * 1000 #convert to ms
        #remove outliers where mic_max_index_diff03 absolute value is greater than 100
        mic_max_index_diff34 = np.where(np.abs(mic_max_index_diff34) > 100, 0, mic_max_index_diff34)
        print(f"mic_max_index_diff03 {mic_max_index_diff34}")

        


        #get the difference between (mic1,4)
        mic_max_index_diff35 = mic_max_index_all_trials[:,3] - mic_max_index_all_trials[:,5]
        #fs = 44100, so divide by fs to get time in seconds
        mic_max_index_diff35 = mic_max_index_diff35 / fs * 1000
        #remove outliers where mic_max_index_diff03 absolute value is greater than 100
        mic_max_index_diff35 = np.where(np.abs(mic_max_index_diff35) > 100, 0, mic_max_index_diff35)
        print(f"mic_max_index_diff14 {mic_max_index_diff35}")

        


        #get the difference between (mic0,1)
        mic_max_index_diff01 = mic_max_index_all_trials[:,0] - mic_max_index_all_trials[:,1]
        #fs = 44100, so divide by fs to get time in seconds
        mic_max_index_diff01 = mic_max_index_diff01 / fs * 1000
        #remove outliers where mic_max_index_diff03 absolute value is greater than 100
        mic_max_index_diff01 = np.where(np.abs(mic_max_index_diff01) > 100, 0, mic_max_index_diff01)
        print(f"mic_max_index_diff25 {mic_max_index_diff01}")

        #get the difference between (mic0,2)
        mic_max_index_diff02 = mic_max_index_all_trials[:,0] - mic_max_index_all_trials[:,2]
        #fs = 44100, so divide by fs to get time in seconds
        mic_max_index_diff02 = mic_max_index_diff02 / fs * 1000
        #remove outliers where mic_max_index_diff03 absolute value is greater than 100
        mic_max_index_diff02 = np.where(np.abs(mic_max_index_diff02) > 100, 0, mic_max_index_diff02)
        print(f"mic_max_index_diff25 {mic_max_index_diff02}")

        


        #evenly space distance from 0 to 20 by num_trials
        distance = np.linspace(0, 20, num_trials)

        #reshape mic_max_index_diff03 to (N trials, 1)
        mic_max_index_diff34 = mic_max_index_diff34.reshape(-1,1)
        mic_max_index_diff35 = mic_max_index_diff35.reshape(-1,1)
        mic_max_index_diff01 = mic_max_index_diff01.reshape(-1,1)
        mic_max_index_diff02 = mic_max_index_diff02.reshape(-1,1)


        #concat into (N tirals, 3)
        mic_max_index_diff03 = np.concatenate([mic_max_index_diff34, mic_max_index_diff35, mic_max_index_diff01, mic_max_index_diff02], axis=1)

        print(f"mic_max_index_diff03 shape before: {mic_max_index_diff03.shape}") #--> (N trials, 3)

        #concatenate distance to mic_max_amplitude_all_trials_normalized at the 0th column
        mic_max_index_diff03 = np.concatenate([distance[:, np.newaxis], mic_max_index_diff03], axis=1)

        print(f"mic_max_index_diff03 shape: {mic_max_index_diff03.shape}") #--> (N trials, 4)

        dataset_distance_micAmplitude = np.vstack((dataset_distance_micAmplitude, mic_max_index_diff03))

        
         
    print(f"dataset_distance_micAmplitude shape: {dataset_distance_micAmplitude.shape}") #--> (N trials, 4 mics)
    
    visualize_timeshift_across_mic_rings(dataset_distance_micAmplitude)


def load_data_and_plot_for_removing_noncollision():
    """
    Load all wav files from dataset 
    get mean and standard dev from dataset
    Iterate through and find, if any, samples where abs amplitude is smaller than 2 std deviation across all mic channels
    plot them to identify non-collision files
    """
    save_path_data = "/home/mark/audio_learning_project/data/wood_T22_L42_Horizontal/"
    total_trial_count = 2400 #50 distance x 15 radian intervals x 5 repeat counts = 3750 trials
    devicelist=[2]
    number_of_mics = 6
    fs = 44100

    mic_data_over_trials = microphone_utils.load_wav_files_as_dataset(devicelist, total_trial_count, save_path_data)

    print(f"mic_data_over_trials shape: {len(mic_data_over_trials)}") #--> 3750 trials
    print(f"mic_data_over_trials[0] shape: {len(mic_data_over_trials[0][0])}") #--> 1st trial, 6 mics

    max_amplitude_over_trials = []
    for trial in range(len(mic_data_over_trials)):
        all_mic_single_trial = mic_data_over_trials[trial][0] #--> 6 mics
        max_amplitude = np.max(np.abs(all_mic_single_trial), axis=1) #--> 6 mics
        max_amplitude_over_trials.append(max_amplitude)

    #convert to np
    max_amplitude_over_trials_np = np.array(max_amplitude_over_trials)
    print(f"max_amplitude_over_trials_np shape: {max_amplitude_over_trials_np.shape}") #--> 3750 trials, 6 mics

    max_amplitude_mean = np.mean(max_amplitude_over_trials_np, axis=0) #--> 6 mics
    max_amplitude_std = np.std(max_amplitude_over_trials_np, axis=0) #--> 6 mics


    for trial in range(len(mic_data_over_trials)):
        all_mic_single_trial = mic_data_over_trials[trial][0] #--> 6 mics
        #convert to np
        all_mic_single_trial = np.array(all_mic_single_trial) #--> 6 mics, 88200 samples

        #get max amplitude for each mic
        max_amplitude = np.max(np.abs(all_mic_single_trial), axis=1) #--> 6 mics

        #TODO: check if any of the mic in this sample is smaller than 2 std deviation from the max_amplitude_mean
        is_smaller = max_amplitude < (max_amplitude_mean - 1.5*max_amplitude_std)

        #TODO: plot them to identify non-collision files
        if any(is_smaller):
            print(f"trial {trial} has non-collision")
            
            #plot
            microphone_utils.grid_plot_time_domain(mic_data_over_trials[trial][0], fs)


def main():
    print(f" ------ starting script ------  ")

    # ----------------- For iterating through all wav files and detecting no collision files ------------------------------------------
    load_data_and_plot_for_removing_noncollision()
    sys.exit()

    # ----------------- For simple time domain plotting checkcing data ------------------------------------------
    # load_data_and_plot()
    # sys.exit()
    # -----------------------------------------------------------

    # ----------------- For visualizing effect of amplitude  ------------------------------------------
    # all_mic_timedomain = load_data_for_amplitude()
    # sys.exit()

    # ----------------- For visualizing effect of time of arrival ------------------------------------------
    all_mic_timedomain = load_data_for_timeshift()
    sys.exit()
    # -----------------------------------------------------------

    # ----------------- For visualizing effect of time of arrival across ring ------------------------------------------
    # all_mic_timedomain = load_data_for_timeshift_across_ring()
    # sys.exit()
    # -----------------------------------------------------------

    

    #create instance of microphone class
    devicelist=[0,1,11,12,13,14]
    number_of_mics = len(devicelist)
    channels_in = 1
    save_path_data = "/home/mark/audio_learning_project/data/franka_init_test_6mic/"

    total_trial_count = 100

    #load all wav files from dataset (all mic, all trials)
    mics_all_trials = microphone_utils.load_wav_files_as_dataset(devicelist, total_trial_count, save_path_data)
    print(f"size of mics_all_trials: {len(mics_all_trials)}") #--> 100 trials

    #sample 1 out of 10 trials from mic_all_trials
    mics_all_trials = mics_all_trials[::5]


    #iterate through all trials and only extract mic1 out of 6 mics
    for i in range(len(mics_all_trials)):
        mics_all_trials[i] = mics_all_trials[i][0]

    print(f"size of mics_all_trials: {len(mics_all_trials)}") #--> 10 trials
    
    #plot each trial
    microphone_utils.grid_plot_1mic_all_trials(mics_all_trials, 22050)

    sys.exit()

    #load all labels from dataset (all trials)
    labels_all_trials = microphone_utils.load_labels_from_dataset(save_path_data, total_trial_count)

    # print(f"labels size of all trials: {len(labels_all_trials)}") #--> 100 
    # labels for trial 0: [0. 0.]

    # get list of [X,Y] from dataset
    XY_list = microphone_utils.preprocess_data(mics_all_trials, labels_all_trials)

    print(f"size of XY_list: {len(XY_list)}") #--> 100 trials

    



if __name__ == '__main__':
    main()