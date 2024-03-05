#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from soundfile import SoundFile
from scipy.io.wavfile import read

import time
import wave
import sys
import librosa

def plot_time_domain(data_list, fs):

    mic_number = len(data_list)

    #trim to the shortest length
    min_length = min(len(data) for data in data_list)
    for i in range(mic_number):
        data_list[i] = data_list[i][:min_length]

    print(f" ------ plotting wav files ------  ")
    # plot data0 and data1 in same plot
    # Create a time array for plotting
    time = np.arange(0, len(data_list[0])) / fs

    plt.figure(figsize=(10, 6))

    #plot data_list
    for i in range(mic_number):
        plt.plot(time, data_list[i], label=f"mic{i}")


    plt.title('Audio Data')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()

def grid_plot_time_domain(data_list, fs):
    """
    subplot 3x2 grid of time domain plots. 
    First column of 3 plots should be mic0,1,2
    Second column of 3 plots should be mic3,4,5
    
    """
    
    mic_number = len(data_list)

    #trim to the shortest length
    min_length = min(len(data) for data in data_list)
    for i in range(mic_number):
        data_list[i] = data_list[i][:min_length]

    print(f" ------ plotting wav files ------  ")
    # plot data0 and data1 in same plot
    # Create a time array for plotting
    time = np.arange(0, len(data_list[0])) / fs

    fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle('Audio Data for all 6 mics')

    for i in range(mic_number):
        if i < 3:
            axs[i, 0].plot(time, data_list[i])
            axs[i, 0].set_title(f"mic{i}")
            axs[i, 0].set_ylabel('Amplitude')
            if i == 2:  # Only set x-label for the bottom plot in the first column
                axs[i, 0].set_xlabel('Time [s]')
        else:
            axs[i-3, 1].plot(time, data_list[i])
            axs[i-3, 1].set_title(f"mic{i}")
            axs[i-3, 1].set_ylabel('Amplitude')
            if i == 5:  # Only set x-label for the bottom plot in the second column
                axs[i-3, 1].set_xlabel('Time [s]')

    plt.show()

def plot_spectrogram(data, fs):
    """
    using librosa spectrogram
    """
    print(f" ------ plotting spectrogram ------  ")
    plt.figure(figsize=(10, 6))
    # Plot the spectrogram
    S = librosa.stft(data)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

def get_spectrogram(data,fs):
    """
    return img of spectrogram
    """

    S = librosa.stft(data)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='log', vmin=-80, vmax=0)

    return img

def plot_spectrogram_all_mics(data_list, fs):
    """
    plot spectrogram for all mics in a 3x2 grid, set intensity same for all (by using vmin and vmax)
    """
    print(f" ------ plotting spectrogram for all mics ------  ")

    mic_number = len(data_list)

    
    # Plot the spectrogram for all 6 mics
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Mel-Spectrogram for all 6 mics')

    for i in range(mic_number):
        # convert to spectrogram
        S = librosa.stft(data_list[i])
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img = librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='log', ax=axs[i//2, i%2], vmin=-80, vmax=0)
        axs[i//2, i%2].set_title(f"mic{i}")
        fig.colorbar(img, ax=axs[i//2, i%2], format='%+2.0f dB')



        #convert to mel-spectrogram
        # melspec = librosa.feature.melspectrogram(y=data_list[i], sr=44100, n_mels=128, fmax=22050) 

        # S_dB = librosa.power_to_db(melspec, ref=np.max)
        # img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=44100, fmax=22050, ax=axs[i//2, i%2], vmin=-80, vmax=0)
        # axs[i//2, i%2].set_title(f"mic{i}")
        # fig.colorbar(img, ax=axs[i//2, i%2], format='%+2.0f dB')


    plt.show()




def plot_wav_files(devicelist, trial_number, load_path, fs=44100):
    """
    single trial multiple wav files are plotted in the same plot
    """
    print(f" ------ plotting wav files ------  ")

    mic_number = len(devicelist)
    data_list = []

    for i in range(mic_number):

        file_name = f"{load_path}trial{trial_number}_mic{devicelist[i]}.wav"
        data, _ = librosa.load(file_name)
        fs = fs
        data_list.append(data)

    print(f"size of data0: {len(data_list[0])}, size of data1: {len(data_list[1])}")

    # trim data0,1,2,3,4,5 to the shortest length in for loop
    # get the minimum length of the data arrays
    min_length = min(len(data) for data in data_list)

    # trim all data arrays to the minimum length
    for i in range(mic_number):
        data_list[i] = data_list[i][:min_length]
    

    plot_time_domain(data_list, fs)

def load_labels_from_dataset(load_path, trial_count):
    """
    return a list of ground truth labels for all trials
    [ [distance along cylinder, joint 6] ... [distance along cylinder, joint 6] ]
    """

    labels_over_trials = []

    for trial_number in range(trial_count):
        
        label_file = f"{load_path}trial{trial_number}/gt_label.npy"
        label = np.load(label_file) #--> [distance along cylinder, joint 6]
        labels_over_trials.append(label)

    return labels_over_trials

def load_wav_files_from_dataset(devicelist, trial_count, load_path):
    """
    return a list of concatenated wav files for all mics for all trials
    """

    mic_data_over_trials = []

    for mic in devicelist:
        #concat 1 mic over all trials
        mic_data_all_trials = concat_wav_files_dataset(load_path, trial_count, mic)
        mic_data_over_trials.append(mic_data_all_trials)

    return mic_data_over_trials

def load_wav_files_as_dataset(devicelist, trial_count, load_path):
    """
    return a list of wav files for all mics for all trials (WITHOUT concatenating them)
    should return [ [mic1...mic6] x trial_count ]
    """

    mic_data_over_trials = []

    for trial in range(trial_count):

        mic_data_single_trial = []

        for mic in devicelist:

            file_name = f"{load_path}trial{trial}/mic{mic}.wav"
            data, fs = librosa.load(file_name, sr=44100)
            mic_data_single_trial.append(data)

        mic_data_over_trials.append(mic_data_single_trial)

    return mic_data_over_trials



def load_specific_wav_files(devicelist, mics_to_plot, save_path_data, total_trial_count):
    """
    return a list of concatenated wav files for all trials ONLY the mics of interest
    """

    mic_data_over_trials = []

    for mic in mics_to_plot:
        #concat 1 mic over all trials
        mic_id = devicelist[mic]
        mic_data_all_trials = concat_wav_files(save_path_data,  total_trial_count , mic_id)
        mic_data_over_trials.append(mic_data_all_trials)

    return mic_data_over_trials

def plot_specific_wav_files(devicelist, mics_to_plot, save_path_data, total_trial_count, fs):
    """
    multiple trial wav files are concatenated then plotted in the same plot
    """

    mics_plot_over_trials = load_specific_wav_files(devicelist, mics_to_plot, save_path_data, total_trial_count)

    # plot data0 and data1 in same plot
    plot_time_domain(mics_plot_over_trials, fs)


def plot_envelope_ratio(data_list):
    ratio = []
    
    first_mic = data_list[0]
    second_mic = data_list[1]

    #divide mic data into 10 even intervals, and return the max value of each interval
    interval = len(first_mic)//10
    for i in range(0, len(first_mic), interval):
        ratio.append(max(first_mic[i:i+interval])/max(second_mic[i:i+interval]))


    frames = range(0, len(first_mic))
    #transparecy of the plot is set to 0.5
    plt.plot(frames, first_mic, second_mic, alpha=0.5)
    plt.title('Amplitude of 2 Mics')
    plt.xlabel('bins [50 msec]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    plt.plot(ratio)
    plt.title('Ratio of Mic1/Mic2 as tapped across PVC length')
    plt.xlabel('Contact count')
    plt.ylabel('Ratio of Amplitude')
    plt.show()


def plot_envelope(frame_size, hop_length, devicelist, mics_to_plot, save_path_data, total_trial_count):
    """
    """
    

    mics_plot_over_trials = load_specific_wav_files(devicelist, mics_to_plot, save_path_data, total_trial_count)

    number_of_mics = len(mics_plot_over_trials)

    mics_envelope_over_trials = []
    for i in range(number_of_mics):
        envelope = amplitude_envelope(mics_plot_over_trials[i], frame_size, hop_length)
        mics_envelope_over_trials.append(envelope)

    # plot envelope ratio
    # plot_envelope_ratio(mics_envelope_over_trials)    

    # Generate the frame indices
    frames = range(0, len(mics_envelope_over_trials[0]))  
    # Convert frames to time
    time = librosa.frames_to_time(frames, hop_length=hop_length)  
    # Create a new figure with a specific size
    plt.figure(figsize=(15, 7))  
    # Display the waveform of the signal
    librosa.display.waveshow(mics_plot_over_trials[0], alpha=0.5)
    # Plot the amplitude envelope over time
    plt.plot(time, mics_envelope_over_trials[0], color="r") 
    # Set the title of the plot
    plt.title("Waveform for (Amplitude Envelope)")  
    # Show the plot
    plt.show()  


def concat_wav_files_dataset(load_path, trial_count,mic_id):
    """
    in: directory, trial number, mic number to concatenate
    output: concatenated wav file
    """

    concat_data = []

    #for all trials, find the wav file and concatenate
    for trial_number in range(trial_count):

        file_name = f"{load_path}trial{trial_number}/mic{mic_id}.wav"
        data, fs = librosa.load(file_name)
        concat_data.append(data)
        
        
    #convert len of 10 list to 1D array
    concat_data = np.concatenate(concat_data, axis=0)
    return concat_data

def concat_wav_files(load_path, trial_count,mic_id):
    """
    in: directory, trial number, mic number to concatenate
    output: concatenated wav file
    """

    concat_data = []

    #for all trials, find the wav file and concatenate
    for trial_number in range(trial_count):

        file_name = f"{load_path}trial{trial_number}_mic{mic_id}.wav"
        data, fs = librosa.load(file_name)
        concat_data.append(data)
        
        
    #convert len of 10 list to 1D array
    concat_data = np.concatenate(concat_data, axis=0)
    return concat_data

def amplitude_envelope(signal, frame_size=1024, hop_length=512):
    """
    Computes the Amplitude Envelope of a signal using a sliding window.

    Args:
        signal (array): The input signal.
        frame_size (int): The size of each frame in samples.
        hop_length (int): The number of samples between consecutive frames.

    Returns:
        np.array: An array of Amplitude Envelope values.
    """
    res = []
    for i in range(0, len(signal), hop_length):
        # Get a portion of the signal
        cur_portion = signal[i:i + frame_size]  
        # Compute the maximum value in the portion
        ae_val = max(cur_portion)  
        # Store the amplitude envelope value
        res.append(ae_val)  
    # Convert the result to a NumPy array
    output = np.array(res)

    #print size of signal and output
    print(f"size of signal: {len(signal)}, size of output: {len(output)}")
    return output


def trim_audio_signal(audio_signals, fs=44100, start_time=0.25, end_clip_time=0.25):
    """
    in: audio signal, fs, start time, end time
    out: trimmed audio signal
    """

    start = int(start_time*fs)
    end = int((len(audio_signals[0]) - end_clip_time*fs))

    for i in range(len(audio_signals)):
        audio_signals[i] = audio_signals[i][start:end]

    return audio_signals

def trim_to_same_length(data_list):
    """
    in: list of audio signals
    out: list of audio signals trimmed to the shortest length
    """
    min_length = min(len(data) for data in data_list)
    for i in range(len(data_list)):
        data_list[i] = data_list[i][:min_length]

    return data_list

def preprocess_data(mic_signals_from_all_trials, GT_labels):
    """
    1. load inputs
    2. downsample 44khz --> 11khz
    3. convert to spectrogram
    4. concat all 6 spectrograms 
    5. return list of [X,Y]  
    """ 
    print(f" ------ preprocessing data ------  ")


    X_list = []
    Y_list = []

    # print(f"mic_signals_from_all_trials shape: {len(mic_signals_from_all_trials)}") # --> 100 trials 
    #print shape of first trial
    # print(f"mic_signals_from_all_trials[0] shape: {len(mic_signals_from_all_trials[0])}") # --> 6 mics


    #enumerate through all trials

    for index, all_mics_single_trial in enumerate(mic_signals_from_all_trials): #0 to 100 trials, mic contains all 6 mic data

        
        
        trimmed_length_all_mics_single_trial = trim_to_same_length(all_mics_single_trial)
        trimmed_all_mics_single_trial = trim_audio_signal(trimmed_length_all_mics_single_trial, fs=44100, start_time=0.6, end_clip_time=0.8)

        # visualize all 6 mics in a single trial
        plot_time_domain(trimmed_all_mics_single_trial, 44100)
        plot_spectrogram_all_mics(trimmed_all_mics_single_trial, 44100)

        sys.exit()


        melspecs = []

        for mic in trimmed_all_mics_single_trial:
            # print(f"mic shape: {mic.shape}") #--> (44100,)            
            # downsampled_single_trial = librosa.resample(mic, orig_sr=44100, target_sr=11025) 
            # print(f"downsampled_single_trial shape: {downsampled_single_trial.shape}") #--> (110250,)

            plot_spectrogram(mic, 11025)

            # -------------------------------------
            # melspec_img = get_spectrogram(mic, 44100)

            # #plot quadmesh of melspec_img
            # plt.figure(figsize=(10, 6))
            # plt.title('Mel-Spectrogram')
            # plt.imshow(melspec_img)
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()

            # print(f" size of melspec_img: {melspec_img.shape}")
            # sys.exit()
            # melspecs.append(melspec_img)

            #concat all 6 spectrograms
            # X = np.concatenate(melspecs, axis=0)

            # -------------------------------------
            
            
        
        

        # print(f"size of X: {X.shape}, size of GT: {GT_labels[index].shape}") #--> (768, 216)
        #Y is only the first element of GT_labels
        Y = GT_labels[index][0]

        X_list.append(X)
        Y_list.append(Y)


    #return list of [X,Y]
    return [X_list, Y_list]

