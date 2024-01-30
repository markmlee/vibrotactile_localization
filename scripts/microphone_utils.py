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

def load_specific_wav_files(devicelist, mics_to_plot, save_path_data, total_trial_count):
    """
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

    #downsample data 1 out of every 1 samples
    mics_plot_over_trials = [data[::1] for data in mics_plot_over_trials]
    print(f"size of data0: {len(mics_plot_over_trials[0])}, size of data1: {len(mics_plot_over_trials[1])}")


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
