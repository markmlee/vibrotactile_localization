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

def plot_wav_files(devicelist, trial_number, load_path):
    print(f" ------ plotting wav files ------  ")

    mic_number = len(devicelist)
    data_list = []

    for i in range(mic_number):

        file_name = f"{load_path}trial{trial_number}_mic{devicelist[i]}.wav"
        data, fs = librosa.load(file_name)
        data_list.append(data)

    print(f"size of data0: {len(data_list[0])}, size of data1: {len(data_list[1])}")

    # trim data0,1,2,3,4,5 to the shortest length in for loop
    # get the minimum length of the data arrays
    min_length = min(len(data) for data in data_list)

    # trim all data arrays to the minimum length
    for i in range(mic_number):
        data_list[i] = data_list[i][:min_length]
    


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
    