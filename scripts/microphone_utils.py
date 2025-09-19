#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys

from itertools import combinations


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
import torch
import torchaudio
import os
import torch.nn.functional as F
import noisereduce as nr




def animate_ylabel(X_mic_data,Y_label_data):
    """
    plot all height,x,y in 3D scatter plot.
    and then animate each h,x,y individually over the 3D scatter plot.
    Y_label_data: [N,3] tensor
    move on to the next animation after keyboard input "enter", exit with "exit"
    """
    
    #get height, x, y
    height = Y_label_data[:,0]
    x = Y_label_data[:,1]
    y = Y_label_data[:,2]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    #plot all height, x, y
    ax1.scatter(x, y, height, c='r', marker='o', label='height, x, y')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    #plot height, x, y individually
    for i in range(len(height)):
        
        #plot waveform of mic data in separate figure
        ax2.cla()
        ax2.plot(X_mic_data[i])
        plt.pause(0.1)

        ax1.cla()
        ax1.scatter(x, y, height, c='r', marker='o', label='height, x, y')  # plot all points again
        ax1.scatter(x[i], y[i], height[i], c='b', marker='x', s=100, label='height, x, y')  # plot individual point
        plt.pause(0.01)

        #wait for user input
        print(f"trial: {i}, height: {height[i]}, x: {x[i]}, y: {y[i]}")
        user_input = input("Press Enter to continue or type 'exit' to quit: ").strip()
        if user_input.lower() == 'exit':
            break

    plt.show()
    

def verify_dataset(cfg, data_dir):
    """
    load all wav files
    plot all wav files
    interactively delete the bad files
    """

    
    dir_raw = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
    #filter out directory that does not start with 'trial'
    dir_raw = [d for d in dir_raw if d.split('/')[-1].startswith('trial')]
    len_data = len(dir_raw)

    count = 0
    #get all directory path to trials
    data_dir = f"{data_dir}trial"
    dir = []
    while len(dir) < len_data:
        file_name = f"{data_dir}{count}"

        if file_name in dir_raw:
            dir.append(file_name)
        count += 1
    
    #load data (for multiple mics in device list, get wav files)
    len_data = len(dir)
    
    X_mic_data = []
    Y_label_data = []

    for trial_n in range(len_data):

        wav_filename = f"{dir[trial_n]}/mic{cfg.device_list[0]}.wav"
        goal_joint = f"{dir[trial_n]}/goal_j1_angle.npy"
        #load np
        goal_joint = np.load(goal_joint)
        # print(f"goal_joint: {goal_joint}")

        # print(f"loading wav file: {wav_filename}")

        wav, sample_rate = torchaudio.load(wav_filename)
        sample_rate = sample_rate

        #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
        if wav.size(1) < cfg.max_num_frames:
            wav = F.pad(wav, (0, cfg.max_num_frames - wav.size(1)), mode='circular'   )
        else:
            wav = wav[:, :cfg.max_num_frames]

        #append to list of wavs
        wav = (wav.squeeze(0)) # remove the dimension of size 1

        

        #get label from directory 
        label_file = f"{dir[trial_n]}/gt_label.npy"
        label = np.load(label_file) #--> [distance along cylinder, joint 6] i.e. [0.0 m, -2.7 radian]

        #convert label m unit to cm
        label[0] = label[0] * 100

        #normalize height to [0,1], and radian to [0,1]
        # label[0] = label[0] / 20.32 #(8" to 20cm)
        # label[1] = (label[1] + np.pi) / (2*np.pi) #[-pi,pi] to [0,1]

        # #convert radian to x,y coordinate with unit 1
        x,y  = np.cos(label[1]), np.sin(label[1])

        x_data, y_label = wav, [label[0], x, y]

        X_mic_data.append(x_data)
        Y_label_data.append(y_label)

    # stack wav files into a tensor of shape (num_mics, num_samples)
    X_mic_data = torch.stack(X_mic_data, dim=0)
    print(f"dimension of X input tensor: {X_mic_data.size()}") #--> dimension of wav tensor: torch.Size([N, 88200])
    Y_label_data = torch.tensor(Y_label_data)
    print(f"dimension of Y label tensor: {Y_label_data.size()}") #--> dimension of label tensor: torch.Size([N, 3])

    animate_ylabel(X_mic_data,Y_label_data)

    sys.exit()



def plot_spectrogram_with_cfg(cfg, data, fs):
    """
    subplot 3x2 grid of time domain plots. 
    First column of 3 plots should be mic0,1,2
    Second column of 3 plots should be mic3,4,5
    """


    # Plot the spectrogram for all 6 mics
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Mel-Spectrogram for all 6 mics')

    for i,S in enumerate((data)):
        if i < 3:
            axs[i, 0].set_title(f"mic{i}")
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, ax=axs[i, 0], vmin=-80, vmax=0, 
                                           hop_length=cfg.hop_length, n_fft=cfg.n_fft)
            fig.colorbar(img, ax=axs[i, 0], format='%+2.0f dB')

            
            if i == 2:
                axs[i, 0].set_xlabel('Time [s]')
        else:
            axs[i-3, 1].set_title(f"mic{i}")
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB,x_axis='time', y_axis='mel', sr=fs, ax=axs[i-3, 1], vmin=-80, vmax=0, 
                                           hop_length=cfg.hop_length, n_fft=cfg.n_fft)
            fig.colorbar(img, ax=axs[i-3, 1], format='%+2.0f dB')

            if i == 5:
                axs[i-3, 1].set_xlabel('Time [s]')
        
    # plt.show()


    
def save_spectrogram_with_cfg(cfg, data, fs, name):
    """
    subplot 3x2 grid of time domain plots. 
    First column of 3 plots should be mic0,1,2
    Second column of 3 plots should be mic3,4,5
    """


    # Plot the spectrogram for all 6 mics
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Mel-Spectrogram for all 6 mics')

    for i,S in enumerate((data)):
        if i < 3:
            axs[i, 0].set_title(f"mic{i}")
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, ax=axs[i, 0], vmin=-80, vmax=0, 
                                           hop_length=cfg.hop_length, n_fft=cfg.n_fft)
            fig.colorbar(img, ax=axs[i, 0], format='%+2.0f dB')

            
            if i == 2:
                axs[i, 0].set_xlabel('Time [s]')
        else:
            axs[i-3, 1].set_title(f"mic{i}")
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB,x_axis='time', y_axis='mel', sr=fs, ax=axs[i-3, 1], vmin=-80, vmax=0, 
                                           hop_length=cfg.hop_length, n_fft=cfg.n_fft)
            fig.colorbar(img, ax=axs[i-3, 1], format='%+2.0f dB')

            if i == 5:
                axs[i-3, 1].set_xlabel('Time [s]')

    #save the plot into image with name
    plt.savefig(f"{cfg.checkpoint_dir}/latentdim{name}.png")


def plot_single_fft(waves, sample_rate, device_list):
    """
    Average the normalized FFT of 6 different mics to get 1 output FFT
    """
    num_mic = len(waves)
    assert num_mic == 6, "The number of input wav tensors should be 6."

    # Initialize sum of magnitudes
    sum_magnitude = None

    # Apply FFT to signal to determine the frequency content
    for i in range(num_mic):
        # Ensure the tensor is on the appropriate device
        wave = waves[i]
        
        # Compute the FFT
        fft = torch.fft.rfft(wave)
        magnitude = torch.abs(fft)
        
        # Normalize the magnitude
        magnitude /= magnitude.max()
        
        # Initialize sum_magnitude with the shape of the first magnitude tensor
        if sum_magnitude is None:
            sum_magnitude = torch.zeros_like(magnitude)
        
        # Sum the normalized magnitudes
        sum_magnitude += magnitude

    # Compute the average normalized magnitude
    avg_magnitude = sum_magnitude / num_mic
    
    # Generate frequency axis
    frequency = torch.fft.rfftfreq(waves[0].shape[0], d=1/sample_rate)

    # Filter out frequencies below 100 Hz (motor noise) and above 1000 Hz (robot jerk noise) 
    valid_indices = (frequency >= 100) & (frequency <= 1000)
    filtered_frequencies = frequency[valid_indices]
    filtered_magnitude = avg_magnitude[valid_indices]

    # Find the frequency with the maximum amplitude
    max_index = torch.argmax(filtered_magnitude)
    max_frequency = filtered_frequencies[max_index].item()
    max_amplitude = filtered_magnitude[max_index].item()

    # Print the frequency with the maximum amplitude
    print(f'Maximum amplitude at frequency: {max_frequency} Hz')

    # Plot the average FFT
    plt.figure(figsize=(10, 6))
    plt.plot(frequency.cpu().numpy(), avg_magnitude.cpu().numpy())
    
    # Plot vertical red dashed line at the frequency with maximum amplitude
    plt.axvline(x=max_frequency, color='r', linestyle='--', label=f'Max Frequency: {max_frequency} Hz')
    
    # Set x-axis limit to 1000 Hz
    plt.xlim(0, 1000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Average Frequency Components via Normalized FFT')
    plt.legend()
    plt.grid()
    plt.show()

def plot_fft(waves, sample_rate, device_list):
        """
        plot the fft of the waves to visualize background freq or signal freq components
        """
        background_fft = []

        # Apply FFT to background noise to determine the frequency content
        for i in range(len(waves)):
            fft = torch.fft.rfft(waves[i])
            background_fft.append(fft)

        # Plot 6 subplots of background noise to see freq components
        fig, axs = plt.subplots(6, 1, figsize=(10, 15))
        fig.suptitle('Frequency Components of Background Noise')
        
        for i, ax in enumerate(axs):
            if i < len(waves):
                # Calculate magnitude of FFT and frequency bins
                magnitude = torch.abs(background_fft[i])
                frequency = torch.fft.rfftfreq(waves[i].shape[0], d=1/sample_rate)
                
                 # Plotting with black line color
                ax.plot(frequency.numpy(), magnitude.numpy(), color='black')
                # ax.set_title(f'Mic {device_list[i]}', color='black')
                ax.set_xlabel('Frequency (Hz)', color='black')
                ax.set_ylabel('Magnitude', color='black')
                ax.tick_params(colors='black')
            else:
                ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()

        #plot FFT waves again but limit xrange to 0-2,000 Hz
        fig, axs = plt.subplots(6, 1, figsize=(10, 15))
        fig.suptitle('Frequency Components of Background Noise (0-2,000 Hz)')


        for i, ax in enumerate(axs):
            if i < len(waves):
                # Calculate magnitude of FFT and frequency bins
                magnitude = torch.abs(background_fft[i])
                frequency = torch.fft.rfftfreq(waves[i].shape[0], d=1/sample_rate)
                
                # Plotting with black line color
                ax.plot(frequency.numpy(), magnitude.numpy(), color='black')
                # ax.set_title(f'Mic {device_list[i]}', color='black')
                ax.set_xlabel('Frequency (Hz)', color='black')
                ax.set_ylabel('Magnitude', color='black')
                ax.tick_params(colors='black')
                ax.set_xlim(0, 2000)
            else:
                ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

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

    plt.ylim(-1, 1)
    plt.legend()

    plt.show()
    

def grid_plot_1mic_all_trials(data_list, fs):
    """
    plot 1 mic for all trials in a 10x1 grid
    """

    trial_number = len(data_list)
    print(f"trial_number: {trial_number}")

    #trim to the shortest length
    min_length = min(len(data_list[i]) for i in range(trial_number))
    for i in range(trial_number):
        data_list[i] = data_list[i][:min_length]

    print(f" ------ plotting wav files ------  ")
    # Plot the time domain for all 10 trials, mic1
    fig, axs = plt.subplots(trial_number, 1, figsize=(15, 10), tight_layout=True)
    fig.suptitle('Audio Data for all 100 trials [::5], mic1')

    for i in range(trial_number):
        axs[i].plot(data_list[i])
        axs[i].set_title(f"trial{i}")
        axs[i].set_ylabel('Amplitude')
        if i == 9:
            axs[i].set_xlabel('Time [s]')

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
            axs[i, 1].plot(time, data_list[i])
            axs[i, 1].set_title(f"mic{i}")
            axs[i, 1].set_ylabel('Amplitude')
            if i == 2:  # Only set x-label for the bottom plot in the first column
                axs[i, 1].set_xlabel('Time [s]')
        else:
            axs[i-3, 0].plot(time, data_list[i])
            axs[i-3, 0].set_title(f"mic{i}")
            axs[i-3, 0].set_ylabel('Amplitude')
            if i == 5:  # Only set x-label for the bottom plot in the second column
                axs[i-3, 0].set_xlabel('Time [s]')

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
    print(f"data shape: {data.shape}")
    print(f"data: {data}")

    #convert tensor to numpy array
    data = data.numpy()

    S = librosa.stft(data)
    # S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    S_db = librosa.power_to_db(S, ref=np.max)
    ax = librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='log', vmin=-80, vmax=0)

    return S_db, ax

def plot_spectrogram_all_mics(data_list, fs):
    """
    plot spectrogram for all mics in a 3x2 grid, set intensity same for all (by using vmin and vmax)
    input: time domain, output: mel-spectrogram
    """
    print(f" ------ plotting spectrogram for all mics ------  ")

    mic_number = len(data_list)

    
    # Plot the spectrogram for all 6 mics
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Mel-Spectrogram for all 6 mics')

    for i in range(mic_number):
        # convert to spectrogram
        S_db, ax = get_spectrogram(data_list[i], fs)
        img = axs[i//2, i%2].imshow(S_db, aspect='auto', cmap='inferno', origin='lower')
        axs[i//2, i%2].set_title(f"mic{i}")
        fig.colorbar(img, ax=axs[i//2, i%2], format='%+2.0f dB')



    plt.show()


def grid_plot_spectrogram(data_list, fs):
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

    print(f" ------ plotting spectrogram ------  ")
    # Plot the spectrogram for all 6 mics
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Mel-Spectrogram for all 6 mics')

    for i in range(mic_number):
        # convert to spectrogram
        if i < 3:
            axs[i, 0].set_title(f"mic{i}")
            S_dB, ax = get_spectrogram(data_list[i], fs)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, ax=axs[i, 0], vmin=-80, vmax=0, 
                                           hop_length=64, n_fft=512)
            fig.colorbar(img, ax=axs[i, 0], format='%+2.0f dB')

            if i == 2:
                axs[i, 0].set_xlabel('Time [s]')
        else:
            axs[i-3, 1].set_title(f"mic{i}")
            S_dB, ax = get_spectrogram(data_list[i], fs)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, ax=axs[i-3, 1], vmin=-80, vmax=0, 
                                           hop_length=64, n_fft=512)
            fig.colorbar(img, ax=axs[i-3, 1], format='%+2.0f dB')

            if i == 5:
                axs[i-3, 1].set_xlabel('Time [s]')

        
    plt.show()

   
   

def plot_spectrogram_of_all_data(cfg, data, fs):
    """
    plot spectrogram of 1st mic for all trials
    [batch_size, mic, freq, time]
    """

    number_of_trials = len(data)

    fig, axs = plt.subplots(number_of_trials, 1, figsize=(10, 4*number_of_trials))

    for i in range(number_of_trials):
        S = data[i][0] #just take the first mic
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, vmin=-80, vmax=0, 
                                       hop_length=cfg.hop_length, n_fft=cfg.n_fft, ax=axs[i])
        fig.colorbar(img, ax=axs[i], format='%+2.0f dB')
        axs[i].set_title(f'Spectrogram of trial {i+1}')

    plt.tight_layout()
    plt.show()

def plot_spectrogram_of_all_data_independent( data, fs):
    """
    plot spectrogram of only mic1 for all trials.
    This is independent of the training loop and for visualization purposes only 
    plot in 10x1 grid tight layout"
    """

    trial_number = len(data)
    print(f"trial_number: {trial_number}")

    #trim to the shortest length
    min_length = min(len(data[i]) for i in range(trial_number))
    for i in range(trial_number):
        data[i] = data[i][:min_length]

    print(f" ------ plotting spectrogram for all trials ------  ")
    # Plot the spectrogram for all 10 trials, mic1
    fig, axs = plt.subplots(trial_number, 1, figsize=(15, 10), tight_layout=True)
    fig.suptitle('Mel-Spectrogram for all trials, mic1')

    for i in range(trial_number):
        # convert to spectrogram
        img = get_spectrogram(data[i], fs)
        axs[i].set_title(f"trial{i}")
        fig.colorbar(img, ax=axs[i], format='%+2.0f dB')

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
    mic_max_amplitude_over_trials = []

    for mic in devicelist:
        #concat 1 mic over all trials
        mic_data_all_trials, mic_max_all_trials = concat_wav_files_dataset(load_path, trial_count, mic)
        mic_data_over_trials.append(mic_data_all_trials)
        mic_max_amplitude_over_trials.append(mic_max_all_trials)

    return mic_data_over_trials, mic_max_amplitude_over_trials

def load_wav_files_from_dataset_sections(devicelist, trial_start, trial_end, load_path):
    """
    return a list of concatenated wav files for all mics for all trials
    """

    mic_data_over_trials = []
    mic_max_amplitude_over_trials = []
    mic_max_index_over_trials = []

    for mic in devicelist:
        #concat 1 mic over all trials
        mic_data_all_trials, mic_max_all_trials, mic_max_index_data = concat_wav_files_dataset_sections(load_path, trial_start, trial_end, mic)
        mic_data_over_trials.append(mic_data_all_trials)
        mic_max_amplitude_over_trials.append(mic_max_all_trials)
        mic_max_index_over_trials.append(mic_max_index_data)

    return mic_data_over_trials, mic_max_amplitude_over_trials, mic_max_index_over_trials

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
            data, fs = librosa.load(file_name, sr=44100, mono=False)
            mic_data_single_trial.append(data)

        mic_data_over_trials.append(mic_data_single_trial)

    return mic_data_over_trials

def load_background_noise(cfg):
        """
        Load the data from a single trial to subtract from the dataset
        return a list of background noise for all mics, dim [num_mics]
        """
        
        num_mics = len(cfg.device_list)

        background_wavs = []
        path_name = cfg.background_dir

        for i in range(num_mics):
            wav_filename = f"{path_name}/mic{cfg.device_list[i]}.wav"
            wav, sample_rate = torchaudio.load(wav_filename)

            #extend wav length by wrapping around twice
            wav = torch.cat([wav, wav], dim=1)

            # background_wavs.append(wav.squeeze(0)) # remove the dimension of size 1
        
        return background_wavs

def load_background_noise_multiChannel(cfg):
        """
        Load the data from a single trial to subtract from the dataset (when using multichannel signal from UMC DAQ)
        return a list of background noise for all mics, dim [num_mics]
        """
        
        num_mics = cfg.num_channel

        background_wavs = []
        path_name = cfg.background_dir

        wav_filename = f"{path_name}/mic{cfg.device_list[0]}.wav"
        wav, sample_rate = torchaudio.load(wav_filename)
        
        #convert torch tensor into list tensor([6,88200]) --> [88200, ..., 88200] 
        for i in range(num_mics):
            background_wavs.append(wav[i][:])
        
        
        return background_wavs

def subtract_background_noise(y_single_waveform, noise_single_waveform, sample_rate):
    """
    in: array of single waveform of y and noise
    out: y without noise
    """

    #trim length of background noise to match the length of wav file
    background = noise_single_waveform[:y_single_waveform.size()[0]]

    # print(f"dimensions of all 3 wavs: {input_wav.size()}, {background.size()}, {wavs[i].size()}")
    # Apply noise reduction
    y_single_waveform = nr.reduce_noise(y=y_single_waveform, sr=sample_rate, y_noise=background, stationary=True, n_std_thresh_stationary = 2.5)
    #convert to tensor
    y_single_waveform = torch.tensor(y_single_waveform, dtype=torch.float32)
    return y_single_waveform


def load_specific_wav_files(devicelist, mics_to_plot, save_path_data, total_trial_count):
    """
    return a list of concatenated wav files for all trials ONLY the mics of interest
    """

    mic_data_over_trials = []

    for mic in mics_to_plot:
        # #concat 1 mic over all trials
        # mic_id = devicelist[mic]
        # mic_data_all_trials = concat_wav_files(save_path_data,  total_trial_count , mic_id)
        # mic_data_over_trials.append(mic_data_all_trials)

        file_name = f"{save_path_data}/mic{devicelist[mic]}.wav"
        print(f"file_name: {file_name}")
        data, fs = librosa.load(file_name, sr=44100, mono=True)
        mic_data_over_trials.append(data)
        
        

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
    given directory
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
    # librosa.display.waveshow(mics_plot_over_trials[0], alpha=0.5)
    # Plot the amplitude envelope over time
    plt.plot(time, mics_envelope_over_trials[0], color="r") 
    # Set the title of the plot
    plt.title("Waveform for (Amplitude Envelope)")  
    # Show the plot
    plt.show()  

def plot_envelope_from_signal(signal, frame_size, hop_length):
    """
    given signal, plot the amplitude envelope
    """
    envelope = amplitude_envelope(signal, frame_size, hop_length)

    # Generate the frame indices
    frames = range(0, len(envelope))  
    # Convert frames to time
    time = librosa.frames_to_time(frames, hop_length=hop_length)  
    # Create a new figure with a specific size
    plt.figure(figsize=(15, 7))  
    # Display the waveform of the signal
    librosa.display.waveshow(signal, alpha=0.5, color='b')
    # Plot the amplitude envelope over time
    plt.plot(time, envelope, color="r") 
    # Set the title of the plot
    plt.title("Waveform for (Amplitude Envelope)")  
    
    # Show the plot
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def concat_wav_files_dataset(load_path, trial_count,mic_id):
    """
    in: directory, trial number, mic number to concatenate
    output: concatenated wav file
    """

    concat_data = []

    #for all trials, find the wav file and concatenate
    for trial_number in range(trial_count):

        file_name = f"{load_path}trial{trial_number}/mic{mic_id}.wav"
        data, fs = librosa.load(file_name, sr=44100, mono=False)
        concat_data.append(data)
        
    # print(f"concat_data shape: {len(concat_data)}") #--> 50 trials
    # print(f"concat_data[0] shape: {concat_data[0].shape}") #--> (6, 88320)
    
    #find the minimum length of all trials and then trim all trials to the minimum length
    min_length = min(len(single_trial_data[1]) for single_trial_data in concat_data)
    # print(f"min_length: {min_length}")

    for i in range(trial_count):
        concat_data[i] = concat_data[i][:, :min_length]

    # print(f"concat_data[0] shape: {concat_data[0].shape}") #--> (6, min_length)
        

    #include the max amplitude of each mic in the dataset --> (N_trial, 6)
    mic_max_data = []
    for i in range(trial_count):
        max_for_all_mic = np.max(np.abs(concat_data[i]), axis=1)
        # print(f"shape of max_for_all_mic: {max_for_all_mic.shape}") #--> (6,
        mic_max_data.append(max_for_all_mic)

    # print(f"shape of mic_max_data: {len(mic_max_data)}") #--> 50
    #convert to numpy array (N_trial, 6)
    mic_max_data = np.array(mic_max_data)
    # print(f"shape of mic_max_data: {mic_max_data.shape}") #--> (50, 6

    #convert (N_trial, 6, min_length) to (6, N_trial*min_length)
    concat_data = np.concatenate(concat_data, axis=1)
    # print(f"concat_data shape: {concat_data.shape}") #--> (6, 50*min_length)
    
    return concat_data, mic_max_data


def concat_wav_files_dataset_sections(load_path, trial_start, trial_end,mic_id):
    """
    in: directory, trial number, mic number to concatenate
    output: concatenated wav file
    """

    concat_data = []

    #for all trials, find the wav file and concatenate
    for trial_number in range(trial_start, trial_end):

        file_name = f"{load_path}trial{trial_number}/mic{mic_id}.wav"
        data, fs = librosa.load(file_name, sr=44100, mono=False)
        concat_data.append(data)
        
    # print(f"concat_data shape: {len(concat_data)}") #--> 50 trials
    # print(f"concat_data[0] shape: {concat_data[0].shape}") #--> (6, 88320)
    
    #find the minimum length of all trials and then trim all trials to the minimum length
    min_length = min(len(single_trial_data[1]) for single_trial_data in concat_data)
    # print(f"min_length: {min_length}")

    trial_count = trial_end - trial_start
    for i in range(trial_count):
        concat_data[i] = concat_data[i][:, :min_length]

    # print(f"concat_data[0] shape: {concat_data[0].shape}") #--> (6, min_length)
        
    #clip concat_data to the first half to prevent bad signal from the second half
    for i in range(trial_count):
        concat_data[i] = concat_data[i][:, :min_length//2]

    #include the max amplitude of each mic in the dataset --> (N_trial, 6)
    mic_max_data = []
    mic_max_index_data = []
    for i in range(trial_count):
        max_for_all_mic = np.max(np.abs(concat_data[i]), axis=1)
        max_index_for_all_mic = np.argmax(np.abs(concat_data[i]), axis=1)
        # print(f"shape of max_for_all_mic: {max_for_all_mic.shape}") #--> (6,
        # print(f"max_index_for_all_mic {max_index_for_all_mic}")

        #plot time domain 
        # plot_time_domain(concat_data[i], 44100)
        mic_max_data.append(max_for_all_mic)
        mic_max_index_data.append(max_index_for_all_mic)

    # print(f"shape of mic_max_data: {len(mic_max_data)}") #--> 50
    #convert to numpy array (N_trial, 6)
    mic_max_data = np.array(mic_max_data)
    # print(f"shape of mic_max_data: {mic_max_data.shape}") #--> (50, 6

    #convert (N_trial, 6, min_length) to (6, N_trial*min_length)
    concat_data = np.concatenate(concat_data, axis=1)
    # print(f"concat_data shape: {concat_data.shape}") #--> (6, 50*min_length)
    
    return concat_data, mic_max_data, mic_max_index_data

def concat_wav_files(load_path, trial_count,mic_id):
    """
    in: directory, trial number, mic number to concatenate
    output: concatenated wav file
    """

    concat_data = []

    #for all trials, find the wav file and concatenate
    for trial_number in range(trial_count):

        file_name = f"{load_path}trial{trial_number}/mic{mic_id}.wav"
        print(f"file_name: {file_name}")
        data, fs = librosa.load(file_name, sr=44100, mono=True)
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
    # print(f"size of signal: {len(signal)}, size of output: {len(output)}")
    return output

def trim_audio_around_peak(data_list, fs, sample_window=44100):
    """
    Finds the peak of the audio signal and trims the signal around the peak.

    Args:
        data_list (list): The input list of audio signals.

    Returns:
        list: The list of trimmed audio signals.
    """

    # Create a list to store the trimmed audio signals
    trimmed_data = []

    # print(f"size of data_list[0] {len(data_list[0])}")
    
    peak_index = np.argmax(data_list[0])

    # print(f"peak_index: {peak_index}")

    start_index = peak_index - int(sample_window)//2
    start_index = max(0, start_index)
    # end_index = peak_index + int(sample_window)//2
    end_index = start_index + sample_window

    # Trim the audio signals wiht new start and end indices
    for data in data_list:
        trimmed_data.append(data[start_index:end_index])

        # print(f"start index {start_index}, end index {end_index}")

        if len(data[start_index:end_index])==0:
            #plot data
            plot_time_domain(data_list, fs)
            print(f"size is 0")
            sys.exit()
        

    # if trimmed data size is not equal to sample_window, then pad with zeros
    for i in range(len(trimmed_data)):
        if len(trimmed_data[i]) < sample_window:
            print(f"len of trimmed_data: {len(trimmed_data[i])}, sample_window: {sample_window}")
            print(f" ************ padding with zeros ************ ")
            # Calculate the number of zeros needed
            pad_size = sample_window - len(trimmed_data[i])
            # Convert the data to a tensor and pad it
            trimmed_data[i] = F.pad(torch.tensor(trimmed_data[i]), (0, pad_size))

    return trimmed_data


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

def trim_or_pad(wav, max_num_frames):
    """
    either pad or wav clips to be same length as desired max_num_frames
    return modified wav
    """
    # print(f"wav size: {wav.size()}, max_num_frames: {max_num_frames}") #wav size: torch.Size([6, 87968]), max_num_frames: 88200

    #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
    if wav.size(1) < max_num_frames:
        wav = F.pad(wav, (0, max_num_frames - wav.size(1)), mode='replicate')
        # wav = F.pad(wav, (0, max_num_frames - wav.size(1)), mode='circular'   )
    else:
        wav = wav[:, :max_num_frames]

    return wav

def trim_or_pad_single(wav, max_num_frames):
    """
    either pad or clip to be same length as desired max_num_frames
    [samples, channel]
    return modified wav
    """
    
    #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
    if wav.size(0) < max_num_frames:
        wav = F.pad(wav, (0, max_num_frames - wav.size(0)), mode='circular'   )
    else:
        wav = wav[:max_num_frames,:]

    
    return wav

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


def compute_gcc_phat_pair(x: torch.Tensor, y: torch.Tensor, fs: int = 44100, max_tau: float = 0.01):
    """
    Compute GCC-PHAT between a pair of signals
    
    Parameters:
    -----------
    x : torch.Tensor
        First signal [n_samples]
    y : torch.Tensor
        Second signal [n_samples]
    fs : int
        Sampling frequency (default: 44100)
    max_tau : float
        Maximum time lag in seconds (default: 10ms)
    
    Returns:
    --------
    gcc : torch.Tensor
        GCC-PHAT result [n_time_lags]
    tau : torch.Tensor
        Time lag axis in seconds [n_time_lags]
    """
    # Calculate number of lag points
    max_lag_samples = int(max_tau * fs)
    
    # Compute FFT
    X = torch.fft.rfft(x)
    Y = torch.fft.rfft(y)
    
    # Cross-spectrum
    Gxy = X * Y.conj()
    
    # PHAT weighting
    eps = 1e-10
    Gxy_phat = Gxy / (torch.abs(Gxy) + eps)
    
    # IFFT real value
    gcc = torch.fft.irfft(Gxy_phat)
    
    # Limit to max lag and shift
    gcc = torch.cat((gcc[-max_lag_samples:], gcc[:max_lag_samples+1]))
    
    # Create time lag axis
    tau = torch.linspace(-max_tau, max_tau, 2*max_lag_samples+1)
    
    return gcc, tau

def compute_gcc_phat_matrix(signals: torch.Tensor, fs: int = 44100, max_tau: float = 0.01):
    """
    Compute GCC-PHAT for all microphone pairs from multichannel wav file
    
    Parameters:
    -----------
    signals : torch.Tensor
        Microphone signals [n_mics=6, n_samples=66150]
    fs : int
        Sampling frequency (default: 44100)
    max_tau : float
        Maximum time lag in seconds (default: 10ms)
    
    Returns:
    --------
    gcc_matrix : torch.Tensor
        GCC-PHAT results [n_pairs=15, n_time_lags]
    tau : torch.Tensor
        Time lag axis in seconds [n_time_lags]
    pairs : list
        List of microphone pairs
    """
    # Input validation
    assert signals.dim() == 2, f"Expected 2D tensor, got {signals.dim()}D"
    assert signals.shape[0] == 6, f"Expected 6 channels, got {signals.shape[0]}"
    
    # Generate all microphone pairs
    pairs = list(combinations(range(6), 2))  # 15 pairs

    # print(f"pairs: {pairs}")
    
    # Compute first pair to get output size
    gcc, tau = compute_gcc_phat_pair(signals[0], signals[1], fs, max_tau)
    
    # print(f"dimensions of gcc: {gcc.shape}, dim of tau: {tau.shape}")

    # Initialize output matrix
    gcc_matrix = torch.zeros(len(pairs), len(gcc))
    gcc_matrix[0] = gcc
    
    # Compute GCC-PHAT for remaining pairs
    for i, (mic1, mic2) in enumerate(pairs[1:], 1):
        gcc_matrix[i], _ = compute_gcc_phat_pair(signals[mic1], signals[mic2], fs, max_tau)
    
    # print(f"dimensions of gcc_matrix: {gcc_matrix.shape}")

    return gcc_matrix, tau, pairs

def plot_gcc_phat_results(gcc_matrix: torch.Tensor, tau: torch.Tensor, pairs: list, 
                         figsize=(15, 20)):
    """
    Visualize GCC-PHAT results for all microphone pairs
    
    Parameters:
    -----------
    gcc_matrix : torch.Tensor
        GCC-PHAT results [15, n_time_lags]
    tau : torch.Tensor
        Time lag axis in seconds
    pairs : list
        List of microphone pairs
    figsize : tuple
        Figure size (width, height)
    """
    # Convert to milliseconds for better readability
    tau_ms = tau * 1000
    
    # Create subplots (3 columns x 5 rows)
    fig, axes = plt.subplots(5, 3, figsize=figsize)
    fig.suptitle('GCC-PHAT Results for All Microphone Pairs', fontsize=16)
    
    # Plot each pair
    for i, ((mic1, mic2), ax) in enumerate(zip(pairs, axes.flatten())):
        # Plot GCC-PHAT
        ax.plot(tau_ms, gcc_matrix[i])
        
        # Find and mark the peak
        peak_idx = torch.argmax(gcc_matrix[i])
        peak_time = tau_ms[peak_idx]
        peak_value = gcc_matrix[i][peak_idx]
        
        # Add peak marker
        ax.plot(peak_time, peak_value, 'r*', markersize=10, 
                label=f'Peak: {peak_time:.2f} ms')
        
        # Customize plot
        ax.set_title(f'Mic {mic1} - Mic {mic2}')
        ax.set_xlabel('Time Lag (ms)')
        ax.set_ylabel('Correlation')
        ax.grid(True)
        ax.legend()
        
        # Set x-axis limits to ±10ms
        ax.set_xlim([-10, 10])
        
        # Set y-axis limits with some padding
        max_val = torch.max(gcc_matrix[i])
        min_val = torch.min(gcc_matrix[i])
        padding = 0.1 * (max_val - min_val)
        ax.set_ylim([min_val - padding, max_val + padding])
    
    plt.tight_layout()
    plt.show()

    # Print peak time delays
    print("\nPeak Time Delays:")
    print("-----------------")
    for i, (mic1, mic2) in enumerate(pairs):
        peak_idx = torch.argmax(gcc_matrix[i])
        peak_time = tau_ms[peak_idx]
        print(f"Mic {mic1}-{mic2}: {peak_time:.2f} ms")

def get_time_differences(gcc_matrix: torch.Tensor, tau: torch.Tensor):
    """
    Extract Time Difference of Arrival (TDOA) from GCC-PHAT matrix
    
    Parameters:
    -----------
    gcc_matrix : torch.Tensor
        GCC-PHAT results [15, n_time_lags]
    tau : torch.Tensor
        Time lag axis in seconds

    Returns:
    --------
    tdoa : torch.Tensor
        Time differences [15] in seconds for each mic pair
    tdoa_idx : torch.Tensor
        Indices [15] of peak locations in gcc_matrix

    """
    # Find peaks for all pairs
    tdoa_idx = torch.argmax(gcc_matrix, dim=1)
    
    # Get corresponding time delays
    tdoa = tau[tdoa_idx]
    
    # convert to milliseconds
    tdoa = tdoa * 1000
    
    return tdoa, tdoa_idx
