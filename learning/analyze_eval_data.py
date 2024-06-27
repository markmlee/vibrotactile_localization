import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

from datasets import AudioDataset 
import torch
import logging
import os
import math
from tqdm import tqdm
import sys
from easydict import EasyDict

from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np

import microphone_utils as mic_utils

#logger
import logging
log = logging.getLogger(__name__)
from logger import Logger

#dataset
from datasets import AudioDataset

#models
# from models.KNN import KNN
from models.CNN import CNNRegressor, CNNRegressor2D

#eval
from sklearn.metrics import mean_squared_error
import eval_utils as eval_utils

torch.manual_seed(42)
import matplotlib.pyplot as plt

import torchaudio
from scipy.signal import butter, lfilter

def load_data(cfg, data_directory):
    #load data
    dataset = AudioDataset(cfg=cfg, data_dir = data_directory, transform = cfg.transform, augment = False)

    #visuaize dataset
    if cfg.visuaize_dataset:
        for i in range(1):
            
            #get first element of dataset
            x, y, wav = dataset[i]
            #convert to numpy
            x = x.numpy()
            #convert to list of 1st channel --> [[40, 345] ... [40, 345] ]
            x = [x[i] for i in range(x.shape[0])]
            # print(f"size of x: {len(x)}")
            # print(f"i: {i} and y: {y}")

            # print(f"wav shape: {wav.shape}")

            # plot mel spectrogram
            # mic_utils.plot_spectrogram_with_cfg(cfg, x, dataset.sample_rate)

            #plot wav
            # mic_utils.grid_plot_time_domain(wav, dataset.sample_rate)

    # sys.exit()

    # split the dataset into train and validation 80/20
    train_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    #USE SEQUENTIAL SPLIT
    # Created using indices from 0 to train_size.
    train_size = 0
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    # Created using indices from train_size to the end.
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))



    #load train and val loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False)

    return train_loader, val_loader



def calculate_radian_error(rad_pred, rad_val):
    """
    calculate the degree error between the predicted and ground truth radian values
    This resolves wrap around issues
    """
    #diff = pred - GT
    #add +pi
    #mod by 2pi
    #subtract pi

    # print(f"0. rad_pred: {rad_pred}, rad_val: {rad_val}")
    rad_diff = rad_pred - rad_val
    # print(f"1. rad_diff: {rad_diff}")
    rad_diff = rad_diff + math.pi
    # print(f"2. rad_diff: {rad_diff}")
    rad_diff = torch.remainder(rad_diff, 2*math.pi)
    # print(f"3. rad_diff: {rad_diff}")
    radian_error = rad_diff - math.pi
    # print(f"4. radian_error: {radian_error}")

    return radian_error

def analyze_freq_across_stick_location(cfg):
    """
    Analyze the frequency across different stick locations
    """
    #directory of data to load
    data_dir0 = '/home/mark/audio_learning_project/data/test_generalization/stick_T32L42_Y_25/'
    data_dir1 = '/home/mark/audio_learning_project/data/test_generalization/stick_T32L42_Y_35/'
    data_dir2 = '/home/mark/audio_learning_project/data/test_generalization/stick_T32L42_Y_40/'

    data_dir_list = [data_dir0, data_dir1, data_dir2]

    for data_dir in data_dir_list:

        #load data
        train_loader, val_loader = load_data(cfg, data_dir)

        wav_first_sample = None
        spectrogram_first_sample = None

        for _, (x, y, wav) in enumerate(tqdm(val_loader)):

            #load the wav file
            denoised_wav = wav
            print(f"denoised_wav shape: {denoised_wav.shape}") #--> [batch, num mic, 66150]

            #get 1st sample
            wav_first_sample = denoised_wav[12]
            spectrogram_first_sample = x[12]
            continue

        

        #plot spectorgram of the first sample
        # mic_utils.plot_spectrogram_with_cfg(cfg, spectrogram_first_sample, cfg.sample_rate)

        #plot FFT of the first sample
        # mic_utils.plot_fft(wav_first_sample, cfg.sample_rate, [2])

        #plot average FFT of the first sample
        mic_utils.plot_single_fft(wav_first_sample, cfg.sample_rate, [2])

def analyze_freq_across_stick_thickness(cfg):
    """
    Analyze the frequency across different stick locations
    """
    #directory of data to load
    data_dir0 = '/home/mark/audio_learning_project/data/test_generalization/stick_T12L42_Y_35/'
    data_dir1 = '/home/mark/audio_learning_project/data/test_generalization/stick_T22L42_Y_35/'
    data_dir2 = '/home/mark/audio_learning_project/data/test_generalization/stick_T32L42_Y_35/'
    data_dir3 = '/home/mark/audio_learning_project/data/test_generalization/stick_T50L42_Y_35/'

    data_dir_list = [data_dir0, data_dir1, data_dir2, data_dir3]

    for data_dir in data_dir_list:

        #load data
        train_loader, val_loader = load_data(cfg, data_dir)

        wav_first_sample = None
        spectrogram_first_sample = None

        for _, (x, y, wav) in enumerate(tqdm(val_loader)):

            #load the wav file
            denoised_wav = wav
            print(f"denoised_wav shape: {denoised_wav.shape}") #--> [batch, num mic, 66150]

            #get 1st sample
            wav_first_sample = denoised_wav[3]
            spectrogram_first_sample = x[3]
            continue

        

        #plot spectorgram of the first sample
        # mic_utils.plot_spectrogram_with_cfg(cfg, spectrogram_first_sample, cfg.sample_rate)

        #plot FFT of the first sample
        mic_utils.plot_fft(wav_first_sample, cfg.sample_rate, [2])

        #plot average FFT of the first sample
        mic_utils.plot_single_fft(wav_first_sample, cfg.sample_rate, [2])
    

def butter_lowpass(cutoff, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, sample_rate, cutoff=2000, order=5):
    b, a = butter_lowpass(cutoff, sample_rate, order=order)
    filtered_data = lfilter(b, a, data.numpy(), axis=-1)
    return torch.tensor(filtered_data)

@hydra.main(version_base='1.3',config_path='configs', config_name = 'analyze_eval')
def main(cfg: DictConfig):
    
    # ---------- visualize FFT plot across different stick locations ----------
    analyze_freq_across_stick_location(cfg)
    sys.exit()
    # ----------------------------------------------------------------------

    # ---------- visualize FFT plot across different stick thickness ----------
    analyze_freq_across_stick_thickness(cfg)
    sys.exit()
    # ----------------------------------------------------------------------


    # # ---------- visualize a single wav file for sanity check ----------
    # #load wav file
    # wav_filename = '/home/mark/audio_learning_project/data/test_generalization/stick_T50_L42_Y_35/trial0/mic2.wav'
    # wav, sample_rate = torchaudio.load(wav_filename)
    # #plot time domain
    # mic_utils.grid_plot_time_domain(wav, sample_rate)

    # #plot FFT
    # mic_utils.plot_fft(wav, sample_rate, [2])

    # # Apply low-pass  filter freq above 800Hz
    # filtered_waveform = apply_lowpass_filter(wav, sample_rate, cutoff=800, order=5)

    # mic_utils.grid_plot_time_domain(filtered_waveform, sample_rate)
    # mic_utils.plot_fft(filtered_waveform, sample_rate, [2])

    # #save the filtered wav file
    # torchaudio.save('/home/mark/audio_learning_project/data/test_generalization/stick_T50_L42_Y_35/trial0/mic2_filtered.wav', filtered_waveform, sample_rate)

    # # sys.exit()
    # # ----------------------------------------------------------------------


    







   

if __name__ == '__main__':
    main()