import torch
import torchaudio
from torch.utils.data import Dataset

import random
import os
from collections import defaultdict
import torch.nn.functional as F
import torchaudio.transforms as T

from transforms import to_mel_spectrogram, get_signal

import sys
import numpy as np
"""
Define AudioDataset class to load and process audio data
referenced from: https://github.com/abitha-thankaraj/audio-robot-learning
"""

torch.manual_seed(42)

TRANSFORMS = {
    'wav' : get_signal,
    'mel': to_mel_spectrogram
}


class AudioDataset(Dataset):
    """
    AudioDataset class to load and process audio data    
    """

    def __init__(self, cfg = None, data_dir = None, transform = None):
        print(f" --------- initializing DS ---------")
        self.cfg = cfg

        if transform is not None: # Only for audio datasets 
            # self.resampler = T.Resample(orig_freq = self.cfg.sample_rate, new_freq = self.cfg.resample_rate)
            self.transform = TRANSFORMS[transform]

        #get all directory path to trials
        self.dir = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
        
        #load data (for multiple mics in device list, get wav files)
        len_data = len(self.dir)
        
        self.X_mic_data = []
        self.Y_label_data = []

        for trial_n in range(len_data):
            x_data, y_label = self.load_xy_single_trial(self.cfg, trial_n)
            self.X_mic_data.append(x_data)
            self.Y_label_data.append(y_label)

        self.X_mic_data = torch.stack(self.X_mic_data)  # Shape: [len_data, num_mics, num_mels, num_bins_time]

        if cfg.normalize_audio_data: 
            
            if cfg.transform == 'mel': #torch.Size([6, 16, 442]) - [num_mics, num_mels, num_bins_time]

                # Calculate mean and variance for each microphone across all trials
                mean = self.X_mic_data.mean(dim=[0, 2, 3], keepdim=True)  # Mean across trials, num_mels, and num_bins_time
                var = self.X_mic_data.var(dim=[0, 2, 3], keepdim=True)    # Variance across trials, num_mels, and num_bins_time

                # Normalize the X_mic_data
                for i in range(len_data):
                    self.X_mic_data[i] = (self.X_mic_data[i] - mean) / torch.sqrt(var)

            print(f" --------- DS normalized ---------")


        
    def load_xy_single_trial(self, cfg, trial_n):
        """
        Load the data from a single trial
        """
        num_mics = len(self.cfg.device_list)
        wavs = []
        melspecs = []
        for i in range(num_mics):
            wav_filename = f"{self.dir[trial_n]}/mic{self.cfg.device_list[i]}.wav"
            wav, sample_rate = torchaudio.load(wav_filename)

            #trim 0.5s from start and end of wav file
            wav = wav[:, int(0.5 * sample_rate):int(-0.5 * sample_rate)]
            

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            if wav.size(1) < self.cfg.max_num_frames:
                wav = F.pad(wav, (0, self.cfg.max_num_frames - wav.size(1)))
            else:
                wav = wav[:, :self.cfg.max_num_frames]

            #append to list of wavs
            wavs.append(wav.squeeze(0)) # remove the dimension of size 1

            #apply transform to wav file
            if self.transform:
                mel = self.transform(self.cfg, wav.float())
                melspecs.append(mel.squeeze(0)) # remove the dimension of size 1

        # stack wav files into a tensor of shape (num_mics, num_samples)
        wav_tensor = torch.stack(wavs, dim=0)
        # print(f"dimension of wav tensor: {wav.size()}") #--> dimension of wav tensor: torch.Size([6, 88200])

        #stack mel spectrograms into a tensor of shape (num_mics, num_mels, num_samples)
        mel_tensor = torch.stack(melspecs, dim=0)
        # print(f"size of mel_tensor: {mel_tensor.size()}") #--> size of data: torch.Size([6, 16, 442])

        if cfg.transform == 'mel':
            data = mel_tensor

        if cfg.transform == 'wav':
            data = wav_tensor
            
        #get label from directory 
        label_file = f"{self.dir[trial_n]}/gt_label.npy"
        label = np.load(label_file) #--> [distance along cylinder, joint 6]

        return data, label
    
    def __len__(self):
        return len(self.dir)
    
    def __getitem__(self, idx):
        return self.X_mic_data[idx], self.Y_label_data[idx]

        



    
def load_data(cfg):
    """
    Load the dataset and split it into train and validation
    """

    dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform)

    # split the dataset into train and validation 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])



    #load train and val loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False)

    return train_loader, val_loader

