import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop, RandomAffine, Compose, Pad

import random
import os
from collections import defaultdict
import torch.nn.functional as F
import torchaudio.transforms as T

from transforms import to_mel_spectrogram, get_signal

import sys
import numpy as np

import matplotlib.pyplot as plt
import librosa.display
import librosa

#import function from another directory for plotting
sys.path.insert(0,'/home/mark/audio_learning_project/vibrotactile_localization/scripts')
import microphone_utils


"""
Define AudioDataset class to load and process audio data
referenced from: https://github.com/abitha-thankaraj/audio-robot-learning
"""

torch.manual_seed(42)

TRANSFORMS = {
    'wav' : get_signal,
    'mel': to_mel_spectrogram
}

AUGMENTATIONS = {
    'time_shift': Compose([
        RandomAffine(degrees = (0,0), translate=(0.1, 0))
    ])

}

class AudioDataset(Dataset):
    """
    AudioDataset class to load and process audio data    
    """

    def __init__(self, cfg = None, data_dir = None, transform = None, augmentation = None):
        print(f" --------- initializing DS ---------")
        self.cfg = cfg
        self.augmentation = AUGMENTATIONS[augmentation]

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

            if cfg.augment_audio:
                x_data, y_label = self.load_xy_single_trial_with_augmentation(self.cfg, trial_n)
            else:
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


    def load_xy_single_trial_with_augmentation(self, cfg, trial_n):
        """
        Load the data from a single trial
        augmentation1: time shift
        """

        num_mics = len(self.cfg.device_list)
        trim_lengths = self.cfg.trim_duration #0.5s to trim from start and end of wav file. overwrite when augmenting

        if cfg.augment_audio:
            num_mics = cfg.augment_num_channel

            if cfg.augment_timeshift:
                trim_lengths = random.uniform(0.1, self.cfg.trim_duration) #randomly time shift the audio, keep same across mics 


        wavs = []
        melspecs = []


        for i in range(num_mics):
            wav_filename = f"{self.dir[trial_n]}/mic{self.cfg.device_list[i]}.wav"

            if cfg.augment_audio and cfg.augment_device_list_used is not None:
                wav_filename = f"{self.dir[trial_n]}/mic{self.cfg.augment_device_list_used[i]}.wav"

            wav, sample_rate = torchaudio.load(wav_filename)
            self.sample_rate = sample_rate
            # print(f"sample rate: {sample_rate}") #--> sample rate: 44100

            #trim 0.5s from start and end of wav file
            wav = wav[:, int(trim_lengths * sample_rate):int(-trim_lengths * sample_rate)]
            

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            if wav.size(1) < self.cfg.max_num_frames:
                wav = F.pad(wav, (0, self.cfg.max_num_frames - wav.size(1)), mode='circular'   )
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

        #convert label m unit to cm
        label[0] = label[0] * 100


        return data, label
    
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
            self.sample_rate = sample_rate
            # print(f"sample rate: {sample_rate}") #--> sample rate: 44100

            #trim 0.5s from start and end of wav file
            wav = wav[:, int(self.cfg.trim_duration * sample_rate):int(-self.cfg.trim_duration * sample_rate)]
            

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            if wav.size(1) < self.cfg.max_num_frames:
                wav = F.pad(wav, (0, self.cfg.max_num_frames - wav.size(1)), mode='circular'   )
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

        #convert label m unit to cm
        label[0] = label[0] * 100

        return data, label
    
    def __len__(self):
        return len(self.dir)
    
    def __getitem__(self, idx):
        x,y = self.X_mic_data[idx], self.Y_label_data[idx]

        #apply augmentation
        if self.cfg.apply_augmentation:
            x = self.augmentation(x)
            # plot_spectrogram(self.cfg, x, self.sample_rate)

        if self.cfg.augment_timemask:
            time_masking = T.TimeMasking(time_mask_param=40)
            x = time_masking(x)

        if self.cfg.augment_freqmask:

            freq_masking = T.FrequencyMasking(freq_mask_param=5)
            x = freq_masking(x)

        

        return x,y 



class AudioDataset_validation(Dataset):
    """
    AudioDataset class without any augmenation for validation (as similar to real world)  
    """

    def __init__(self, cfg = None, data_dir = None, transform = None, augmentation = None):
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
            self.sample_rate = sample_rate
            # print(f"sample rate: {sample_rate}") #--> sample rate: 44100

            #trim 0.5s from start and end of wav file
            wav = wav[:, int(self.cfg.trim_duration * sample_rate):int(-self.cfg.trim_duration * sample_rate)]
            

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            if wav.size(1) < self.cfg.max_num_frames:
                wav = F.pad(wav, (0, self.cfg.max_num_frames - wav.size(1)), mode='circular'   )
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

        #convert label m unit to cm
        label[0] = label[0] * 100

        return data, label
    
    def __len__(self):
        return len(self.dir)
    
    def __getitem__(self, idx):
        x,y = self.X_mic_data[idx], self.Y_label_data[idx]
        return x,y 

class AudioDataset_test(Dataset):
    """
    AudioDataset class to manually augment the data for testing. Verify robustness against augmentation
    """

    def __init__(self, cfg = None, data_dir = None, transform = None, augmentation = None):
        print(f" --------- initializing DS ---------")
        self.cfg = cfg
        self.augmentation = AUGMENTATIONS[augmentation]

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

            # if cfg.augment_test:
            #     x_data, y_label = self.load_xy_single_trial_with_augmentation(self.cfg, trial_n)

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

    def load_xy_single_trial_with_augmentation(self, cfg, trial_n):
        """
        Load the data from a single trial
        augmentation1: time shift
        """
        

        num_mics = len(self.cfg.device_list)
        trim_lengths = self.cfg.trim_duration #0.5s to trim from start and end of wav file. overwrite when augmenting

        if cfg.augment_audio:
            num_mics = cfg.augment_num_channel

            if cfg.augment_timeshift:
                trim_lengths = random.uniform(0.2, self.cfg.trim_duration) #randomly time shift the audio, keep same across mics 


        wavs = []
        melspecs = []


        for i in range(num_mics):
            wav_filename = f"{self.dir[trial_n]}/mic{self.cfg.device_list[i]}.wav"

            if cfg.augment_audio and cfg.augment_device_list_used is not None:
                wav_filename = f"{self.dir[trial_n]}/mic{self.cfg.augment_device_list_used[i]}.wav"

            wav, sample_rate = torchaudio.load(wav_filename)
            self.sample_rate = sample_rate
            # print(f"sample rate: {sample_rate}") #--> sample rate: 44100

            #trim 0.5s from start and end of wav file
            wav = wav[:, int(trim_lengths * sample_rate):int(-trim_lengths * sample_rate)]
            

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            if wav.size(1) < self.cfg.max_num_frames:
                wav = F.pad(wav, (0, self.cfg.max_num_frames - wav.size(1)), mode='circular'   )
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

        #convert label m unit to cm
        label[0] = label[0] * 100


        return data, label
    
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
            self.sample_rate = sample_rate
            # print(f"sample rate: {sample_rate}") #--> sample rate: 44100

            #trim 0.5s from start and end of wav file
            wav = wav[:, int(self.cfg.trim_duration * sample_rate):int(-self.cfg.trim_duration * sample_rate)]
            

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            if wav.size(1) < self.cfg.max_num_frames:
                wav = F.pad(wav, (0, self.cfg.max_num_frames - wav.size(1)), mode='circular'   )
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

        #convert label m unit to cm
        label[0] = label[0] * 100

        return data, label
    
    def __len__(self):
        return len(self.dir)
    
    def __getitem__(self, idx):
        x,y = self.X_mic_data[idx], self.Y_label_data[idx]
        if self.cfg.apply_augmentation:
            x = self.augmentation(x)

        if self.cfg.augment_timemask:
            time_masking = T.TimeMasking(time_mask_param=40)
            x = time_masking(x)

        if self.cfg.augment_freqmask:

            freq_masking = T.FrequencyMasking(freq_mask_param=5)
            x = freq_masking(x)

 


        return x,y 


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

def plot_spectrogram(cfg, data, fs):
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
        
    plt.show()


    



    
def load_data(cfg, train_or_val = 'val'):
    """
    Load the dataset and split it into train and validation
    """

    if train_or_val == 'train':
        dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augmentation = cfg.augmentation_type)

    elif train_or_val == 'val':
        dataset = AudioDataset_validation(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augmentation = None)
    
    else:
        print("Error: train_or_val should be either 'train' or 'val'")
        sys.exit()

    #visuaize dataset
    if cfg.visuaize_dataset:
        for i in range(15):
            print(f"size of dataset: {len(dataset)}")

            #get first element of dataset
            x, y = dataset[i]
            print(f"size of x, y: {x.size()}, {len(y)}") #--> torch.Size([6, 40, 345]), [0,0]

            #convert to numpy
            x = x.numpy()

            #convert to list of 1st channel --> [[40, 345] ... [40, 345] ]
            x = [x[i] for i in range(x.shape[0])]
            print(f"size of x: {len(x)}")

            # plot mel spectrogram
            print(f"sample rate: {dataset.sample_rate}")
            plot_spectrogram(cfg, x, dataset.sample_rate)
        sys.exit()


    # split the dataset into train and validation 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])



    #load train and val loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False)

    return train_loader, val_loader

