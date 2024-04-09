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
import microphone_utils as mic_utils


"""
Define AudioDataset class to load and process audio data
referenced from: https://github.com/abitha-thankaraj/audio-robot-learning
"""

torch.manual_seed(42)
np.random.seed(42)

TRANSFORMS = {
    'wav' : get_signal,
    'mel': to_mel_spectrogram
}


class AudioDataset(Dataset):
    """
    AudioDataset class to load and process audio data    
    """

    def __init__(self, cfg = None, data_dir = None, transform = None, augment = False):
        print(f" --------- initializing DS ---------")
        self.cfg = cfg
        self.augment = augment

        if transform is not None: # Only for audio datasets 
            # self.resampler = T.Resample(orig_freq = self.cfg.sample_rate, new_freq = self.cfg.resample_rate)
            self.transform = TRANSFORMS[transform]

        #get all directory path to trials
        self.dir = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])

        #filter out directory that does not start with 'trial'
        self.dir = [d for d in self.dir if d.split('/')[-1].startswith('trial')]
        
        #load data (for multiple mics in device list, get wav files)
        len_data = len(self.dir)
        
        self.X_mic_data = []
        self.Y_label_data = []

        #load background noise
        print(f" --------- loading background noise ---------")
        self.background_wavs = mic_utils.load_background_noise(cfg)
        # mic_utils.plot_fft(self.background_wavs, self.sample_rate)

        print(f" --------- loading data ---------")
        for trial_n in range(len_data):
            print(f"loading trial: {trial_n}")
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

                #confirm if data is normalized
                # print(f"dataset mean: {self.X_mic_data.mean()}, dataset std: {self.X_mic_data.std()}")
                # mean = self.X_mic_data.mean(dim=[0, 2, 3], keepdim=True)  # Mean across trials, num_mels, and num_bins_time
                # var = self.X_mic_data.var(dim=[0, 2, 3], keepdim=True)    # Variance across trials, num_mels, and num_bins_time
                # print(f"dataset mean: {mean}, dataset std: {torch.sqrt(var)}")
                # sys.exit()

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

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            if wav.size(1) < self.cfg.max_num_frames:
                wav = F.pad(wav, (0, self.cfg.max_num_frames - wav.size(1)), mode='circular'   )
            else:
                wav = wav[:, :self.cfg.max_num_frames]


            #append to list of wavs
            wavs.append(wav.squeeze(0)) # remove the dimension of size 1

            if self.cfg.subtract_background:
                wavs[i] = mic_utils.subtract_background_noise(wavs[i], self.background_wavs[i], self.sample_rate)
                # print(f"dim of wav after noise reduction: {wavs[i].size()}")

            
            #apply transform to wav file
            if self.transform:
                mel = self.transform(self.cfg, wavs[i].float())
                melspecs.append(mel.squeeze(0)) # remove the dimension of size 1

        # mic_utils.plot_fft(wavs, sample_rate, self.cfg.device_list)
        # sys.exit()
                
        # stack wav files into a tensor of shape (num_mics, num_samples)
        wav_tensor = torch.stack(wavs, dim=0)
        # print(f"dimension of wav tensor: {wav.size()}") #--> dimension of wav tensor: torch.Size([6, 88200])

        #stack mel spectrograms into a tensor of shape (num_mics, num_mels, num_samples)
        mel_tensor = torch.stack(melspecs, dim=0)
        # print(f"size of mel_tensor: {mel_tensor.size()}") #--> size of data: torch.Size([6, 16, 690])

        if self.cfg.transform == 'mel':
            data = mel_tensor

        if self.cfg.transform == 'wav':
            data = wav_tensor
            
        #get label from directory 
        label_file = f"{self.dir[trial_n]}/gt_label.npy"
        label = np.load(label_file) #--> [distance along cylinder, joint 6] i.e. [0.0 m, -2.7 radian]

        #convert label m unit to cm
        label[0] = label[0] * 100

        #normalize height to [0,1], and radian to [0,1]
        if cfg.output_representation_normalize:
            label[0] = label[0] / 20.32 #(8" to 20cm)

        if cfg.output_representation == 'radian':
            label[1] = label[1]

            if cfg.output_representation_normalize:
                label[1] = (label[1] + np.pi) / (2*np.pi) #[-pi,pi] to [0,1]

        elif cfg.output_representation == 'xy':
            #convert radian to x,y coordinate with unit 1
            x,y  = np.cos(label[1]), np.sin(label[1])
            # label = [label[0], x, y] #dont make into list, but stack as numpy label[2]
            label[1] = x
            label = np.append(label, y) #--> [height, x, y]

        elif cfg.output_representation == 'height':
            label = label[0]

        elif cfg.output_representation == 'height_radianclass':
            radian = label[1]
            #convert radian  ranging from -2.7 to 2.7 into K classes
            #-2.7 : 0, ..., 2.7 : K-1
            #-2.7:0, -2.025:1, -1.35:2, -0.675:3, 0:4, 0.675:5, 1.35:6, 2.025:7, 2.7:8
            
            original_min=-2.7
            original_max=2.7
            num_classes=8

            # Normalize
            normalized = (radian - original_min) / (original_max - original_min)
            # Scale to number of classes and convert to integer
            class_label = int(normalized * num_classes)
            # Ensure the class label is within the range [0, num_classes-1]
            class_label = min(class_label, num_classes - 1)

            label[1] = class_label




        # #convert to tensor
        # label = torch.tensor(label, dtype=torch.float32)

        return data, label
    
    def __len__(self):
        return len(self.dir)
    
    def __getitem__(self, idx):
        x,y = self.X_mic_data[idx], self.Y_label_data[idx] # --> x: full 2 sec wav/spectrogram

        #default starting bin is center of spectrogram
        starting_bin = 175


        if self.augment:

            if self.cfg.augment_timeshift:
                #crop spectrogram to 1 sec (bin size of 345) from a random starting point
                starting_bin = random.randint(150,220) #--> 0 to 345
            x = x[:, :, starting_bin:starting_bin + 345] #--> [num_mics, num_mels, 345] 

            if self.cfg.affine_transform:
                #apply affine transformation
                affine_transform = RandomAffine(degrees=0, translate=(0.1, 0.))
                x = affine_transform(x)


            #apply augmentation
            if self.cfg.augment_timemask:
                time_masking = T.TimeMasking(time_mask_param=40)
                x = time_masking(x)

            if self.cfg.augment_freqmask:

                freq_masking = T.FrequencyMasking(freq_mask_param=5)
                x = freq_masking(x)

        else: #no augmentation
            #center crop the spectrogram around starting bin (bin size of 345). 
            # Should be 175/2 on the left and 175/2 on the right
            start = (x.shape[2] - 345) // 2
            end = start + 345
            x = x[:, :, start:end]

        

        return x,y 



def load_data(cfg, train_or_val = 'val'):
    """
    Load the dataset and split it into train and validation
    """
    
    dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augment = True)

  
    #visuaize dataset
    if cfg.visuaize_dataset:
        for i in range(5):
            
            #get first element of dataset
            x, y = dataset[i]
            #convert to numpy
            x = x.numpy()
            #convert to list of 1st channel --> [[40, 345] ... [40, 345] ]
            x = [x[i] for i in range(x.shape[0])]
            # print(f"size of x: {len(x)}")
            print(f"i: {i} and y: {y}")
            # plot mel spectrogram
            mic_utils.plot_spectrogram_with_cfg(cfg, x, dataset.sample_rate)


    # split the dataset into train and validation 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])



    #load train and val loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False)

    return train_loader, val_loader

