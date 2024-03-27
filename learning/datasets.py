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
import noisereduce as nr

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

    def __init__(self, cfg = None, data_dir = None, transform = None):
        print(f" --------- initializing DS ---------")
        self.cfg = cfg

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
        self.load_background_noise()

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

    def load_background_noise(self):
        """Load the data from a single trial to subtract from the data"""
        print(f" --------- loading background noise ---------")
        num_mics = len(self.cfg.device_list)

        background_wavs = []
        path_name = self.cfg.background_dir

        for i in range(num_mics):
            wav_filename = f"{path_name}/mic{self.cfg.device_list[i]}.wav"
            wav, sample_rate = torchaudio.load(wav_filename)

            #extend wav length by wrapping around twice
            wav = torch.cat([wav, wav], dim=1)

            background_wavs.append(wav.squeeze(0)) # remove the dimension of size 1
        
        self.background_wavs = background_wavs

        # self.plot_fft(self.background_wavs, sample_rate)


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
                #trim length of background noise to match the length of wav file
                background = self.background_wavs[i][:wav.size(1)]

                # print(f"dimensions of all 3 wavs: {input_wav.size()}, {background.size()}, {wavs[i].size()}")
                # Apply noise reduction
                wavs[i] = nr.reduce_noise(y=wavs[i], sr=self.sample_rate, y_noise=background, stationary=True, n_std_thresh_stationary = 4.5)
                #convert to tensor
                wavs[i] = torch.tensor(wavs[i], dtype=torch.float32)
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
        # label[0] = label[0] / 20.32 #(8" to 20cm)
        # label[1] = (label[1] + np.pi) / (2*np.pi) #[-pi,pi] to [0,1]

        # #convert radian to x,y coordinate with unit 1
        x,y  = np.cos(label[1]), np.sin(label[1])
        # label = [label[0], x, y] # converted radian for continuous 0 to 2pi representation in xy coordinate
        # label = [label[0], label[1]] #TODO: REMOVE converted radian for continuous 0 to 2pi representation in xy coordinate


        # #convert to tensor
        # label = torch.tensor(label, dtype=torch.float32)

        return data, label
    
    def __len__(self):
        return len(self.dir)
    
    def __getitem__(self, idx):
        x,y = self.X_mic_data[idx], self.Y_label_data[idx] # --> x: full 2 sec wav/spectrogram

        #default starting bin is center of spectrogram
        starting_bin = 175

        if self.cfg.augment_timeshift:
            #crop spectrogram to 1 sec (bin size of 345) from a random starting point
            starting_bin = random.randint(150,220) #--> 0 to 345
        x = x[:, :, starting_bin:starting_bin + 345] #--> [num_mics, num_mels, 345] 


        #apply augmentation
        if self.cfg.augment_timemask:
            time_masking = T.TimeMasking(time_mask_param=40)
            x = time_masking(x)

        if self.cfg.augment_freqmask:

            freq_masking = T.FrequencyMasking(freq_mask_param=5)
            x = freq_masking(x)

        

        return x,y 






    
    
def load_data(cfg, train_or_val = 'val'):
    """
    Load the dataset and split it into train and validation
    """
    
    dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform)

  

    #visuaize dataset
    if cfg.visuaize_dataset:
        # print(f"size of dataset: {len(dataset)}")
        # x_list = []
        # y_list = []
        # height_list = []

        # for i in range(len(dataset)):
        #     #plot all labels
        #     x, y = dataset[i]

            
        #     x_list.append(y[1].item())
        #     y_list.append(y[2].item())
        #     height_list.append(y[0].item())

        # #plot 3D scatter plot of rad_x,rad_y,height
        # #point should be color coded based on index number
        # # Create a new figure
        # fig = plt.figure()

        # # Add a 3D subplot
        # ax = fig.add_subplot(111, projection='3d')

        # # Create a color map based on index number
        # colors = plt.cm.viridis(np.linspace(0, 1, len(dataset)))

        # # Plot 3D scatter plot of rad_x, rad_y, height
        # scatter = ax.scatter(x_list, y_list, height_list, c=colors)

        # # Add a color bar
        # plt.colorbar(scatter)

        # # Set labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Height')

        # # Show the plot
        # plt.show()


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

