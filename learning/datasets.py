import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop, RandomAffine, Compose, Pad

import random
import os
from collections import defaultdict
import torch.nn.functional as F
import torchaudio.transforms as T

from transforms import to_mel_spectrogram, get_signal, to_magnitude_phase

import sys
import numpy as np
from itertools import combinations


import matplotlib.pyplot as plt
import librosa.display
import librosa

#import function from another directory for plotting
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
import microphone_utils as mic_utils


"""
Define AudioDataset class to load and process audio data
"""

torch.manual_seed(42)
np.random.seed(42)

TRANSFORMS = {
    'wav' : get_signal,
    'mel': to_mel_spectrogram,
    'stft': to_magnitude_phase
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
        dir_raw = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])

        #filter out directory that does not start with 'trial'
        dir_raw = [d for d in dir_raw if d.split('/')[-1].startswith('trial')]
        len_data = len(dir_raw)

        count = 0
        data_dir = f"{data_dir}trial"
        self.dir = []

        print(f"data_dir: {data_dir}, len(self.dir): {len(self.dir)}, len_data: {len_data}")
        while len(self.dir) < len_data:
            file_name = f"{data_dir}{count}"

            if file_name in dir_raw:
                self.dir.append(file_name)
            count += 1
        
        #load data (for multiple mics in device list, get wav files)
        len_data = len(self.dir)
        


        self.num_mic = cfg.num_channel

        self.X_mic_data = []
        self.Y_label_data = []
        self.wav_data_list = []
        self.qt_data_list = []
        self.xt_data_list = []
        self.xdot_t_data_list = []
        self.tdoa_data_list = []
        self.gcc_data_list = []

        #load background noise
        print(f" --------- loading background noise ---------")
        self.background_wavs = mic_utils.load_background_noise_multiChannel(cfg)
        

        print(f" --------- loading data ---------")
        for trial_n in range(len_data):
            print(f"loading trial: {trial_n}")
            x_data, y_label, wav_data, qt_data, xt_data, xdot_t_data,  tdoa_data, gcc_data = self.load_xy_single_trial(self.cfg, trial_n)

            # print(f"size of x_data: {x_data.size()}")

            self.X_mic_data.append(x_data)
            self.Y_label_data.append(y_label)
            self.wav_data_list.append(wav_data)
            self.qt_data_list.append(qt_data)
            self.xt_data_list.append(xt_data)
            self.xdot_t_data_list.append(xdot_t_data)
            self.tdoa_data_list.append(tdoa_data)
            self.gcc_data_list.append(gcc_data)

        print(f"size of X_mic_data: {len(self.X_mic_data)}") #--> 100 trials
        self.X_mic_data = torch.stack(self.X_mic_data)  # Shape: [len_data, num_mics, num_mels, num_bins_time]

        if cfg.normalize_audio_data: 
            
            if cfg.transform == 'mel': #torch.Size([6, 16, 442]) - [num_mics, num_mels, num_bins_time]

                # Calculate mean and variance for each microphone across all trials
                mean = self.X_mic_data.mean(dim=[0, 2, 3], keepdim=True)  # Mean across trials, num_mels, and num_bins_time
                var = self.X_mic_data.var(dim=[0, 2, 3], keepdim=True)    # Variance across trials, num_mels, and num_bins_time

                #save the mean and var output to a npy file (mean dimension: [num_mics, 1, 1, 1], var dimension: [num_mics, 1, 1, 1])
                if cfg.save_meanvar_output:
                    #stack mean and var into a single tensor
                    meanvar = torch.cat((mean, var), dim=0)
                    #save to npy file
                    np.save(f"{cfg.data_dir}/meanvar.npy", meanvar.cpu().numpy())
                    
                    
                if cfg.load_meanvar_output:
                    #load mean and var from npy file
                    meanvar = np.load(f"{cfg.model_directory}/meanvar.npy")
                    mean = torch.tensor(meanvar[0])
                    var = torch.tensor(meanvar[1])
                    # print(f"Loaded mean: {mean}, var: , {var}")

                for i in range(len_data):
                    self.X_mic_data[i] = (self.X_mic_data[i] - mean) / torch.sqrt(var)
                    # self.X_mic_data[i] = (self.X_mic_data[i] - mean) / np.sqrt(var)

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

        
        wavs = []
        melspecs = []
        qt = None
        xt = None
        xdot_t = None

        for i in range(len(cfg.device_list)):
            wav_filename = f"{self.dir[trial_n]}/mic{self.cfg.device_list[i]}.wav"

            wav, sample_rate = torchaudio.load(wav_filename)
            self.sample_rate = sample_rate
            # print(f"wave size: {wav.size()}, sample rate: {sample_rate}") #--> sample rate: 44100

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            wav = mic_utils.trim_or_pad(wav, self.cfg.max_num_frames)

            #append to list of wavs
            wavs.append(wav.squeeze(0)) # remove the dimension of size 1

            # print(f"size of wav {wav.size()}") # --> torch.Size([6, 88200])

            #extract robot propriocetion data (joint trajectory)
            qt_filename = f"{self.dir[trial_n]}/q_t.npy"
            qt = np.load(qt_filename)

            xt_filename = f"{self.dir[trial_n]}/x_t.npy"
            xdot_t_filename = f"{self.dir[trial_n]}/xdot_t.npy"

            xt = np.load(xt_filename)
            xdot_t = np.load(xdot_t_filename)


        # print(f"size of qt: {qt.shape}") #--> size of qt: (200, 6)
        # extract only the first 100 samples (1 sec) of the joint trajectory
        qt = qt[:50]
        xt = xt[:50]
        xdot_t = xdot_t[:50]

        # print(f" dimensions of qt: {qt.shape}, xt: {xt.shape}, xdot_t: {xdot_t.shape}")
        # sys.exit()


        # print(f"len of wavs {len(wavs)}") # --> 1 
        #convert [ torch.Size([6, 88200]) ] into list of 6 items with 88200
        wavs_list = []
        for i in range(self.num_mic ):
            wavs_list.append(wavs[0][i])
        wavs = wavs_list

        # print(f"len of wavs {len(wavs)}") # --> 6
            
        # center around the max spike in the audio data 
        trimmed_wavs = mic_utils.trim_audio_around_peak(wavs, self.sample_rate, self.cfg.window_frame)

        #plot trimmed wav to check if it is centered around the peak
        # mic_utils.plot_time_domain(trimmed_wavs, self.sample_rate)


        if self.cfg.subtract_background: 
            #trim the background noise length to match the wav length
            trimmed_background_wavs = mic_utils.trim_audio_around_peak(self.background_wavs, self.sample_rate, self.cfg.window_frame)

        dummy_visualization_before_background = []
        for i in range(self.num_mic ):
            if self.cfg.subtract_background: 
                dummy_visualization_before_background.append(trimmed_wavs[i])
                trimmed_wavs[i] = mic_utils.subtract_background_noise(trimmed_wavs[i], trimmed_background_wavs[i], self.sample_rate)
                
                # print(f"dim of wav after noise reduction: {wavs[i].size()}")

            #apply transform to wav file
            if self.transform:
                # print(f"input shape of wav transform: {trimmed_wavs[i].size()}")
                mel = self.transform(self.cfg, trimmed_wavs[i].float())
                # print(f"trial number: {trial_n} output shape of mel transform: {mel.size()}") #--> output shape of transform: torch.Size([16, 690])
                melspecs.append(mel.squeeze(0)) # remove the dimension of size 1


        if cfg.visualize_subtract_background:
            # mic_utils.plot_fft(self.background_wavs, cfg.sample_rate, [1,2,3,4,5,6])
            mic_utils.plot_fft(wavs, sample_rate, [1,2,3,4,5,6])
            mic_utils.plot_fft(trimmed_wavs, sample_rate, [1,2,3,4,5,6])


            # mic_utils.plot_spectrogram_with_cfg(cfg, self.background_wavs, cfg.sample_rate)
            # mic_utils.plot_spectrogram_with_cfg(cfg, wavs, cfg.sample_rate)
            # mic_utils.plot_spectrogram_with_cfg(cfg, trimmed_wavs, cfg.sample_rate)

            # mic_utils.grid_plot_spectrogram(trimmed_background_wavs, cfg.sample_rate)
            # mic_utils.grid_plot_spectrogram(dummy_visualization_before_background, cfg.sample_rate)
            # mic_utils.grid_plot_spectrogram(trimmed_wavs, cfg.sample_rate)
            
            
            # sys.exit()
            
                
        # stack wav files into a tensor of shape (num_mics, num_samples)
        wav_tensor = torch.stack(trimmed_wavs, dim=0)
        # print(f"dimension of wav tensor: {wav.size()}") #--> dimension of wav tensor: torch.Size([6, 88200])

        #stack mel spectrograms into a tensor of shape (num_mics, num_mels, num_samples)
        mel_tensor = torch.stack(melspecs, dim=0)
        # print(f"size of mel_tensor: {mel_tensor.size()}") #--> size of data: torch.Size([6, 16, 690])

        # get GCC-PHAT from the time domain wav_tensor (6, 88200)
        max_delay_between_microphones=0.01 # 10 ms based on empirical data
        gcc_matrix, tau, pairs  = mic_utils.compute_gcc_phat_matrix(wav_tensor, self.sample_rate, max_delay_between_microphones)

        # print(f"dimensions of gcc_matrix: {gcc_matrix.shape}, tau: {tau.shape}, pairs: {pairs}")
        # sys.exit()

        tdoa, tdoa_idx = mic_utils.get_time_differences(gcc_matrix, tau)
        
        # Plot gcc phat, tdoa
        # mic_utils.plot_gcc_phat_results(gcc_matrix, tau, pairs)



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

        return data, label, wav_tensor, qt, xt, xdot_t, tdoa, gcc_matrix
    
    def __len__(self):
        return len(self.dir)
    
    def __getitem__(self, idx):
        x,y = self.X_mic_data[idx], self.Y_label_data[idx] # --> x: full 2 sec wav/spectrogram
        wav = self.wav_data_list[idx]
        qt = self.qt_data_list[idx]
        xt = self.xt_data_list[idx]
        xdot_t = self.xdot_t_data_list[idx]

        tdoa = self.tdoa_data_list[idx]
        gcc = self.gcc_data_list[idx]

        # print(f"index of x: {idx}, y: {y}")

        # print(f"shape of x: {x.size()}") #--> shape of x: torch.Size([6, 16, 690]
        
        num_freq_bins = x.shape[2] #345
        desired_bin_length = self.cfg.input_spectrogram_bins #0.2 second for given NFFT and hoplength

        start = num_freq_bins//2 - desired_bin_length // 2
        end = start + desired_bin_length

        if self.augment:

            if self.cfg.augment_timeshift:
                #randomly shift the starting bin (plus or minus) by 10% of the desired bin length
                shift = random.randint(-desired_bin_length//10, desired_bin_length//10)
                start = start + shift
                start = max(0, start) #ensure start is not negative
                end = start + desired_bin_length

                # print(f"start: {start}, end: {end}, size of x: {x.size()}")
            x = x[:, :, start:end] #--> [num_mics, num_mels, 276] 

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
            #input size of 690. 
            #center crop the spectrogram around starting bin (so uses 138 bins before and after)

            start = num_freq_bins//2 - desired_bin_length // 2
            end = start + desired_bin_length

            x = x[:, :, start:end]

        
        # print(f"size of x after augmentation: {x.size()}") #--> size of x after augmentation: torch.Size([6, 16, 276])
        return x,y, wav, qt, xt, xdot_t, tdoa, gcc




def plot_tdoa(dataset):
    """
    input: dataset
    output: plot of tdoa values over trials

    There are 150 trials in the dataset
    5 trials per line.
    10 lines per locations.
    3 different locations.
    
    tdoa input dim is 15 for each trial, because of all possible combinations of 6 microphones.
    Append sequences of 5 trials to complete a line.
    There should be 15 subplots in total for all combination pairs. Each subplot has 5 tdoa values.
    Show this subplot 10 times.

    Only perform this for the first 50 trials.
    """
    # Generate all microphone pairs
    pairs = list(combinations(range(6), 2))  # 15 pairs

    # Initialize tdoa list
    tdoa_list = []

    # Collect first 50 trials
    for i, (x, y, _, qt, xt, xdot_t, tdoa) in enumerate(dataset):
        if i >= 50:  # Only process first 50 trials
            break
        tdoa_list.append(tdoa)

    # Convert to tensor for easier manipulation
    tdoa_tensor = torch.stack(tdoa_list)  # [50, 15]

    print(f"dimensions of tdoa_tensor: {tdoa_tensor.shape}")

    time_delay_max = 3 #millisecond

    # Plot 10 sets of 5 trials each
    for block in range(10):  # 10 blocks
        start_idx = block * 5
        end_idx = start_idx + 5
        
        # Create figure with 15 subplots (3x5 grid)
        fig, axes = plt.subplots(5, 3, figsize=(15, 25))
        fig.suptitle(f'TDOA Values - Trials {start_idx} to {end_idx-1}', fontsize=16)
        
        # Plot each mic pair
        for pair_idx, ((mic1, mic2), ax) in enumerate(zip(pairs, axes.flatten())):
            # Get TDOA values for this pair across 5 trials
            tdoa_values = tdoa_tensor[start_idx:end_idx, pair_idx]
            
            # Create plot
            trial_nums = range(start_idx, end_idx)
            ax.plot(trial_nums, tdoa_values , 'bo-')  # shown in  milliseconds
            ax.set_title(f'Mic {mic1+1}-{mic2+1}')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('TDOA (ms)')
            ax.grid(True)
            
            # Set reasonable y-axis limits
            max_val = tdoa_values.max() 
            min_val = tdoa_values.min() 
            padding = 0.1 * (max_val - min_val)
            if padding == 0:  # Handle case where all values are the same
                padding = 0.1 * abs(max_val)
            ax.set_ylim([min_val - padding, max_val + padding])
            
            # Set x-axis limits
            ax.set_xlim([start_idx - 0.5, end_idx - 0.5])

            # set y-axis limits
            ax.set_ylim([-time_delay_max, time_delay_max])
            
            # Add trial points
            ax.scatter(trial_nums, tdoa_values , color='blue')

        plt.tight_layout()
        plt.show()

def load_data(cfg, train_or_val = 'val'):
    """
    Load the dataset and split it into train and validation
    """
    
    dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augment = cfg.augment_data)

    # plot_tdoa(dataset)
    # sys.exit()

  
    #visuaize dataset
    if cfg.visuaize_dataset:
        for i in range(5):
            
            #get first element of dataset
            x, y = dataset[i]
            #convert to numpy
            x = x.numpy()
            #convert to list of 1st channel --> [[40, 345] ... [40, 345] ]
            x = [x[i] for i in range(x.shape[0])]
            # print(f"x {x}")
            # print(f"size of x: {len(x)}")
            # print(f"i: {i} and y: {y}")
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

