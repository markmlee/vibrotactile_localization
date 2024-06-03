import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

from datasets import AudioDataset 
import torch
import torchaudio
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

from transforms import to_mel_spectrogram, get_signal


TRANSFORMS = {
    'wav' : get_signal,
    'mel': to_mel_spectrogram
}

audio_transform = TRANSFORMS['mel'] # Transform to apply to the audio data


def load_xy_single_trial(cfg, trial_n, dir):
    """
    Load the data from a single trial
    """

    num_mics = len(cfg.device_list)
    

    wavs = []
    melspecs = []

    for i in range(num_mics):
        wav_filename = f"{dir[trial_n]}/mic{cfg.device_list[i]}.wav"

        if i == 0:
            print(f"wav_filename: {wav_filename}")

        wav, sample_rate = torchaudio.load(wav_filename)
        sample_rate = sample_rate
        # print(f"sample rate: {sample_rate}") #--> sample rate: 44100

        #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
        wav = mic_utils.trim_or_pad(wav, cfg.max_num_frames)


        #append to list of wavs
        wavs.append(wav.squeeze(0)) # remove the dimension of size 1

        #apply transform to wav file
        if audio_transform:
            mel = audio_transform(cfg, wavs[i].float())
            melspecs.append(mel.squeeze(0)) # remove the dimension of size 1

    # stack wav files into a tensor of shape (num_mics, num_samples)
    wav_tensor = torch.stack(wavs, dim=0)
    # print(f"dimension of wav tensor: {wav.size()}") #--> dimension of wav tensor: torch.Size([6, 88200])

    #stack mel spectrograms into a tensor of shape (num_mics, num_mels, num_samples)
    mel_tensor = torch.stack(melspecs, dim=0)
    # print(f"size of mel_tensor: {mel_tensor.size()}") #--> size of data: torch.Size([6, 16, 690])

    if cfg.audio_transform == 'mel':
        data = mel_tensor

    #get label from directory 
    label_file = f"{dir[trial_n]}/gt_label.npy"
    label = np.load(label_file) #--> [distance along cylinder, joint 6] i.e. [0.0 m, -2.7 radian]

    #convert label m unit to cm
    label[0] = label[0] * 100

    x,y  = np.cos(label[1]), np.sin(label[1])
    label[1] = x
    label = np.append(label, y) #--> [height, x, y]

    return data, label

@hydra.main(version_base='1.3',config_path='configs', config_name = 'eval2D_mini')
def main(cfg: DictConfig):
    print(f" --------- eval --------- ")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")


    #load model.pth from checkpoint
    model = CNNRegressor2D(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))

    #verify if model is loaded by checking the model parameters
    # print(model)

    model.eval()


    # ======================================== load data ========================================
    debug_trial_counter = 0
    total_height_error = 0
    num_trials = 100
    for trial_n in range(num_trials):
        #load validation data
        dir = sorted([os.path.join(cfg.data_dir, f) for f in os.listdir(cfg.data_dir) if f.startswith('trial')], key=lambda x: int(os.path.basename(x)[5:]))                #filter out directory that does not start with 'trial'
        dir = [d for d in dir if d.split('/')[-1].startswith('trial')]

        data, label = load_xy_single_trial(cfg, debug_trial_counter, dir)

        #normalize the raw input data using the mean,var from the training dataset
        meanvar_path = os.path.join(cfg.data_dir, 'meanvar.npy')
        meanvar_np = np.load(meanvar_path) #--> dimension [6,1,1,1] and [6,1,1,1] stacked together

        #trim audio to 345 frames (like getitem in dataset)
        start = (data.shape[2] - 345) // 2
        end = start + 345
        data = data[:, :, start:end]

        mean, var = meanvar_np[0], meanvar_np[1]
        data = (data - mean) / np.sqrt(var)

        mic_utils.plot_spectrogram_with_cfg(cfg, data, 44100)

        #unsqueeze to add batch dimension [1, num_mics, num_mels, num_samples]
        data = data.unsqueeze(0)


        
        Y_pred = model(data) 
        sys.exit()
        
        #tensor to numpy
        Y_pred = Y_pred.cpu().detach().numpy()

        print(f"Y_pred: {Y_pred}, Y_val: {label}")

        #height error
        height_error = np.abs(Y_pred[0][0] - label[0])
        print(f"height error: {height_error}")
        total_height_error += height_error

        debug_trial_counter += 1

    #average height error
    total_height_error = total_height_error / num_trials
    print(f"total height error: {total_height_error}")



    # ====================================================================================================
    


    


   

if __name__ == '__main__':
    main()