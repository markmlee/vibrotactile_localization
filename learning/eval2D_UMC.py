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


def load_data(cfg):
    #load data
    dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augment = False)

    #visuaize dataset
    if cfg.visuaize_dataset:
        for i in range(5):
            
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

@hydra.main(version_base='1.3',config_path='configs', config_name = 'eval2D_UMC')
def main(cfg: DictConfig):
    print(f" --------- eval --------- ")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")


    #load model.pth from checkpoint
    model = CNNRegressor2D(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))

    #verify if model is loaded by checking the model parameters
    # print(model)
    model.to(device)


    #load data
    train_loader, val_loader = load_data(cfg)

    model.eval()

    height_error, xy_error = 0,0
    degree_error = 0
    y_val_list = []
    y_pred_list = []

    for _, (x, y, _) in enumerate(tqdm(val_loader)):

        #plot spectrogram for all data
        # mic_utils.plot_spectrogram_of_all_data(cfg, x, 44100) # --> [batch_size, mic, freq, time]
        # sys.exit()

        #plot spectrogram for visualization
        # plot_spectrogram(cfg, x[0], 44100)
        # sys.exit()

        # if cfg.visuaize_dataset:
            # for i in range(5):
            #     #convert to list of 1st channel --> [[40, 345] ... [40, 345] ]
            #     x_input_list = [x[i] for i in range(x.shape[0])]
            #     print(f"size of x_input_list: {len(x_input_list)}")
            #     print(f"i: {i} and y: {y}")
            #     # plot mel spectrogram
            #     mic_utils.plot_spectrogram_with_cfg(cfg, x_input_list[i], cfg.sample_rate)

        x_input, Y_val = x.float().to(device), y.float().to(device)

        with torch.no_grad():
            Y_output = model(x_input) 

            #split prediction to height and radian
            height_pred = Y_output[:,0]
            x_pred = Y_output[:,1]
            y_pred = Y_output[:,2]
            # radian_pred = torch.atan2(y_pred, x_pred)

            #convert y_val to radian
            x_val = Y_val[:,1]
            y_val = Y_val[:,2]
            radian_val = torch.atan2(y_val, x_val)

            #convert y_pred to radian
            radian_pred = torch.atan2(y_pred, x_pred)

            #resolve wrap around angle issues
            radian_error = calculate_radian_error(radian_pred, radian_val)
            degree_diff = torch.rad2deg(radian_error)

            
            #reshape height and radian to be same shape as y_val
            height_pred = height_pred.view(-1)
            x_pred = x_pred.view(-1)
            y_pred = y_pred.view(-1)

            height_diff = height_pred - Y_val[:,0]
            x_diff = x_pred - Y_val[:,1]
            y_diff = y_pred - Y_val[:,2]

            print(f"height pred: {height_pred}, height GT: {Y_val[:,0]}")
            print(f"rad pred: {radian_pred}, rad GT: {radian_val}")
            print(f"height diff: {height_diff}, degree_diff: {degree_diff}")

        

            #get absolute error
            height_error += torch.mean(torch.abs(height_diff)) 
            degree_error  += torch.mean(torch.abs(torch.rad2deg(radian_error) ) )
            xy_error += torch.mean(torch.abs(x_diff) + torch.abs(y_diff))

            #combine height and radian to y_pred
            y_pred = torch.stack((height_pred, x_pred, y_pred), dim=1)
            y_val_ = torch.stack((Y_val[:,0], x_val, y_val), dim=1)

        
        #get tensor values and append them to list
        y_val_list.extend(y_val_.cpu().numpy())
        y_pred_list.extend(y_pred.cpu().numpy())
        
            
    #sum up the rmse and divide by number of batches
    height_error = (height_error) / len(val_loader)
    xy_error = (xy_error) / len(val_loader)
    degree_error = (degree_error) / len(val_loader)

    error = height_error + xy_error
    print(f"Mean Absolute Error: {error}, Height Error: {height_error}, xy Error: {xy_error}, Degree Error: {degree_error}")

    #save y_pred and y_val npy files
    np.save('y_pred.npy', y_pred_list)
    np.save('y_val.npy', y_val_list)
    
    #convert list to numpy array
    y_pred_list = np.array(y_pred_list)
    y_val_list = np.array(y_val_list)

    #plot regression line
    # eval_utils.plot_regression(cfg,y_pred_list, y_val_list)


    return error



   

if __name__ == '__main__':
    main()