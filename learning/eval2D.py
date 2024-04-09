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
from models.KNN import KNN
from models.CNN import CNNRegressor, CNNRegressor2D

#eval
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import eval_utils as eval_utils

torch.manual_seed(42)


def load_data(cfg):
    #load data
    dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augment = False)

    # split the dataset into train and validation 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    #USE SEQUENTIAL SPLIT
    # Created using indices from 0 to train_size.
    # train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    # Created using indices from train_size to the end.
    # val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))



    #load train and val loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False)

    return train_loader, val_loader




@hydra.main(version_base='1.3',config_path='configs', config_name = 'eval2D')
def main(cfg: DictConfig):
    print(f" --------- eval --------- ")


    #load model.pth from checkpoint
    model = CNNRegressor2D(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #verify if model is loaded by checking the model parameters
    print(model)

    #load data
    train_loader, val_loader = load_data(cfg)

    model.eval()

    height_error, xy_error = 0,0
    y_val_list = []
    y_pred_list = []

    for _, (x, y) in enumerate(tqdm(val_loader)):

        #plot spectrogram for all data
        # mic_utils.plot_spectrogram_of_all_data(cfg, x, 44100) # --> [batch_size, mic, freq, time]
        # sys.exit()

        #plot spectrogram for visualization
        # plot_spectrogram(cfg, x[0], 44100)
        # sys.exit()

        x_val, Y_val = x,y

        with torch.no_grad():
            Y_pred = model(x_val) 

            #split prediction to height and radian
            height_pred = Y_pred[:,0]
            x_pred = Y_pred[:,1]
            y_pred = Y_pred[:,2]
            # radian_pred = torch.atan2(y_pred, x_pred)

            #convert y_val to radian
            x_val = Y_val[:,1]
            y_val = Y_val[:,2]
            # radian_val = torch.atan2(y_val, x_val)

            
            #reshape height and radian to be same shape as y_val
            height_pred = height_pred.view(-1)
            x_pred = x_pred.view(-1)
            y_pred = y_pred.view(-1)

            height_diff = height_pred - Y_val[:,0]
            x_diff = x_pred - Y_val[:,1]
            y_diff = y_pred - Y_val[:,2]

            print(f"height diff: {height_diff}, x_diff: {x_diff}, y_diff: {y_diff}")

        

            #get absolute error
            height_error += torch.mean(torch.abs(height_diff)) 
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

    error = height_error + xy_error
    print(f"Mean Absolute Error: {error}, Height Error: {height_error}, xy Error: {xy_error}")

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