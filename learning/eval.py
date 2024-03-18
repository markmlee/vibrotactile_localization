import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

from datasets import AudioDataset 
import torch
import logging
import os
from tqdm import tqdm
import sys
from easydict import EasyDict

from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np

#logger
import logging
log = logging.getLogger(__name__)
from logger import Logger

#dataset
from datasets import AudioDataset, AudioDataset_test
from datasets import load_data

#models
from models.KNN import KNN
from models.CNN import CNNRegressor

#eval
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from datasets import plot_spectrogram, plot_spectrogram_of_all_data
from train import plot_regression

torch.manual_seed(42)


def load_data(cfg):
    #load data
    dataset = AudioDataset_test(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augmentation = cfg.augmentation_type)

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




@hydra.main(version_base='1.3',config_path='configs', config_name = 'eval')
def main(cfg: DictConfig):
    print(f" --------- eval --------- ")


    #load model.pth from checkpoint
    model = CNNRegressor(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))

    #verify if model is loaded by checking the model parameters
    print(model)

    #load data
    train_loader, val_loader = load_data(cfg)

    model.eval()

    height_error, radian_error = 0,0
    y_val_list = []
    y_pred_list = []

    for _, (x, y) in enumerate(tqdm(val_loader)):

        #plot spectrogram for all data
        # plot_spectrogram_of_all_data(cfg, x, 44100) # --> [batch_size, mic, freq, time]
        # sys.exit()

        #plot spectrogram for visualization
        # plot_spectrogram(cfg, x[0], 44100)
        # sys.exit()

        x_val = x
        y_val = y

        with torch.no_grad():
            height, radian = model(x_val) # --> CNN separate-head output

            #reshape height and radian to be same shape as y_val
            height = height.view(-1)
            radian = radian.view(-1)

            height_diff = height - y_val[:,0]
            radian_diff = radian - y_val[:,1]
        

            #get absolute error
            height_error += torch.mean(torch.abs(height_diff)) 
            radian_error += torch.mean(torch.abs(radian_diff))

            #combine height and radian to y_pred
            y_pred = torch.stack((height, radian), dim=1)

        
        #get tensor values and append them to list
        y_val_list.extend(y_val.cpu().numpy())
        y_pred_list.extend(y_pred.cpu().numpy())
        
            
    #sum up the rmse and divide by number of batches
    height_error = (height_error) / len(val_loader)
    radian_error = (radian_error) / len(val_loader)

    error = height_error + radian_error
    print(f"Mean Absolute Error: {error}")

    #save y_pred and y_val npy files
    np.save('y_pred.npy', y_pred_list)
    np.save('y_val.npy', y_val_list)
    
    
    #plot regression line
    plot_regression(cfg,y_pred_list, y_val_list)


    return error



   

if __name__ == '__main__':
    main()