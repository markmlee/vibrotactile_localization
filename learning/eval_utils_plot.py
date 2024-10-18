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
from datasets import AudioDataset
from datasets import load_data

#models
from models.KNN import KNN
from models.CNN import CNNRegressor

#eval
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

#import function from another directory for plotting
sys.path.insert(0,'/home/mark/audio_learning_project/vibrotactile_localization/scripts')
import microphone_utils as mic_utils


import matplotlib.pyplot as plt
import math

torch.manual_seed(42)
np.random.seed(42)

def plot_regression(cfg, y_pred_list, y_val_list):
    """
    Plot regression plot and histogram of errors for each variable in y
    """
    num_var = 2
    if cfg.output_representation == 'radian':
        num_var = 2
    elif cfg.output_representation == 'xy':
        num_var = 3

    elif cfg.output_representation == 'height':
        num_var = 1

    # Check if y_val_list and y_pred_list are 1D arrays
    if y_val_list.ndim == 1:
        y_val_list = y_val_list[:, np.newaxis]
    if y_pred_list.ndim == 1:
        y_pred_list = y_pred_list[:, np.newaxis]

    # Initialize layout
    fig, axs = plt.subplots(2, num_var, figsize = (18, 12))

    for i in range(num_var):
        # Regression plot
        ax = axs[0, i] if num_var > 1 else axs[0]
        # print(f"y_val_list: {y_val_list}, y_pred_list: {y_pred_list}")
        #add scatter plot
        ax.scatter(y_val_list[:, i], y_pred_list[:, i], alpha=0.7, edgecolors='k')
        #add regression line
        b,a, = np.polyfit(y_val_list[:, i], y_pred_list[:, i], deg=1)
        xseq = np.arange(0,len(y_val_list),1)
        ax.plot(xseq, a + b * xseq, color = 'r', lw=2.5)

        # Set x and y axis limits
        ax.set_xlim([y_val_list[:, i].min(), y_val_list[:, i].max()])
        ax.set_ylim([y_pred_list[:, i].min(), y_pred_list[:, i].max()])
        
        # Calculate R-squared metric
        r2 = r2_score(y_val_list[:, i], y_pred_list[:, i])
        # Display R-squared on the plot
        ax.text(0.05, 0.95, f'R^2 = {r2:.2f}', transform=ax.transAxes)
        if num_var == 2:
            if i == 0:
                ax.set_title('Height')
            else:
                ax.set_title('Radian')
        if num_var == 3:
            if i == 0:
                ax.set_title('Height')
            elif i == 1:
                ax.set_title('X')
            else:
                ax.set_title('Y')
        
        ax.set_xlabel('y_val')
        ax.set_ylabel('y_pred')

        # Histogram of errors
        ax = axs[1, i] if num_var > 1 else axs[1]
        errors = np.abs(y_val_list[:, i] - y_pred_list[:, i])
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='k')
        ax.set_title(f'Histogram of prediction errors {i+1}')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')

    #save the plots in output directory
    if cfg.visuaize_regression:
        plt.savefig(os.path.join(cfg.checkpoint_dir, 'plots.png'))
        plt.show()



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
