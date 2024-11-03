import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

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
# from models.KNN import KNN
# from models.CNN import CNNRegressor

#eval
# from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

#import function from another directory for plotting
sys.path.insert(0,'/home/mark/audio_learning_project/vibrotactile_localization/scripts')
import microphone_utils as mic_utils
import eval_utils as pcloud_eval_utils

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


def predict_from_eval_dataset(cfg, model, device, val_loader ):
    
    total_trials = len(val_loader)
    y_val_list = []
    y_pred_list = []

    all_distances = []

    height_error, degree_error, xy_error = 0, 0, 0


    for _, (x, y, _, qt, xt, xdot_t, tdoa, gcc) in enumerate(tqdm(val_loader)):

        
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
        qt_val = qt.float().to(device)
        tdoa_val = tdoa.float().to(device)
        xt_val = xt.float().to(device)
        xdot_t_val = xdot_t.float().to(device)
        gcc_val = gcc.float().to(device)

        xt_xdot_t = torch.cat((xt_val, xdot_t_val), dim=2)

        with torch.no_grad():
            #TODO: MODIFY ACCORDING TO MODEL
            # Y_output = model(x_input) # --> single input model
            # Y_output = model(x_input, qt_val)
            # Y_output = model(x_input, xt_val)
            # Y_output = model(x_input, xt_xdot_t)
            # Y_output = model(x_input, qt_val, tdoa_val)
            # Y_output = model(x_input, xt_xdot_t, tdoa_val)
            Y_output = model(x_input, xt_xdot_t, gcc_val)
            # Y_output = model(x_input, xt_xdot_t, phase)

            #split prediction to height and radian
            height_pred = Y_output[:,0]

            #clip height to [-11, +11]
            height_pred = torch.clamp(height_pred, -11, 11)

            x_pred = Y_output[:,1]
            y_pred = Y_output[:,2]

            #clip x and y to [-1, +1]
            x_pred = torch.clamp(x_pred, -1, 1)
            y_pred = torch.clamp(y_pred, -1, 1)

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
            print(f"x pred: {x_pred}, x GT: {x_val}, y pred: {y_pred}, y GT: {y_val}")
            print(f"height diff: {height_diff}, degree_diff: {degree_diff}")

        
            #get absolute error
            height_error += torch.mean(torch.abs(height_diff)) 
            degree_error  += torch.mean(torch.abs(torch.rad2deg(radian_error) ) )
            xy_error += torch.mean(torch.abs(x_diff) + torch.abs(y_diff))

            # ------------------------------------------ #EVAL IN BATCHES ------------------------------------------

            # # convert to xyz point
            # pt_predict = pcloud_eval_utils.convert_h_rad_to_xyz(height_pred.cpu().numpy(), radian_pred.cpu().numpy(), cfg.cylinder_radius)
            # pt_gt = pcloud_eval_utils.convert_h_rad_to_xyz(Y_val[:,0].cpu().numpy(), radian_val.cpu().numpy(), cfg.cylinder_radius)

            # # get distances for all points in batch
            # # distances = np.sqrt(np.sum((pt_predict - pt_gt) ** 2, axis=1))  # Compute for each point in batch

            # source, target = [pt_predict], [pt_gt]
            # # Convert lists of points to numpy arrays
            # source_points = np.array([[pt.x, pt.y, pt.z] for pt in source])
            # target_points = np.array([[pt.x, pt.y, pt.z] for pt in target])

            # # Compute the Euclidean distance between the contact points
            # distances = np.linalg.norm(source_points - target_points, axis=1)

            # all_distances.extend(distances)  # Add all distances from this batch

            # # get MAE between 2 pts
            # # MED_pt = pcloud_eval_utils.compute_euclidean_distance([pt_predict], [pt_gt])
            # #  MAE_pt = pcloud_eval_utils.compute_MAE_contact_point([pt_predict], [pt_gt])

            # ------------------------------------------    ------------------------------------------

            # Convert predictions to 3D points for distance calculation
            for i in range(len(height_pred)):
                # Convert individual predictions to xyz points
                pt_predict = pcloud_eval_utils.convert_h_rad_to_xyz(
                    height_pred[i].cpu().numpy(),
                    radian_pred[i].cpu().numpy(),
                    cfg.cylinder_radius
                )
                pt_gt = pcloud_eval_utils.convert_h_rad_to_xyz(
                    Y_val[i,0].cpu().numpy(),
                    radian_val[i].cpu().numpy(),
                    cfg.cylinder_radius
                )
                
                # Calculate Euclidean distance for this pair
                distance = np.sqrt(
                    (pt_predict.x - pt_gt.x)**2 + 
                    (pt_predict.y - pt_gt.y)**2 + 
                    (pt_predict.z - pt_gt.z)**2
                )
                all_distances.append(distance)


            # MED_error += MED_pt
            # MED_list.append(MED_pt)

            #combine height and radian to y_pred
            # y_pred = torch.stack((height_pred, x_pred, y_pred), dim=1)
            # y_val_ = torch.stack((Y_val[:,0], x_val, y_val), dim=1)

        
            # #get tensor values and append them to list
            # y_val_list.extend(y_val_.cpu().numpy())
            # y_pred_list.extend(y_pred.cpu().numpy())
        
            
    #sum up the rmse and divide by number of batches
    height_error = (height_error) / len(val_loader)
    xy_error = (xy_error) / len(val_loader)
    degree_error = (degree_error) / len(val_loader)


    #convert list to numpy array for MED list
    # MED_list = np.array(MED_list)
    all_distances = np.array(all_distances)

    return height_pred, x_pred, y_pred, total_trials, height_error, xy_error, degree_error, all_distances, y_val_list, y_pred_list

def predict_and_evaluate_val_dataset(cfg, model, device, val_loader ):
    """
    predict and evaluate the model
    return MAE of height, xy, and degree
    """

    model.eval()

    height_error, xy_error = 0,0
    degree_error = 0
    MED_list = []
    

    all_distances = []  # Store all individual distances

    #predict model
    height_pred, x_pred, y_pred, total_trials, height_error, xy_error, degree_error, all_distances, y_val_list, y_pred_list = predict_from_eval_dataset(cfg, model, device, val_loader)
    

    #compute mean and std of MED
    MED_mean = np.mean(all_distances)
    MED_std = np.std(all_distances)


    print(f"Height Error: {height_error}, xy Error: {xy_error}, Degree Error: {degree_error}, Mean MED: {MED_mean}, STD MED: {MED_std}")

    #save y_pred and y_val npy files
    # np.save('y_pred.npy', y_pred_list)
    # np.save('y_val.npy', y_val_list)
    
    #stack the list of array to numpy array
    # y_val_list = np.stack(y_val_list)
    # y_pred_list = np.stack(y_pred_list)

    if cfg.visuaize_regression:
        #plot regression line
        plot_regression(cfg,y_pred_list, y_val_list)


    return height_error, xy_error, degree_error, MED_mean, MED_std, all_distances