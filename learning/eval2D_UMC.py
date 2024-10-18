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
from models.Resnet import ResNet18_audio, ResNet50_audio, ResNet50_audio_proprioceptive, ResNet50_audio_proprioceptive_dropout
from models.AudioSpectrogramTransformer import AST

#eval
from sklearn.metrics import mean_squared_error
import eval_utils_plot as eval_utils_plot

#import function from another directory for helper func
sys.path.insert(0,'/home/mark/audio_learning_project/vibrotactile_localization/scripts')
import eval_utils as pcloud_eval_utils


torch.manual_seed(42)


def load_data(cfg):
    #load data
    dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augment = False)

    #visuaize dataset
    if cfg.visuaize_dataset:
        for i in range(10):
            
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
            mic_utils.plot_spectrogram_with_cfg(cfg, x, dataset.sample_rate)

            #plot wav
            mic_utils.grid_plot_time_domain(wav, dataset.sample_rate)

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

def load_data_eval(cfg, data_dir):
    #load data
    dataset = AudioDataset(cfg=cfg, data_dir = data_dir, transform = cfg.transform, augment = False)

    #visuaize dataset
    if cfg.visuaize_dataset:
        for i in range(10):
            
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
            mic_utils.plot_spectrogram_with_cfg(cfg, x, dataset.sample_rate)

            #plot wav
            mic_utils.grid_plot_time_domain(wav, dataset.sample_rate)

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




def predict_and_eval(cfg, model, val_loader, device):
    """
    predict and evaluate the model
    return MAE of height, xy, and degree
    """

    height_error, xy_error = 0,0
    degree_error = 0
    MAE_error = 0
    y_val_list = []
    y_pred_list = []

    for _, (x, y, _, qt) in enumerate(tqdm(val_loader)):

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

        with torch.no_grad():
            # Y_output = model(x_input) # --> single input model
            Y_output = model(x_input, qt_val)

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
            if cfg.offset_radian_rightside:
                #offset the radian to the opposite side of the circle while handling the wrap around issue
                radian_val = radian_val + math.pi
                radian_val = torch.remainder(radian_val, 2*math.pi)
            if cfg.offset_radian_lateralside:
                #offset the radian for laterial strike (by -90 degrees) while handling the wrap around issue
                radian_val = radian_val - math.pi/2
                radian_val = torch.remainder(radian_val, 2*math.pi)
                
                

            #convert y_pred to radian
            radian_pred = torch.atan2(y_pred, x_pred)

            #resolve wrap around angle issues
            radian_error = eval_utils_plot.calculate_radian_error(radian_pred, radian_val)
            degree_diff = torch.rad2deg(radian_error)

            
            #reshape height and radian to be same shape as y_val
            height_pred = height_pred.view(-1)
            x_pred = x_pred.view(-1)
            y_pred = y_pred.view(-1)

            height_diff = height_pred - Y_val[:,0]
            x_diff = x_pred - Y_val[:,1]
            y_diff = y_pred - Y_val[:,2]

            


            if cfg.offset_radian_lateralside:
                print(f"height diff before: {height_diff}, length: {len(height_diff)}")
                #discard every 5th element in the height_diff (keep 0-4, discard 5, keep 6-9, discard 10, etc.)
                height_diff = [height_diff[i] for i in range(len(height_diff)) if i % 5 != 4]
                #convert list to tensor
                height_diff = torch.tensor(height_diff)
                print(f"height diff after: {height_diff}, length: {len(height_diff)}")

            print(f"height pred: {height_pred}, height GT: {Y_val[:,0]}")
            print(f"rad pred: {radian_pred}, rad GT: {radian_val}")
            print(f"x pred: {x_pred}, x GT: {x_val}, y pred: {y_pred}, y GT: {y_val}")
            print(f"height diff: {height_diff}, degree_diff: {degree_diff}")

        

            #get absolute error
            height_error += torch.mean(torch.abs(height_diff)) 
            degree_error  += torch.mean(torch.abs(torch.rad2deg(radian_error) ) )
            xy_error += torch.mean(torch.abs(x_diff) + torch.abs(y_diff))

            # ------------------------------------------

            # convert to xyz point
            pt_predict = pcloud_eval_utils.convert_h_rad_to_xyz(height_pred.cpu().numpy(), radian_pred.cpu().numpy(), cfg.cylinder_radius)
            pt_gt = pcloud_eval_utils.convert_h_rad_to_xyz(Y_val[:,0].cpu().numpy(), radian_val.cpu().numpy(), cfg.cylinder_radius)

            # get MAE between 2 pts
            MAE_pt = pcloud_eval_utils.compute_MAE_contact_point([pt_predict], [pt_gt])

            # ------------------------------------------

            MAE_error += MAE_pt

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
    MAE_error = (MAE_error) / len(val_loader)


    print(f"Height Error: {height_error}, xy Error: {xy_error}, Degree Error: {degree_error}, MAE Error: {MAE_error}")

    #save y_pred and y_val npy files
    # np.save('y_pred.npy', y_pred_list)
    # np.save('y_val.npy', y_val_list)
    
    #convert list to numpy array
    # y_pred_list = np.array(y_pred_list)
    # y_val_list = np.array(y_val_list)

    #plot regression line
    # eval_utils.plot_regression(cfg,y_pred_list, y_val_list)


    return height_error, xy_error, degree_error, MAE_error


@hydra.main(version_base='1.3',config_path='configs', config_name = 'eval2D_UMC')
def main(cfg: DictConfig):
    print(f" --------- eval --------- ")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")


    #load model.pth from checkpoint
    # model = CNNRegressor2D(cfg)
    # model = ResNet50_audio(cfg)
    # model = AST(cfg)
    # model = ResNet50_audio_proprioceptive(cfg)
    model = ResNet50_audio_proprioceptive_dropout(cfg)


    print(f"model: {model}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    

    model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))

    #verify if model is loaded by checking the model parameters
    # print(model)
    model.to(device)

    #list of eval data directories to iterate over
    data_dir_list1 = ['/home/mark/audio_learning_project/data/test_generalization/stick_T12L42_Y_25/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T12L42_Y_35/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T12L42_Y_40/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T22L42_Y_25/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T22L42_Y_35/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T22L42_Y_40/', 
                     '/home/mark/audio_learning_project/data/test_generalization/stick_T25L42_Y_25/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T25L42_Y_35/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T25L42_Y_40/',
                      '/home/mark/audio_learning_project/data/test_generalization/stick_T32L42_Y_25/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T32L42_Y_35/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T32L42_Y_40/'   ] #test set that contains training set data
    
    data_dir_list2 = ['/home/mark/audio_learning_project/data/test_generalization/stick_T22L80_Y_25/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T22L80_Y_35/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T22L80_Y_40/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T50L42_Y_25/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T50L42_Y_35/',
                    '/home/mark/audio_learning_project/data/test_generalization/stick_T50L42_Y_40/' ] # val set that contains novel stick data
    
    data_dir_list3 = ['/home/mark/audio_learning_project/data/test_generalization/cross_easy_Y_25_Left/',
                    '/home/mark/audio_learning_project/data/test_generalization/cross_easy_Y_25_Right/',
                    '/home/mark/audio_learning_project/data/test_generalization/cross_easy_Y_32_Left/',
                    '/home/mark/audio_learning_project/data/test_generalization/cross_easy_X_10_Left/',
                    '/home/mark/audio_learning_project/data/test_generalization/cross_easy_X_15_Left/' ] # val set that contains novel obj data
    
    data_dir_list4 = ['/home/mark/audio_learning_project/data/test_mapping/cross_easy_uniformexplore_v2/']
    data_dir_list5 = ['/home/mark/audio_learning_project/data/test_generalization/stick_T22L42_Y_40_w_suctionv5/']
    data_dir_list6 = ['/home/mark/audio_learning_project/data/test_generalization/stick_T25L42_Y_25_consistent_test_noAmpl_100/']

    data_dir_debug = ['/home/mark/audio_learning_project/data/wood_T25_L42_Horizontal_v2_mini/']

    data_dir_list = data_dir_list3 #choose the data_dir_list to use

    num_eval_dirs = len(data_dir_list)

    model.eval()

    height_error_list = []
    xy_error_list = []
    degree_error_list = []
    MAE_error_list = []

    #predict and evaluate
    for eval_dir in data_dir_list:

        #load data
        train_loader, val_loader = load_data_eval(cfg, data_dir = eval_dir)

        #predict and evaluate
        height_error, xy_error, degree_error, MAE_error = predict_and_eval(cfg, model, val_loader, device)

        #append to list
        height_error_list.append(height_error)
        xy_error_list.append(xy_error)
        degree_error_list.append(degree_error)
        MAE_error_list.append(MAE_error)

    
    #average the errors
    height_error_avg = sum(height_error_list) / num_eval_dirs
    xy_error_avg = sum(xy_error_list) / num_eval_dirs
    degree_error_avg = sum(degree_error_list) / num_eval_dirs
    MAE_error_avg = sum(MAE_error_list) / num_eval_dirs

    print(f"Average Height Error: {height_error_avg}, Average xy Error: {xy_error_avg}, Average Degree Error: {degree_error_avg}, MAE_error_avg: {MAE_error_avg}")
    



   

if __name__ == '__main__':
    main()