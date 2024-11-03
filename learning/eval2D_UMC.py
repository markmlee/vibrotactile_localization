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
from models.AudioSpectrogramTransformer import AST, AST_multimodal
from models.multimodal_transformer import MultiModalTransformer
from models.multimodal_transformer_xt_xdot import MultiModalTransformer_xt_xdot_t
from models.multimodal_transformer_xt_xdot_gcc import MultiModalTransformer_xt_xdot_t_gccphat
from models.multimodal_transformer_xt_xdot_gcc_tokens import MultiModalTransformer_xt_xdot_t_gccphat_tokens
from models.multimodal_transformer_xt_xdot_phase import MultiModalTransformer_xt_xdot_t_phase
from models.multimodal_transformer_xt_xdot_toda import MultiModalTransformer_xt_xdot_t_toda

#eval
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






@hydra.main(version_base='1.3',config_path='configs', config_name = 'eval2D_UMC')
def main(cfg: DictConfig):
    print(f" --------- eval --------- ")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    #TODO: update architecture
    #load model.pth from checkpoint
    # model = CNNRegressor2D(cfg)
    # model = ResNet50_audio(cfg)
    # model = AST(cfg)
    # model = AST_multimodal(cfg)
    # model = MultiModalTransformer(cfg)
    # model = MultiModalTransformer_xt_xdot_t(cfg)
    model = MultiModalTransformer_xt_xdot_t_gccphat(cfg)
    # model = MultiModalTransformer_xt_xdot_t_phase(cfg)
    # model = MultiModalTransformer_xt_xdot_t_toda(cfg)
    # model = MultiModalTransformer_xt_xdot_t_gccphat_tokens(cfg)
    # model = ResNet50_audio_proprioceptive(cfg)
    # model = ResNet50_audio_proprioceptive_dropout(cfg)


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
    
    data_dir_list4 = ['/home/mark/audio_learning_project/data/test_mapping/cross_easy_full_new_v4/']
    data_dir_list5 = ['/home/mark/audio_learning_project/data/test_generalization/stick_T22L42_Y_40_w_suctionv5/']
    data_dir_list6 = ['/home/mark/audio_learning_project/data/test_generalization/stick_T25L42_Y_25_consistent_test_noAmpl_100/']

    data_dir_debug = ['/home/mark/audio_learning_project/data/wood_T25_L42_Horizontal_v2_mini/']

    data_dir_list = data_dir_list4 #choose the data_dir_list to use

    num_eval_dirs = len(data_dir_list)

    model.eval()

    height_error_list = []
    xy_error_list = []
    degree_error_list = []
    MED_mean_list = []
    MED_std_list = []

    #predict and evaluate
    for eval_dir in data_dir_list:

        #load data
        train_loader, val_loader = load_data_eval(cfg, data_dir = eval_dir)

        #predict and evaluate
        height_error, xy_error, degree_error, MED_mean, MED_std, all_distances = eval_utils_plot.predict_and_evaluate_val_dataset(cfg, model, device, val_loader)

        #append to list
        height_error_list.append(height_error)
        xy_error_list.append(xy_error)
        degree_error_list.append(degree_error)
        MED_mean_list.append(MED_mean)
        MED_std_list.append(MED_std)

    
    #average the errors
    height_error_avg = sum(height_error_list) / num_eval_dirs
    xy_error_avg = sum(xy_error_list) / num_eval_dirs
    degree_error_avg = sum(degree_error_list) / num_eval_dirs
    MED_error_avg = sum(MED_mean_list) / num_eval_dirs
    MED_std_avg = sum(MED_std_list) / num_eval_dirs

    print(f"Average Height Error: {height_error_avg}, Average xy Error: {xy_error_avg}, Average Degree Error: {degree_error_avg}, MED_error_avg: {MED_error_avg}, MED_std_avg: {MED_std_avg}")
    



   

if __name__ == '__main__':
    main()