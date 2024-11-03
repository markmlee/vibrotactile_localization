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
import matplotlib.pyplot as plt

#logger
import logging
log = logging.getLogger(__name__)
from logger import Logger

#dataset
from datasets import AudioDataset
from datasets import load_data

#models
from models.KNN import KNN
from models.CNN import CNNRegressor, CNNRegressor2D, CNNRegressor1D, CNNRegressor_Classifier
from models.Resnet import ResNet18_audio, ResNet50_audio, ResNet50_audio_proprioceptive, ResNet50_audio_proprioceptive_dropout
from models.AudioSpectrogramTransformer import AST, AST_multimodal, AST_multimodal_qt
from models.multimodal_transformer import MultiModalTransformer
from models.multimodal_transformer_xt_xdot import MultiModalTransformer_xt_xdot_t
from models.multimodal_transformer_xt_xdot_gcc import MultiModalTransformer_xt_xdot_t_gccphat
from models.multimodal_transformer_xt_xdot_gcc_tokens import MultiModalTransformer_xt_xdot_t_gccphat_tokens
from models.multimodal_transformer_xt_xdot_toda import MultiModalTransformer_xt_xdot_t_toda
from models.multimodal_transformer_xt_xdot_phase import MultiModalTransformer_xt_xdot_t_phase
#eval
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from sklearn.metrics import accuracy_score
import eval_utils_plot as eval_utils_plot

#import function from another directory for plotting
sys.path.insert(0,'/home/mark/audio_learning_project/vibrotactile_localization/scripts')
import microphone_utils as mic_utils


#import function from another directory for helper func
sys.path.insert(0,'/home/mark/audio_learning_project/vibrotactile_localization/scripts')
import eval_utils as pcloud_eval_utils



torch.manual_seed(42)
np.random.seed(42)




def model_prediction(cfg, device, model, x, y, qt, xt, xdot_t, tdoa, gcc_matrix, criterion_list, weight_list):
    """
    Called every batch to predict y values and return the loss 
    """
    x_, y_ = x.float().to(device), y.float().to(device)
    qt_ = qt.float().to(device)
    xt_ = xt.float().to(device)
    xdot_t_ = xdot_t.float().to(device)
    tdoa_ = tdoa.float().to(device)
    gcc_matrix_ = gcc_matrix.float().to(device)

    # print(f"shapes of x_train, y_train: {x_train.shape}, {y_train.shape}") #--> torch.Size([80, 6, 40, 690]), torch.Size([80, 2])
    #TODO: concat xt_ and xdot_t_
    # print(f"xt_: {xt_.shape}, xdot_t_: {xdot_t_.shape}") #--> xt_: torch.Size([8, 50, 7]), xdot_t_: torch.Size([8, 50, 3])
    xt_xdot_t = torch.cat((xt_, xdot_t_), dim=2)

    #TODO: MODIFY HERE TO USE THE RIGHT MODEL
    # y_pred = model(x_) # --> CNN single-head output
    # y_pred = model(x_, xt_xdot_t) # --> Audio + proprioceptive input
    # y_pred = model(x_, qt_, tdoa_) # --> Audio + proprioceptive + tdoa input
    y_pred = model(x_, xt_xdot_t, gcc_matrix_) # --> Audio + proprioceptive + gcc input
    # y_pred = model(x_, xt_xdot_t, tdoa_) # --> Audio + proprioceptive + tdoa input
    # y_pred = model(x_, xt_xdot_t, phase_) # --> Audio + proprioceptive + tdoa input

    if cfg.output_representation == 'height':
        y_pred_height = y_pred
        y_train_height = y_
        
        
        criterion_height = criterion_list[0]

        total_loss = criterion_height(y_pred_height, y_train_height)
        # print(f"dim of x: {x.size()}, y: {y.size()}")
        # print(f"y: {y}, y_pred: {y_pred}, total_loss: {total_loss}")

    elif cfg.output_representation == 'xy':
        # Split y_pred and y_train into height, x, and y components
        y_pred_height, y_pred_x, y_pred_y = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        y_train_height, y_train_x, y_train_y = y_[:, 0], y_[:, 1], y_[:, 2]

        #unpack criterion_list and weight_list
        criterion_height, criterion_x, criterion_y = criterion_list
        weight_height, weight_x, weight_y = weight_list

        # Compute individual losses
        loss_height = criterion_height(y_pred_height, y_train_height)
        loss_x = criterion_x(y_pred_x, y_train_x)
        loss_y = criterion_y(y_pred_y, y_train_y)
        
        # print(f"y: {y}, y_pred: {y_pred}, loss_height: {loss_height}, loss_x: {loss_x}, loss_y: {loss_y}")
        total_loss = weight_height * loss_height + weight_x * loss_x + weight_y * loss_y

    elif cfg.output_representation == 'height_radianclass':

        # print(f"y_pred: {y_pred}, y_: {y_}")
        # Split y_pred and y_train into height, x, and y components
        y_pred_height, y_pred_class = y_pred
        y_train_height, y_train_class = y_[:, 0], y_[:, 1]

        #convert y_train_class to be int
        y_train_class = y_train_class.long()

        #unpack criterion_list and weight_list
        criterion_height, criterion_classifier = criterion_list
        weight_height, weight_classifier = weight_list

        # Compute individual losses
        loss_height = criterion_height(y_pred_height, y_train_height)
        loss_classifier = criterion_classifier(y_pred_class, y_train_class)
        
        print(f"y: {y}, y_pred: {y_pred}")
        print(f"y_pred_class: {y_pred_class}, y_train_class: {y_train_class}")
        total_loss = weight_height * loss_height + weight_classifier * loss_classifier
    return total_loss


def train_CNN(cfg,device, wandb, logger):
    """
    model = train_CNN(cfg)
    error = eval(cfg, model)
    """
    logger.log(" --------- training ---------")

    #TODO: MODIFY HERE TO USE THE RIGHT MODEL
    #choose the model for training (CNN 7layer, ResNet50, AST)
    # model = CNNRegressor2D(cfg)
    
    # model = ResNet50_audio(cfg)
    # model = AST(cfg)
    # model = AST_multimodal(cfg)
    # model = AST_multimodal_qt(cfg)

    # model = ResNet50_audio_proprioceptive(cfg)
    # model = ResNet50_audio_proprioceptive_dropout(cfg)

    # model = MultiModalTransformer(cfg)
    # model = MultiModalTransformer_xt_xdot_t(cfg)
    model = MultiModalTransformer_xt_xdot_t_gccphat(cfg)
    # model = MultiModalTransformer_xt_xdot_t_gccphat_tokens(cfg)
    # model = MultiModalTransformer_xt_xdot_t_toda(cfg)
    # model = MultiModalTransformer_xt_xdot_t_phase(cfg)


    model.to(device)
    logger.log(f"model: {model}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Number of parameters: {total_params}")


    #load data
    train_loader, val_loader = load_data(cfg, train_or_val='train')

    
    
    if cfg.output_representation == 'height_radianclass':
        model = CNNRegressor_Classifier(cfg)
        criterion_classifier = torch.nn.CrossEntropyLoss()

    #define loss and optimizer
    criterion_height = torch.nn.MSELoss() #--> L1 loss using mean absolute error
    criterion_rad = torch.nn.MSELoss()
    criterion_x = torch.nn.MSELoss()
    criterion_y = torch.nn.MSELoss()

    # Weights for each loss component (can be adjusted)
    weight_height = 1
    weight_rad = 5
    weight_x = 5
    weight_y = 5
    weight_classifier = 5

    if cfg.output_representation == 'xy':
        criterion_list = [criterion_height, criterion_x, criterion_y]
        weight_list = [weight_height, weight_x, weight_y]
    
    elif cfg.output_representation == 'rad':
        criterion_list = [criterion_height, criterion_rad]
        weight_list = [weight_height, weight_rad]

    elif cfg.output_representation == 'height':
        criterion_list = [criterion_height]
        weight_list = [weight_height]

    elif cfg.output_representation == 'height_radianclass':
        criterion_list = [criterion_height, criterion_classifier]
        weight_list = [weight_height, weight_classifier]

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Adjust step_size and gamma as needed

    

    train_loss_history = []
    val_loss_history = []

    best_val_loss = torch.inf
    # ---------------------------- epoch  ----------------------------
    for i in tqdm(range(cfg.train_epochs)):

        epoch_train_loss = 0
        epoch_val_loss = 0

        model.train()

        #---------------------------- train ----------------------------
        for _, (x, y, _, qt, xt, xdot_t, tdoa, gcc_matrix) in enumerate(train_loader):

            if cfg.visuaize_dataset:
                mic_utils.plot_spectrogram_of_all_data(cfg, x, 44100) # --> [batch_size, mic, freq, time]
                sys.exit()

            
            optimizer.zero_grad()

            # print(f"x shape in train: {x.shape}")
            train_loss = model_prediction(cfg, device, model, x, y, qt, xt, xdot_t, tdoa, gcc_matrix, criterion_list, weight_list)
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()
            
        epoch_train_loss = epoch_train_loss / len(train_loader)

        # Adjust learning rate
        scheduler.step()
        

        #---------------------------- val ----------------------------
        if i%cfg.eval_frequency == 0:

            model.eval()
            
            for _, (x, y, _, qt, xt, xdot_t, tdoa, gcc_matrix) in enumerate(tqdm(val_loader)):
                with torch.no_grad():

                    val_loss = model_prediction(cfg,device, model, x, y, qt, xt, xdot_t, tdoa, gcc_matrix, criterion_list, weight_list)
                    epoch_val_loss += val_loss.item()
                
            epoch_val_loss = epoch_val_loss / len(val_loader)

            logger.log(f"epoch: {i}, train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}, learning_rate: {scheduler.get_last_lr()[0]}")

            if cfg.log_wandb:
                wandb.log({'epoch': i, 'train_loss': epoch_train_loss})
                wandb.log({'epoch': i, 'val_loss': epoch_val_loss})
                wandb.log({'epoch': i, 'learning_rate': scheduler.get_last_lr()[0]})
            train_loss_history.append(epoch_train_loss)
            val_loss_history.append(epoch_val_loss)

            #save model
            if epoch_val_loss < best_val_loss:
                logger.log(f"Saving best model")
                best_val_loss = epoch_val_loss

                #save model to output directory
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, 'model.pth'))
                # wandb.save('best_model.pth')

                best_model = model

                #print location of saved model
                logger.log("Saved model : {}/model.pth".format(cfg.checkpoint_dir))

        #at last epoch, run eval_CNN to plot regression line
        if i == cfg.train_epochs-1:
            height_error, xy_error, degree_error, MED_mean, MED_std, all_distances = eval_utils_plot.predict_and_evaluate_val_dataset(cfg,best_model,device,val_loader)
            print(f"MED mean: {MED_mean}, MED std: {MED_std}")

    


    print(f" --------- training complete ---------")
    return best_model, train_loss_history, val_loss_history

def eval_random_prediction(cfg, device):
    """
    select random prediction from val_loader and evaluate
    """

    #load data
    train_loader, val_loader = load_data(cfg, train_or_val='val')

    error = 0
    y_val_list = []
    y_pred_list = []

    for _, (x, y, _, qt, xt, xdot_t, tdoa, gcc_matrix) in enumerate(tqdm(val_loader)):

        x_val, y_val = x.to(device), y.to(device)

        with torch.no_grad():
            # Get 20 random indices from y_val to assign to y_pred
            rand_idx = torch.randperm(y_val.size(0))[:20]
            y_pred = y_val[rand_idx]

            #use only first column element of y_pred and y_val
            y_pred = y_pred[:,0]
            y_val = y_val[:,0]

            # print(f"y_pred: {y_pred}, y_val: {y_val}")
            y_diff = y_pred - y_val
            print(f"y_diff: {y_diff}")

            #get absolute error
            error += torch.mean(torch.abs(y_diff))
        
        #get tensor values and append them to list
        y_val_list.extend(y_val.cpu().numpy())
        y_pred_list.extend(y_pred.cpu().numpy())
        
            
    #sum up the rmse and divide by number of batches
    print(f"len(val_loader): {len(val_loader)}")
    error = error / len(val_loader)
    print(f"MAE: {error}")

    #size of y_pred_list and y_val_list
    print(f"size of y_pred_list, y_val_list: {len(y_pred_list)}, {len(y_val_list)}")

    #plot regression line
    eval_utils_plot.plot_regression(y_pred_list, y_val_list)


    return error


def evaluate_CNN(cfg, model, device, val_loader):
    """
    evaluate without creating a new dataset 
    """
    model.eval()



# ==================================================================================================
def init_wandb(cfg):
    """
    Initialize wandb before each run
    """
    
    # start a new wandb run to track this script
    wandb.init(
        name = cfg.wandb_run_name,
        # set the wandb project where this run will be logged
        project=cfg.wandb_project,
        # track hyperparameters and run metadata
        config={
        "learning_rate": cfg.learning_rate,
        "architecture": "res50",
        "batch_size": cfg.batch_size,
        }
    )
# ==================================================================================================

    return wandb





@hydra.main(version_base='1.3',config_path='configs', config_name = 'train2D_UMC')
def main(cfg: DictConfig):

    if cfg.inspect_data_label:
        mic_utils.verify_dataset(cfg, cfg.data_dir)

    logger = Logger(log_wandb=cfg.log_wandb, simple_log = log, cfg=cfg)
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    #  Save the configuration to a file in the output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    config_path = os.path.join(output_dir, 'config.yaml')

    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    wandb = None
    if cfg.log_wandb:
        wandb = init_wandb(cfg)

    # ------------------------------------------

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    logger.log(f"device: {device}")
    logger.log(f"cfg: {cfg}")

    model, train_loss_history, val_loss_history = train_CNN(cfg,device, wandb, logger)


    
    # error = eval_random_prediction(cfg, device)
    # logger.log(f"Mean Absolute Error: {error}")

    # ------------------------------------------

    # model = train_KNN(cfg)
    # error = eval_KNN(cfg, model)
    # print(f" Mean Absolute Error: {error}")

if __name__ == '__main__':
    main()