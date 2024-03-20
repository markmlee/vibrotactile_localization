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

#logger
import logging
log = logging.getLogger(__name__)
from logger import Logger

#dataset
from datasets import AudioDataset
from datasets import load_data

#models
from models.CNN import CNNRegressor, CNNRegressor2D

#eval
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from datasets import plot_spectrogram

import matplotlib.pyplot as plt

def plot_regression(cfg, y_pred_list, y_val_list):
    #plot regression line for height, x, y 

    #given list of [[height, x,y]..] extract height from 1st element of list, x from 2nd element, y from 3rd element
    height = [item[0] for item in y_val_list]
    radian = [item[1] for item in y_val_list]
    
    height_pred = [item[0] for item in y_pred_list]
    radian_pred = [item[1] for item in y_pred_list]

    # Initialize layout
    fig, ax = plt.subplots(figsize = (9, 9))
    #add scatter plot for 1st element (height)
    ax.scatter(height, height_pred, alpha=0.7, edgecolors='k')
    #add regression line
    b,a, = np.polyfit(height, height_pred, deg=1)
    xseq = np.arange(0,len(height),1)
    ax.plot(xseq, a + b * xseq, color = 'r', lw=2.5)

    plt.title('Regression plot 1D line')
    plt.xlabel('height_val')
    plt.ylabel('height_pred')

    #set x and y axis limits
    plt.xlim(0, 25)
    plt.ylim(0, 25)

    #save the regression plot in output directory
    if cfg.visuaize_regression:
        plt.savefig(os.path.join(cfg.checkpoint_dir, 'height_plot.png'))
        plt.show()

    fig, ax = plt.subplots(figsize = (9, 9))
    #add scatter plot for 1st element (height)
    ax.scatter(radian, radian_pred, alpha=0.7, edgecolors='k')
    #add regression line
    b,a, = np.polyfit(radian, radian_pred, deg=1)
    xseq = np.arange(0,len(height),1)
    ax.plot(xseq, a + b * xseq, color = 'r', lw=2.5)

    plt.title('Regression plot radians')
    plt.xlabel('rad_val')
    plt.ylabel('rad_pred')

    #set x and y axis limits
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)

    #save the regression plot in output directory
    if cfg.visuaize_regression:
        plt.savefig(os.path.join(cfg.checkpoint_dir, 'radian_plot.png'))
        plt.show()

    #convert radian to x,y coordinates
    x_pred = np.cos(radian_pred)
    y_pred = np.sin(radian_pred)

    x = np.cos(radian)
    y = np.sin(radian)

    #add a new scatter plot for x,y 
    fig, ax = plt.subplots(figsize = (9, 9))

    ax.scatter(x_pred,y_pred, alpha=0.7, c='g' ,edgecolors='k')
    ax.scatter(x,y, alpha=0.7,c='r' ,edgecolors='k')

    plt.title('Regression plot for radians (x,y coordinates)')
    plt.xlabel('x')
    plt.ylabel('y')

    #add legend
    plt.legend(['y_pred', 'y_val'])

    #save the regression plot in output directory
    if cfg.visuaize_regression:
        plt.savefig(os.path.join(cfg.checkpoint_dir, 'xy_plot_radians.png'))
        plt.show()


def train_KNN(cfg):
    """
    model = train(cfg)
    error = eval(cfg, model)
    """
    print(f" --------- training ---------")

    #load data
    train_loader, val_loader = load_data(cfg, train_of_val='train')

    
    x_train, y_train = None, None

    #train KNN & SVM by looping once through trainloader. 
    #MAKE SURE BATCHSIZE IS SET TO LENGTH(DATASET)
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x_train, y_train = x, y

    print(f"shapes of x_train, y_train: {x_train.shape}, {y_train.shape}") #--> torch.Size([80, 6, 40, 690]), torch.Size([80, 2])

    #flatten x_train feature to be ([80, 6*40*690]) 
    x_train = x_train.view(x_train.size(0), -1)
    print(f"flattened x_train shape: {x_train.shape}") #--> torch.Size([80, 165600])
    #model
    model = KNN()
    print(f" fitting model ")
    model.fit(x_train, y_train)


    

    print(f" --------- training complete ---------")
    return model

def eval_KNN(cfg, model):
    """
    model = train(cfg)
    error = eval(cfg, model)
    """
    print(f" --------- evaluating ---------")

    #load data
    train_loader, val_loader = load_data(cfg, train_or_val='val')

    x_val, y_val = None, None

    #train KNN & SVM by looping once through trainloader. 
    #MAKE SURE BATCHSIZE IS SET TO LENGTH(DATASET)
    for i, (x, y) in enumerate(tqdm(val_loader)):
        x_val, y_val = x, y

    print(f"shapes of x_train, y_train: {x_val.shape}, {y_val.shape}") #--> torch.Size([80, 6, 40, 690]), torch.Size([80, 2])

    #flatten x_train feature to be ([80, 6*40*690]) 
    x_val = x_val.view(x_val.size(0), -1)
    print(f"flattened x_train shape: {x_val.shape}") #--> torch.Size([80, 165600])

    #get MSE of prediction
    mse,y_pred_list, y_val_list = model.mae(x_val, y_val)

    #plot regression line
    plot_regression(y_pred_list, y_val_list)


    print(f" --------- evaluation complete ---------")

    return mse


def compute_one_training_batch(model, x, y, device, criterion_height, criterion_radian):
    """
    To be performed inside the training loop
    in: model, x, y, criterion, optimizer
    out: loss
    """
    #weight for each loss
    weight_height = 1 #[0 to 20cm]
    weight_radian = 10 #[-2pi to 2pi]

    x_train, y_train = x.float().to(device), y.float().to(device)

    # print(f"shapes of x_train, y_train: {x_train.shape}, {y_train.shape}") #--> torch.Size([80, 6, 40, 690]), torch.Size([80, 2])
    height, radianx, radiany = model(x_train) # --> CNN separate-head output

    # print(f"height: {height}, radian: {radian}") # --> torch.Size([80, 1]), torch.Size([80, 1])
    # print(f"height: {height}, x: {radianx}, y:{radiany}") 
    # print(f"y_train[:,0] : {y_train[:,0]}, y_train[:,1]: {y_train[:,1]}, y_train[:,2]: {y_train[:,2]}") # --> torch.Size([80]), torch.Size([80])

    #reshape height and radian to be same shape as y_train
    height = height.view(-1)
    # radian = radian.view(-1)
    radianx = radianx.view(-1)
    radiany = radiany.view(-1)
    
    #separate loss for each output. L1 loss for height, MSE for radians
    train_loss_height = criterion_height(height, y_train[:,0])
    # train_loss_radian = criterion_radian(radian, y_train[:,1])
    train_loss_radian = criterion_radian(radianx, y_train[:,1]) + criterion_radian(radiany, y_train[:,2])

    train_loss = weight_height * train_loss_height + weight_radian * train_loss_radian

    return train_loss



def train_CNN(cfg,device, wandb, logger):
    """
    model = train_CNN(cfg)
    error = eval(cfg, model)
    """
    logger.log(" --------- training ---------")

    #load data
    train_loader, val_loader = load_data(cfg, train_or_val='train')

    
    #model
    model = CNNRegressor2D(cfg)

    #define separate loss for each output. L1 loss for height, MSE for radians
    criterion_height = torch.nn.MSELoss()
    criterion_radian = torch.nn.L1Loss() 

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # Adjust step_size and gamma as needed

    model.to(device)
    logger.log(f"model: {model}")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Number of parameters: {total_params}")

    train_loss_history = []
    val_loss_history = []

    best_val_loss = torch.inf

    for i in tqdm(range(cfg.train_epochs)):

        epoch_train_loss = 0
        epoch_val_loss = 0

        model.train()

        for _, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            train_loss = compute_one_training_batch(model, x, y, device, criterion_height, criterion_radian)
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item()
            
        epoch_train_loss = epoch_train_loss / len(train_loader)

        # Adjust learning rate
        scheduler.step()
        # print(f"Learning rate: {scheduler.get_last_lr()}")
        
        #log epoch loss
        if i%cfg.log_frequency == 0:
            logger.log({'epoch': i, 'train_loss': epoch_train_loss})
            


        #eval model 
        if i%cfg.eval_frequency == 0:

            model.eval()

            for _, (x, y) in enumerate(tqdm(val_loader)):

                with torch.no_grad():
                    val_loss = compute_one_training_batch(model, x, y, device, criterion_height, criterion_radian)
                    epoch_val_loss += val_loss.item()

                
            epoch_val_loss = epoch_val_loss / len(val_loader)

            logger.log(f"epoch: {i}, train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}")

            #print learning rate [0.01] --> 0.01
            logger.log(f"Learning rate: {scheduler.get_last_lr()[0]}")

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

        #at last epoch, run evaluate_CNN to plot regression line
        if i == cfg.train_epochs-1:
            error = evaluate_CNN(cfg,best_model,device,val_loader, logger)
            logger.log(f"Mean Absolute Error: {error}")

    

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

    for _, (x, y) in enumerate(tqdm(val_loader)):

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
    plot_regression(y_pred_list, y_val_list)


    return error


def evaluate_CNN(cfg, model, device, val_loader, logger):
    """
    evaluate without creating a new dataset 
    """
    model.eval()

    height_error, radian_error = 0,0
    y_val_list = []
    y_pred_list = []

    print(f" val_loader length: {len(val_loader)}")

    for _, (x, y) in enumerate(tqdm(val_loader)):

        x_val, y_val = x.float().to(device), y.float().to(device)

    
        with torch.no_grad():
            # height, radian = model(x_val) # --> CNN separate-head output
            height, radianx, radiany = model(x_val) # --> CNN separate-head output

            #reshape height and radian to be same shape as y_val
            height = height.view(-1)
            # radian = radian.view(-1)
            radianx = radianx.view(-1)
            radiany = radiany.view(-1)

            #unnormalize [0,1] to [0,20] and [-pi, pi]
            # height = height * 20.32
            # radian = radian * 2*np.pi - np.pi

            height_diff = height - y_val[:,0] #* (20.32)
            radianx_diff = radianx - y_val[:,1] #* (2*np.pi - np.pi)
            radiany_diff = radiany - y_val[:,2] #* (2*np.pi - np.pi)


            print(f"height_diff: {height_diff}")
            print(f"radianX_diff: {radianx_diff}")
            print(f"radianY_diff: {radiany_diff}")

            #print the dimensions of rad_y and rad_x
            print(f" height: {height} radianx: {radianx}, radiany: {radiany}")
            print(f" dim radianx: {radianx.shape}, dim radiany: {radiany.shape}")
            # sys.exit()


            #convert radianx and radiany to radian
            radian_predict = torch.atan2(radiany, radianx)
            radian_val = torch.atan2(y_val[:,2], y_val[:,1])

            radian_diff = radian_predict - radian_val

            #get absolute error
            height_error += torch.mean(torch.abs(height_diff)) 
            radian_error += torch.mean(torch.abs(radian_diff))

            #print dimension of height and radian
            # print(f"height: {height}, radian: {radian_predict}")
            # print(f"y_val[:,0] : {y_val[:,0]}, y_val[:,1]: {radian_val}")

            #combine height and radian to y_pred
            y_pred = torch.stack((height, radian_predict), dim=1)
            y_val = torch.stack((y_val[:,0], radian_val), dim=1)
        
        #get tensor values and append them to list
        y_val_list.extend(y_val.cpu().numpy())
        y_pred_list.extend(y_pred.cpu().numpy())
        
            
    #sum up the rmse and divide by number of batches
    height_error = (height_error) / len(val_loader)
    radian_error = (radian_error) / len(val_loader)

    error = height_error + radian_error

    logger.log(f"Height Error: {height_error}, Radian Error: {radian_error}")

    if cfg.visuaize_regression:
        #plot regression line
        plot_regression(cfg, y_pred_list, y_val_list)

        
        


    return error


        

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

    )
# ==================================================================================================

    return wandb

@hydra.main(version_base='1.3',config_path='configs', config_name = 'train2D')
def main(cfg: DictConfig):

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

    

    # ------------------------------------------

    # model = train_KNN(cfg)
    # error = eval_KNN(cfg, model)
    # print(f" Mean Absolute Error: {error}")

if __name__ == '__main__':
    main()