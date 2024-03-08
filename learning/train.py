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
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from datasets import plot_spectrogram



def plot_regression(y_pred_list, y_val_list):
    #plot regression line
    import matplotlib.pyplot as plt
    # Initialize layout
    fig, ax = plt.subplots(figsize = (9, 9))
    #add scatter plot
    ax.scatter(y_val_list, y_pred_list, alpha=0.7, edgecolors='k')
    #add regression line
    b,a, = np.polyfit(y_val_list, y_pred_list, deg=1)
    xseq = np.arange(0,len(y_val_list),1)
    ax.plot(xseq, a + b * xseq, color = 'r', lw=2.5)

    plt.title('Regression plot')
    plt.xlabel('y_val')
    plt.ylabel('y_pred')
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

def train_CNN(cfg,device, wandb, logger):
    """
    model = train_CNN(cfg)
    error = eval(cfg, model)
    """
    logger.log(" --------- training ---------")

    #load data
    train_loader, val_loader = load_data(cfg, train_or_val='train')

    
    #model
    model = CNNRegressor(cfg)

    #define loss and optimizer
    criterion = torch.nn.L1Loss() #--> L1 loss using mean absolute error
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
            x_train, y_train = x.to(device), y.to(device)

 

            # print(f"shapes of x_train, y_train: {x_train.shape}, {y_train.shape}") #--> torch.Size([80, 6, 40, 690]), torch.Size([80, 2])

            optimizer.zero_grad()
            y_pred = model(x_train)

            # print(f"y values: {y_train}, y_pred: {y_pred}")

            train_loss = criterion(y_pred, y_train)
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
                x_val, y_val = x, y
                
                x_val, y_val = x_val.to(device), y_val.to(device)
                with torch.no_grad():
                    y_pred = model(x_val)
                    val_loss = criterion(y_pred, y_val)
                    epoch_val_loss += val_loss.item()
                
            epoch_val_loss = epoch_val_loss / len(val_loader)

            logger.log(f"epoch: {i}, train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}")

            if cfg.log_wandb:
                wandb.log({'epoch': i, 'train_loss': epoch_train_loss})
                wandb.log({'epoch': i, 'val_loss': epoch_val_loss})
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

    error = 0
    y_val_list = []
    y_pred_list = []

    for _, (x, y) in enumerate(tqdm(val_loader)):

        x_val, y_val = x.to(device), y.to(device)

    
        with torch.no_grad():
            y_pred = model(x_val)

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
    error = error / len(val_loader)

    logger.log(f"Mean Absolute Error: {error}")

    if cfg.visuaize_regression:
        #plot regression line
        plot_regression(y_pred_list, y_val_list)


    return error


def eval_CNN(cfg, model,device, logger):
    """
    error = eval_CNN(cfg, model)
    plot y_pred vs y_val in a regression plot
    """
    logger.log(" --------- evaluating ---------")
    #load data
    train_loader, val_loader = load_data(cfg, train_or_val='val')

    model.eval()

    error = 0
    y_val_list = []
    y_pred_list = []

    for _, (x, y) in enumerate(tqdm(val_loader)):

        x_val, y_val = x.to(device), y.to(device)

        #convert tensor to numpy array and plot spectrogram
        x_val = x_val.cpu().numpy()



        with torch.no_grad():
            y_pred = model(x_val)

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
    error = error / len(val_loader)

    logger.log(f"Mean Absolute Error: {error}")

    if cfg.visuaize_regression:
        #plot regression line
        plot_regression(y_pred_list, y_val_list)


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
        config={
        "learning_rate": 0.001,
        "architecture": "Convblock3x",
        "batch_size": cfg.batch_size,
        }
    )
# ==================================================================================================

    return wandb

@hydra.main(version_base='1.3',config_path='configs', config_name = 'train')
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

    
    # error = eval_random_prediction(cfg, device)
    # error = eval_CNN(cfg, model,device,logger)
    # logger.log(f"Mean Absolute Error: {error}")

    # ------------------------------------------

    # model = train_KNN(cfg)
    # error = eval_KNN(cfg, model)
    # print(f" Mean Absolute Error: {error}")

if __name__ == '__main__':
    main()