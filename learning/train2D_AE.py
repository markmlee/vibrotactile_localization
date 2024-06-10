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
from models.AutoEncoder import  Encoder, Decoder

#eval
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from sklearn.metrics import accuracy_score
import eval_utils as eval_utils

#import function from another directory for plotting
sys.path.insert(0,'/home/mark/audio_learning_project/vibrotactile_localization/scripts')
import microphone_utils as mic_utils




torch.manual_seed(42)
np.random.seed(42)


    



def train_AE(cfg,device, wandb, logger):

    logger.log(" --------- training ---------")

    #load data
    train_loader, val_loader = load_data(cfg, train_or_val='train')

    #model
    encoder = Encoder(cfg)
    encoder.to(device)
    # pass the dummy input through the encoder 
    time_frame_in_spec = 345
    dummy_input = torch.randn(1, len(cfg.device_list), cfg.n_mels, time_frame_in_spec)

    # get the output (encoded representation)
    _ = encoder(dummy_input.to(device))

    shape_before_flattening = encoder.shape_before_flattening
    decoder = Decoder(cfg, shape_before_flattening)
    decoder.to(device)



    #define loss and optimizer
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.001
    )
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Adjust step_size and gamma as needed

    
    

    logger.log(f"encoder: {encoder}")
    logger.log(f"decoder: {decoder}")

    #get number of parameters
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.log(f"Number of parameters: {total_params}")

    train_loss_history = []
    val_loss_history = []

    best_val_loss = torch.inf
    # ---------------------------- epoch  ----------------------------
    for i in tqdm(range(cfg.train_epochs)):

        epoch_train_loss = 0
        epoch_val_loss = 0

        encoder.train()
        decoder.train()

        #---------------------------- train over batches ----------------------------
        for _, (x, y) in enumerate(train_loader):
            
            

            if cfg.visuaize_dataset:
                mic_utils.plot_spectrogram_of_all_data(cfg, x, 44100) # --> [batch_size, mic, freq, time]
                sys.exit()

            
            optimizer.zero_grad()

            x_ = x.float().to(device)
            encoded = encoder(x_)
            decoded = decoder(encoded)
            train_loss = criterion(decoded, x_)
   

            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()
            
        epoch_train_loss = epoch_train_loss / len(train_loader)

        # Adjust learning rate
        scheduler.step()
        

        #---------------------------- val ----------------------------
        if i%cfg.eval_frequency == 0:

            encoder.eval()
            decoder.eval()
            
            for _, (x, y) in enumerate(tqdm(val_loader)):
                with torch.no_grad():

                    x_ = x.float().to(device)
                    encoded = encoder(x_)
                    decoded = decoder(encoded)
                    val_loss = criterion(decoded, x_)
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
                torch.save(
                    {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, 
                    os.path.join(cfg.checkpoint_dir, 'model.pth')
                )
                # wandb.save('best_model.pth')

                #print location of saved model
                logger.log("Saved model : {}/model.pth".format(cfg.checkpoint_dir))

        
    


    print(f" --------- training complete ---------")

    #select 10 samples and plot reconsturction grid images
    num_samples = 1
    for i in range(num_samples):
        x, y = next(iter(train_loader))
        x_ = x.float().to(device)
        encoded = encoder(x_)
        decoded = decoder(encoded)

        #plot the original and reconstructed spectrogram
        print(f"shape of original : {x.shape}") #--> [batch_size, mic, freq, time]
        ith_sample = x[0,:,:,:].cpu().detach().numpy()
        mic_utils.plot_spectrogram_with_cfg(cfg, ith_sample, 44100)

        print(f"shape of decoded : {decoded.shape}") #--> [batch_size, mic, freq, time]
        ith_sample = decoded[0,:,:,:].cpu().detach().numpy()
        mic_utils.plot_spectrogram_with_cfg(cfg, ith_sample, 44100)

    #plot the latent space in 2D space to visualize similar data points
    points = []
    label_id = []
    for i,data in enumerate(train_loader):
        x, y = data
        x_ = x.float().to(device)
        encoded = encoder(x_)
        encoded = encoded.cpu().detach().numpy()
        print(f"shape of encoded : {encoded.shape}")
        points.append(encoded)
        y = y.cpu().detach().numpy()
        label_id.append(y)

        #TODO:create a scatter plot of the encoded data (where y is regression data with 2 variables [height, radian])
        # Flatten the list of arrays into a single numpy array
        points = np.concatenate(points, axis=0)
        label_id = np.concatenate(label_id, axis=0)

        # Separate the label data
        heights = label_id[:, 0]  # assuming first column is 'height'
        radians = label_id[:, 1]  # assuming second column is 'radian'

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot for heights
    sc1 = axs[0].scatter(points[:, 0], points[:, 1], c=heights, cmap='viridis', alpha=0.6)
    fig.colorbar(sc1, ax=axs[0])
    axs[0].set_title('Latent Space Colored by Height')
    axs[0].set_xlabel('Latent Variable 1')
    axs[0].set_ylabel('Latent Variable 2')

    # Scatter plot for radians
    sc2 = axs[1].scatter(points[:, 0], points[:, 1], c=radians, cmap='plasma', alpha=0.6)
    fig.colorbar(sc2, ax=axs[1])
    axs[1].set_title('Latent Space Colored by Radian')
    axs[1].set_xlabel('Latent Variable 1')
    axs[1].set_ylabel('Latent Variable 2')

    plt.tight_layout()
    plt.show()


        
    
    return train_loss_history, val_loss_history




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

@hydra.main(version_base='1.3',config_path='configs', config_name = 'AutoEncoder')
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

    train_loss_history, val_loss_history = train_AE(cfg,device, wandb, logger)



    

if __name__ == '__main__':
    main()