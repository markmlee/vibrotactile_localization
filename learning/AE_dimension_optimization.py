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


    
def train_AE(cfg,device, wandb, logger, latent_dim):

    logger.log(" --------- training with latent_dim {latent_dim} ---------")

    #load data
    train_loader, val_loader = load_data(cfg, train_or_val='train')

    #model
    encoder = Encoder(cfg, latent_dim)
    encoder.to(device)
    # pass the dummy input through the encoder 
    time_frame_in_spec = 138 #this is hardcoded for AE param 39424 
    dummy_input = torch.randn(1, cfg.num_channel, cfg.n_mels, time_frame_in_spec)

    # get the output (encoded representation)
    _ = encoder(dummy_input.to(device))

    shape_before_flattening = encoder.shape_before_flattening
    decoder = Decoder(cfg, shape_before_flattening, latent_dim)
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

            # print(f"shape of decoded : {decoded.shape}, and x_: {x_.shape}") #--> [batch_size, mic, freq, time]
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
        save_name = f"original_{latent_dim}"
        mic_utils.save_spectrogram_with_cfg(cfg, ith_sample, 44100, save_name)

        print(f"shape of decoded : {decoded.shape}") #--> [batch_size, mic, freq, time]
        ith_sample = decoded[0,:,:,:].cpu().detach().numpy()
        mic_utils.save_spectrogram_with_cfg(cfg, ith_sample, 44100, latent_dim)

        

        
    
    return train_loss_history, val_loss_history, best_val_loss



@hydra.main(version_base='1.3',config_path='configs', config_name = 'AE_optimization')
def main(cfg: DictConfig):

    logger = Logger(log_wandb=cfg.log_wandb, simple_log = log, cfg=cfg)
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    #  Save the configuration to a file in the output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    config_path = os.path.join(output_dir, 'config.yaml')

    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # ------------------------------------------

    
    latent_dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # latent_dims = [2]

    reconstruction_errors = []

    for latent_dim in latent_dims:

        train_loss_history, val_loss_history, best_val_loss = train_AE(cfg,device, wandb, logger, latent_dim)

        #get the min validation loss
        loss = best_val_loss

        reconstruction_errors.append(loss)
        print(f'Latent Dimension: {latent_dim}, Reconstruction Error: {loss}')
    

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims, reconstruction_errors, marker='o')
    plt.title('Reconstruction Error vs Latent Dimension')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Reconstruction Error')
    plt.grid(True)
    
    #save plot
    plt.savefig(os.path.join(output_dir, 'reconstruction_error_vs_latent_dim.png'))
    
    plt.show()

    

    

if __name__ == '__main__':
    main()