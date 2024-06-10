import torch
import torch.nn as nn
import torch.nn.functional as F

# # Model parameters:
# LAYERS = 3
# KERNELS = [3, 3, 3]
# CHANNELS = [32, 64, 128] #[input channel size]
# STRIDES = [2, 2, 2]
# LINEAR_DIM = 39424


# class Encoder(nn.Module):
#     def __init__(self,cfg):
#         super(Encoder, self).__init__()
        
#         encoding_size = cfg.latent_dim
#         self.input_size = len(cfg.device_list)

#         # convolutional layer hyper parameters
#         self.layers = LAYERS
#         self.kernels = KERNELS
#         self.channels = CHANNELS
#         self.strides = STRIDES
#         self.conv = self.get_convs()

#         # layers for latent space projection
#         self.fc_dim = LINEAR_DIM
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(self.fc_dim, encoding_size)
        
#     def get_convs(self):
#         """
#         generating convolutional layers based on model's 
#         hyper parameters
#         """
#         conv_layers = nn.Sequential()
#         for i in range(self.layers):
#             # The input channel of the first layer is 1
#             if i == 0: conv_layers.append(nn.Conv2d(self.input_size, 
#                                             self.channels[i], 
#                                             kernel_size=self.kernels[i],
#                                             stride=self.strides[i],
#                                             padding=1))
            
#             else: conv_layers.append(nn.Conv2d(self.channels[i-1], 
#                                         self.channels[i],
#                                         kernel_size=self.kernels[i],
#                                         stride=self.strides[i],
#                                         padding=1))
                
#             # Here we use GELU as activation function
#             conv_layers.append(nn.GELU()) 
            
#         return conv_layers

#     def forward(self, x):
#         print(f"size of input: {x.shape}")
#         x = self.conv(x)
#         print(f"size after conv: {x.shape}")
#         x = self.flatten(x)
#         print(f"size after flatten: {x.shape}")
#         return self.linear(x)
    

# class Decoder(nn.Module):
#     def __init__(self, cfg):
#         super(Decoder, self).__init__()

#         self.fc_dim = LINEAR_DIM
#         self.input_dim = cfg.latent_dim
#         self.output_channels = len(cfg.device_list)

#         # Conv layer hypyer parameters
#         self.layers = LAYERS
#         self.kernels = KERNELS
#         self.channels = CHANNELS[::-1] # flip the channel dimensions
#         self.strides = STRIDES
        
#         # In decoder, we first do fc project, then conv layers
#         self.linear = nn.Linear(self.input_dim, self.fc_dim)
#         self.conv =  self.get_convs()

#         self.output = nn.Conv2d(self.channels[-1], self.output_channels, kernel_size=1, stride=1)

        
#     def get_convs(self):
#         conv_layers = nn.Sequential()
#         for i in range(self.layers):
            
#             if i == 0: conv_layers.append(
#                             nn.ConvTranspose2d(self.channels[i],
#                                                self.channels[i],
#                                                kernel_size=self.kernels[i],
#                                                stride=self.strides[i],
#                                                padding=1,
#                                                output_padding=1)
#                             )
            
#             else: conv_layers.append(
#                             nn.ConvTranspose2d(self.channels[i-1], 
#                                                self.channels[i],
#                                                kernel_size=self.kernels[i],
#                                                stride=self.strides[i],
#                                                padding=1,
#                                                output_padding=0
#                                               )
#                             )

#             conv_layers.append(nn.GELU())

#         return conv_layers
    
    
#     def forward(self, x):
#         print(f"decode size of input: {x.shape}")
#         x = self.linear(x)
#         print(f"size after fc: {x.shape}")
#         # reshape 3D tensor to 4D tensor
#         x = x.reshape(x.shape[0], 128, 7, 44)
#         print(f"size after reshape: {x.shape}")
#         x = self.conv(x)
#         print(f"size after conv: {x.shape}")
#         return self.output(x)
        
    


class Encoder(nn.Module):
    def __init__(self,cfg, latent_dim):
        super(Encoder, self).__init__()
        
        input_size = cfg.num_channel
        encoding_size = latent_dim
        

        #encoding block
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        #flatten layer
        self.shape_before_flattening = None
        self.flatten = torch.nn.Flatten()
        self.fc = nn.Linear(16128, encoding_size)
        # self.fc = nn.Linear(39424, encoding_size)


    def forward(self, xb):
        # print(f"size of input: {xb.shape}")
        xb = F.relu(self.conv1(xb))
        # print(f"size after conv1: {xb.shape}")
        xb = F.relu(self.conv2(xb))
        # print(f"size after conv2: {xb.shape}")
        xb = F.relu(self.conv3(xb))
        # print(f"size after conv3: {xb.shape}")

        self.shape_before_flattening = xb.shape[1:]
        # print(f"shape before flattening: {self.shape_before_flattening}")

        xb = xb.view(xb.size(0), -1)
        xb = self.fc(xb)
        return xb
    

class Decoder(nn.Module):
    def __init__(self, cfg, shape_before_flattening, latent_dim):
        super(Decoder, self).__init__()

        # FC layer to unflatten the embeddings
        self.reshape_dim = shape_before_flattening
        encoding_size = latent_dim
        self.fc = nn.Linear(encoding_size, 16128)
        # self.fc = nn.Linear(encoding_size, 39424)

        # Decoding block (convolutional layers in reverse order)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Output layer
        self.conv1 = nn.Conv2d(32, cfg.num_channel, kernel_size=3, stride=1, padding=1)

        
        
    def forward(self, xb):
        # print(f"decode size of input: {xb.shape}")
        xb = self.fc(xb)
        # print(f"size after fc: {xb.shape}")
        # reshape the tensor to match shape before flattening
        xb = xb.view(xb.size(0), *self.reshape_dim)
        # print(f"size after reshape: {xb.shape}")

        xb = F.relu(self.deconv3(xb))
        # print(f"size after deconv3: {xb.shape}")
        xb = F.relu(self.deconv2(xb))
        # print(f"size after deconv2: {xb.shape}")
        xb = F.relu(self.deconv1(xb))
        # print(f"size after deconv1: {xb.shape}")
        xb = self.conv1(xb)
        # print(f"size after conv1: {xb.shape}")

        #crop the last column b/c 1 extra
        # xb = xb[:,:,:,:-1]


        return xb