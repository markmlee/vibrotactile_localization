import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRegressor(nn.Module):
    def __init__(self,cfg):
        super(CNNRegressor, self).__init__()
        
        input_size = len(cfg.device_list)

        if cfg.augment_audio:
            input_size = cfg.augment_num_channel
        output_size = 2
    
        # self.network = nn.Sequential(
            
        #     nn.Conv2d(input_size, 32, kernel_size = 3, padding = 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
        
        #     nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ReLU(),
        #     nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
            
        #     nn.Conv2d(128, 512, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ReLU(),
        #     nn.Conv2d(512,512, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
            
        #     nn.Flatten(),
        #     nn.Linear(110080,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512,output_size)
        # )

        # simple test network
        self.conv1 = nn.Conv2d(input_size, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(9296, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
        
    def forward(self, xb):
        # return self.network(xb)

        #simple test network
        xb = self.pool(F.relu(self.conv1(xb)))
        xb = self.pool(F.relu(self.conv2(xb)))
        xb = torch.flatten(xb, 1)
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        xb = self.fc3(xb)
        return xb
    


class CNNRegressor2D(nn.Module):
    def __init__(self,cfg):
        super(CNNRegressor2D, self).__init__()
        
        input_size = len(cfg.device_list)
        if cfg.augment_audio:
            input_size = cfg.augment_num_channel

        self.conv1 = nn.Conv2d(input_size, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # Removed conv4 and conv5 to prevent too much reduction in size
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # The number here (64 * x * y) needs to be calculated based on your input size and architecture
        self.fc1 = nn.Linear(64 * 215, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

        # Output layers
        # self.fc_height = nn.Linear(128, 1)
        # self.fc_radians = nn.Linear(128, 1)

    def forward(self, xb):
        # print(f"Input shape: {xb.shape}")
        xb = self.pool(F.relu(self.bn1(self.conv1(xb))))
        # print(f"Shape after conv1: {xb.shape}")
        xb = self.pool(F.relu(self.bn2(self.conv2(xb))))
        # print(f"Shape after conv2: {xb.shape}")
        xb = self.pool(F.relu(self.bn3(self.conv3(xb))))
        # print(f"Shape after conv3: {xb.shape}")


        xb = torch.flatten(xb, 1)
        xb = F.relu(self.fc1(xb))
        # print(f"Shape after fc1: {xb.shape}")
        xb = self.dropout(xb)
        xb = F.relu(self.fc2(xb))
        # print(f"Shape after fc2: {xb.shape}")
        xb = self.dropout(xb)
        xb = F.relu(self.fc3(xb))
        # print(f"Shape after fc3: {xb.shape}")

        # height = self.fc_height(xb)
        # radian = self.fc_radians(xb)

        #create 0s with same size as height
        # zero_height = torch.zeros_like(height)

        # return height, radian[:,0], radian[:,1]
        return xb