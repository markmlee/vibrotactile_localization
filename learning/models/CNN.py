import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRegressor(nn.Module):
    def __init__(self,cfg):
        super(CNNRegressor, self).__init__()
        
        input_size = len(cfg.device_list)
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
        output_size = 2 #[height, radian]

        if cfg.output_representation == 'rad':
            output_size = 2
        elif cfg.output_representation == 'xy':
            output_size = 3 #[height, x, y]

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
    