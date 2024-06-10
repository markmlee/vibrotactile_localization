import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNRegressor1D(nn.Module):
    def __init__(self,cfg):
        super(CNNRegressor1D, self).__init__()
        
        input_size = len(cfg.device_list)
        output_size = 1

        # simple test network
        self.conv1 = nn.Conv2d(input_size, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(11952, 120)
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
    

class CNNRegressor(nn.Module):
    def __init__(self,cfg):
        super(CNNRegressor, self).__init__()
        
        input_size = len(cfg.device_list)
        output_size = 2

        if cfg.output_representation == 'xy':
            output_size = 3 #[height, x, y]

        # simple test network
        self.conv1 = nn.Conv2d(input_size, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(11952, 120)
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
    

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

class CNNRegressor2D(nn.Module):
    def __init__(self,cfg):
        super(CNNRegressor2D, self).__init__()
        
        input_size = cfg.num_channel
        output_size = 2 #[height, radian]

        if cfg.output_representation == 'rad':
            output_size = 2
        elif cfg.output_representation == 'xy':
            output_size = 3 #[height, x, y]

        elif cfg.output_representation == 'height':
            output_size = 1

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(input_size,32,4,stride=1),
            conv2d_bn_relu(32,32,3),
            nn.MaxPool2d(2,2)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=1),
            conv2d_bn_relu(64,64,3),
            nn.MaxPool2d(2,2)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(64,128,4,stride=1),
            conv2d_bn_relu(128,128,3),
            nn.MaxPool2d(2,2)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(128,256,4,stride=1),
            conv2d_bn_relu(256,256,3),
            nn.MaxPool2d(2,2)
        )

        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(256,256,4,stride=1),
            conv2d_bn_relu(256,256,3)
        )


        self.fc_stack = torch.nn.Sequential(
            torch.nn.Linear(cfg.CNN_layer_size,512), #[torch.nn.Linear(4864,512) for 1.0 second input],[#torch.nn.Linear(1536,512) for 0.2 second input]
            torch.nn.ReLU(),
            torch.nn.Linear(512,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,output_size)
        )
    
    def forward(self, xb):
        # print(f"input xb shape: {xb.size()}")
        xb = self.conv_stack1(xb)
        # print(f"xb1 shape: {xb.size()}")
        xb = self.conv_stack2(xb)
        # print(f"xb2 shape: {xb.size()}")
        xb = self.conv_stack3(xb)
        # print(f"xb3 shape: {xb.size()}")
        xb = self.conv_stack4(xb)
        # print(f"xb4 shape: {xb.size()}")
        xb = self.conv_stack5(xb)
        # print(f"xb5 shape: {xb.size()}")
        # xb = self.adaptive_pool(xb)
        # print(f"xbpool shape: {xb.size()}")
        xb = torch.flatten(xb, 1)
        # print(f"xbflat shape: {xb.size()}")
        xb = self.fc_stack(xb)
        # print(f"output shape: {xb.size()}")
        return xb



class CNNRegressor_Classifier(nn.Module):
    def __init__(self, cfg):
        super(CNNRegressor_Classifier, self).__init__()
  
        input_size = len(cfg.device_list)
        output_size = 2 #[height, radian]


        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(input_size,32,4,stride=1),
            conv2d_bn_relu(32,32,3),
            nn.MaxPool2d(2,2)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=1),
            conv2d_bn_relu(64,64,3),
            nn.MaxPool2d(2,2)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(64,128,4,stride=1),
            conv2d_bn_relu(128,128,3),
            nn.MaxPool2d(2,2)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(128,256,4,stride=1),
            conv2d_bn_relu(256,256,3),
            nn.MaxPool2d(2,2)
        )

        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(256,256,4,stride=1),
            conv2d_bn_relu(256,256,3)
        )


        self.shared_fc_stack = torch.nn.Sequential(
            torch.nn.Linear(4864,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,256),
            torch.nn.ReLU(),
        )

        # Separate output layers for regression and classification
        self.fc_regression = nn.Linear(256, 1)  # Single output for height
        self.fc_classification = nn.Linear(256, 8)  # 9 classes for classification

    def forward(self, xb):
        # print(f"input xb shape: {xb.size()}")
        xb = self.conv_stack1(xb)
        # print(f"xb1 shape: {xb.size()}")
        xb = self.conv_stack2(xb)
        # print(f"xb2 shape: {xb.size()}")
        xb = self.conv_stack3(xb)
        # print(f"xb3 shape: {xb.size()}")
        xb = self.conv_stack4(xb)
        # print(f"xb4 shape: {xb.size()}")
        xb = self.conv_stack5(xb)

        xb = torch.flatten(xb, 1)
        xb = self.shared_fc_stack(xb)

        # Get outputs from both heads
        reg_output = self.fc_regression(xb)
        class_output = self.fc_classification(xb)

        return reg_output, class_output