import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic Block for ResNet 18"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = torch.relu(out)
        return out

class ResNet18_audio(nn.Module):
    def __init__(self, cfg):
        super(ResNet18_audio, self).__init__()
        self.cfg = cfg
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(cfg.num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine number of output features
        if cfg.output_representation == 'xy':
            self.num_output = 3  # height, x, y
        elif cfg.output_representation == 'rad':
            self.num_output = 2  # height, radian
        elif cfg.output_representation == 'height':
            self.num_output = 1  # height only
        else:
            raise ValueError("Invalid output_representation in config")
        
        self.fc = nn.Linear(512, self.num_output)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling and final fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    

class Bottleneck(nn.Module):
    """Bottleneck Block for ResNet 50"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        # First 1x1 convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second 3x3 convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Third 1x1 convolution layer
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # Skip connection
        out = torch.relu(out)
        return out

class ResNet50_audio(nn.Module):
    def __init__(self, cfg):
        super(ResNet50_audio, self).__init__()
        self.cfg = cfg
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(cfg.num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # Global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine number of output features
        if cfg.output_representation == 'xy':
            self.num_output = 3  # height, x, y
        elif cfg.output_representation == 'rad':
            self.num_output = 2  # height, radian
        elif cfg.output_representation == 'height':
            self.num_output = 1  # height only
        else:
            raise ValueError("Invalid output_representation in config")
        
        self.fc = nn.Linear(2048, self.num_output)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling and final fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
class ResNet50_audio_proprioceptive(nn.Module):
    def __init__(self, cfg):
        super(ResNet50_audio_proprioceptive, self).__init__()
        self.cfg = cfg
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(cfg.num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP for joint trajectory
        self.joint_mlp = nn.Sequential(
            nn.Linear(100 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Determine number of output features
        if cfg.output_representation == 'xy':
            self.num_output = 3  # height, x, y
        elif cfg.output_representation == 'rad':
            self.num_output = 2  # height, radian
        elif cfg.output_representation == 'height':
            self.num_output = 1  # height only
        else:
            raise ValueError("Invalid output_representation in config")
        
        # Combined fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_output)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, qt):
        # Process audio input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Process joint trajectory
        qt = qt.view(qt.size(0), -1)  # Flatten
        qt_features = self.joint_mlp(qt)
        
        # Combine features
        combined = torch.cat((x, qt_features), dim=1)
        
        # Final prediction
        output = self.fc(combined)
        
        return output

class ResNet50_audio_proprioceptive_dropout(nn.Module):
    def __init__(self, cfg):
        super(ResNet50_audio_proprioceptive_dropout, self).__init__()
        self.cfg = cfg
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(cfg.num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP for joint trajectory
        self.joint_mlp = nn.Sequential(
            nn.Linear(100 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Determine number of output features
        if cfg.output_representation == 'xy':
            self.num_output = 3  # height, x, y
        elif cfg.output_representation == 'rad':
            self.num_output = 2  # height, radian
        elif cfg.output_representation == 'height':
            self.num_output = 1  # height only
        else:
            raise ValueError("Invalid output_representation in config")
        
        # Combined fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_output)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, qt):
        # Process audio input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Process joint trajectory
        qt = qt.view(qt.size(0), -1)  # Flatten
        qt_features = self.joint_mlp(qt)
        
        # Combine features
        combined = torch.cat((x, qt_features), dim=1)
        
        # Final prediction
        output = self.fc(combined)
        
        return output

class WeightedFeatureCombination(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_weight = nn.Parameter(torch.FloatTensor([0.5]))
        self.proprio_weight = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, audio_features, proprio_features):
        # Ensure weights are positive and sum to 1
        weights = torch.softmax(torch.stack([self.audio_weight, self.proprio_weight]), dim=0)
        
        # Apply weights
        weighted_audio = weights[0] * audio_features
        weighted_proprio = weights[1] * proprio_features
        
        # Combine features
        combined = torch.cat([weighted_audio, weighted_proprio], dim=1)
        return combined

# class ResNet50_audio_proprioceptive_dropout_weightedconcat(nn.Module):
#     def __init__(self, cfg):
#         super(ResNet50_audio_proprioceptive_dropout_weightedconcat, self).__init__()
#         self.cfg = cfg
        
#         # ResNet layers
#         self.conv1 = nn.Conv2d(cfg.num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.layer1 = self._make_layer(64, 64, 3)
#         self.layer2 = self._make_layer(256, 128, 4, stride=2)
#         self.layer3 = self._make_layer(512, 256, 6, stride=2)
#         self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # MLP for joint trajectory
#         self.joint_mlp = nn.Sequential(
#             nn.Linear(100 * 7, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         self.audio_fc = nn.Linear(2048, 1024)
#         self.proprio_fc = nn.Linear(128, 128)
#         self.feature_combiner = WeightedFeatureCombination()
        
#         # Determine number of output features
#         if cfg.output_representation == 'xy':
#             self.num_output = 3  # height, x, y
#         elif cfg.output_representation == 'rad':
#             self.num_output = 2  # height, radian
#         elif cfg.output_representation == 'height':
#             self.num_output = 1  # height only
#         else:
#             raise ValueError("Invalid output_representation in config")
        
#         # Final layer
#         self.fc = nn.Linear(1024+128, self.num_output)

#     def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
#         layers = []
#         layers.append(Bottleneck(in_channels, out_channels, stride))
#         for _ in range(1, num_blocks):
#             layers.append(Bottleneck(out_channels * 4, out_channels, stride=1))
#         return nn.Sequential(*layers)

#     def process_audio(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         return x

#     def forward(self, x, qt):
#         # Process audio
#         x = self.process_audio(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         audio_features = self.audio_fc(x)
        
#         # Process proprioceptive data
#         qt = qt.view(qt.size(0), -1)
#         proprio_features = self.joint_mlp(qt)
#         proprio_features = self.proprio_fc(proprio_features)
        
#         # Combine features with learned weights
#         combined = self.feature_combiner(audio_features, proprio_features)
        
#         # Final prediction
#         output = self.fc(combined)
        
#         return output