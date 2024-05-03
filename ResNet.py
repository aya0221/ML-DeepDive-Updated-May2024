''' AyaOshima - revised on May 3, 2024
- simplified ResNet architecture for image classification (into ten categories)
- constructed using PyTorch
- structures:
    - initial convolutional layer to process input images
    - two custom residual blocks featuring:
        - convolutional layers
        - batch normalization
        - ReLU activations
        - shortcut connections for identity mapping
    - average pooling
    - fully connected layer
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # initial convolution with stride to reduce dimension
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU()
        # pooling to reduce spatial dimensions
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # two simplified residual layers
        self.layer1 = self.construct_layer(64, 64, 1)
        self.layer2 = self.construct_layer(64, 128, 2)

        # adaptive pooling for fixed output size before classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # final dense layer for classification
        self.fc = nn.Linear(128, num_classes)


    def construct_layer(self, in_channels, out_channels, stride):
        # block of convolutions, batch norms and ReLU activations
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        # shortcut connection for identity mapping
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        return nn.Sequential(*layers), shortcut
    

    def apply_layer(self, x, layer, shortcut):
        # integrate main path and shortcut
        out = layer(x)
        shortcut_out = shortcut(x)
        out += shortcut_out
        return out


    def forward(self, x):
        # input through initial layer, then residual blocks, and to classification
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.initial_pool(x)

        x = self.apply_layer(x, *self.layer1)
        x = self.apply_layer(x, *self.layer2)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
model = ResNet(num_classes=10)
print(model)
