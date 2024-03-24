import torch
import torch.nn as nn
import torch.nn.functional as F
from UtilsLayer import AttentionLSTM, SELayer, BahdanauAttention

# Define the basic residual block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.99, eps=0.001):
        super(BasicBlock, self).__init__()

        self.feature = nn.Sequential(   

                                nn.Conv1d(in_channels, out_channels, kernel_size=8,  padding="same"),
                                nn.BatchNorm1d(out_channels, momentum=momentum, eps=eps),
                                nn.ReLU(inplace=True),

                                nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same"),
                                nn.BatchNorm1d(out_channels, momentum=momentum, eps=eps),
                                nn.ReLU(inplace=True),
                    
                                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same"),
                                nn.BatchNorm1d(out_channels, momentum=momentum, eps=eps),
                                nn.ReLU(inplace=True),
                                
                                     )

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = x

        out = self.feature(x)
        out = out + self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, input_size):
        super(ResNet1D, self).__init__()

        self.layer1 = BasicBlock(input_size, 64)
        self.layer2 = BasicBlock(64, 64)
        self.layer3 = BasicBlock(64, 128)
        self.layer4 = BasicBlock(128, 128)
        self.layer5 = BasicBlock(128, 256)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = torch.mean(x, 2)

        return x

