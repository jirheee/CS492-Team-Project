import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Conv(nn.Module):
    def __init__(self, board_width, board_height):
        super(Conv, self).__init__()
        torch.set_default_dtype(torch.float64)

        self.board_width = board_width
        self.board_height = board_height

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(2*self.board_width*self.board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, state):
        output = self.conv_layers(state)
        output = output.view(-1, 2*self.board_width*self.board_height)
        return output