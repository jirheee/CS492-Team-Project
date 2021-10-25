import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Conv(nn.Module):
    def __init__(self, board_width, board_height):
        super(Conv, self).__init__()
        torch.set_default_dtype(torch.float64)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.board_width = board_width
        self.board_height = board_height

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(self.board_width*self.board_height, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.board_width*self.board_height),
            nn.Sigmoid(),
            #nn.Softmax(dim=1)
        )
    
    def forward(self, state):
        output = self.conv_layers(state).to(self.device)
        output = output.view(-1, self.board_width*self.board_height).to(self.device)
        output = self.fc_layers(output)
        return output