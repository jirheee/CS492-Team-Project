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

        self.activ = nn.ReLU()
        self.skip = nn.Identity()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(4, 1, kernel_size=1)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.board_width*self.board_height, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.board_width*self.board_height),
            nn.Tanh()
        )
    
    def forward(self, state):
        state=state.to(self.device)
        out32 = self.activ(self.conv1(state))
        out64 = self.activ(self.conv2(out32))
        out128 = self.activ(self.conv3(out64))

        out64_ = self.activ(self.conv4(out128)) + self.skip(out64)
        out32_ = self.activ(self.conv5(out64_)) + self.skip(out32)
        out4_ = self.activ(self.conv6(out32_)) + self.skip(state)

        output = self.activ(self.conv7(out4_))
        output = output.view(-1, self.board_width*self.board_height).to(self.device)
        output = self.fc_layers(output)
        return output