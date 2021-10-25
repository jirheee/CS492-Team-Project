# Implementation of DQN Algorithm
import os
import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def update_model(source, target, tau) :
    for source_param, target_param in zip(source.parameters(), target.parameters()) :
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

class DQNPlayer(nn.Module):
    def __init__(self, nn_architecture, lr=1e-3, gamma=0.99, eps=0.9, eps_decay=1e-3, eps_threshold=0.1, tau=0.01):
        super(DQNPlayer, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)

        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_threshold = eps_threshold

        self.main_network = nn_architecture.to(self.device)
        self.target_network = nn_architecture.to(self.device)
        update_model(self.main_network, self.target_network, tau=1.0)
        self.tau = tau
        self.step = 0
        self.update_step = 200

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)

    def set_player_ind(self, p):
        self.player = p
    
    def get_action(self, board, temp=1e-3, return_prob=0):       
        state = torch.from_numpy(board.current_state().copy()).unsqueeze(0).to(self.device)
        
        if self.eps > self.eps_threshold:
            self.eps *= self.eps_decay
        else:
            self.eps = self.eps_threshold

        sensible_moves = board.availables
        move_probs = torch.flatten(self.main_network(state)).to(self.device)
        trial = 0
        if len(sensible_moves) > 0:
            if random.random() < self.eps:
                move = random.sample(sensible_moves, 1)[0]
            else:
                move = torch.argmax(move_probs).item()
                moves_sorted = torch.argsort(move_probs, descending=True)
                while move not in sensible_moves: 
                    trial += 1
                    move = moves_sorted[trial].item()
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def train(self, mini_batch):
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.from_numpy(np.stack(states).copy()).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).reshape(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states).copy()).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).reshape(-1, 1).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float64).reshape(-1, 1).to(self.device)

        current_q_values = self.main_network(states).gather(1, actions)
        next_q_values = torch.max(self.target_network(next_states), dim=1)[0].reshape(-1, 1) * (1 - dones)
        target_q_values = rewards + self.gamma * next_q_values.detach()

        printable = lambda x: np.array2string(x.to('cpu').flatten().detach().clone().numpy(),precision = 2)
        mse_loss = self.criterion(target_q_values, current_q_values)
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

        if self.step % self.update_step:
            self.update_target()
        self.step += 1

        return mse_loss.item()

    def update_target(self):
        update_model(self.main_network, self.target_network, tau=self.tau)

    def save_model(self, model_file):
        torch.save(self.state_dict(), model_file)

    def __str__(self):
        return "DQN {}".format(self.player)