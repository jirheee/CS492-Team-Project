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
    # Instead of passing instance of NN, pass the NN model constructor.
    def __init__(self, nn_arch_constructor, board_size, lr=1e-3, gamma=0.99, eps=0.9, eps_decay=1.00007, eps_threshold=0.1, tau=0.01):
        super(DQNPlayer, self).__init__()
        random.seed()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)

        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_threshold = eps_threshold

        self.board_size = board_size

        self.main_network = nn_arch_constructor(*board_size).to(self.device)
        self.target_network = nn_arch_constructor(*board_size).to(self.device)
        update_model(self.main_network, self.target_network, tau=1.0)
        self.tau = tau
        self.step = 0
        self.update_step = 200

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)

        print("self.lr, self.gamma, self.eps, self.eps_decay, self.eps_threshold, self.board_size, self.tau, self.update_step")
        print(self.lr, self.gamma, self.eps, self.eps_decay, self.eps_threshold, self.board_size, self.tau, self.update_step)

        ########### DEBUGGING VARIABLES ###########
        
        ###########################################

    def set_player_ind(self, p):
        self.player = p
    
    def get_action(self, board, temp=1e-3, return_prob=0):       
        state = torch.from_numpy(board.current_state().copy()).unsqueeze(0).to(self.device)

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
        ########### Useful for Debugging ###########
        printable = lambda x: np.array2string(x.detach().clone().to('cpu').flatten().numpy(), precision = 2,suppress_small=True,separator=",")
        debug_view = lambda x: x.detach().clone().to('cpu').numpy()
        ############################################

        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.from_numpy(np.stack(states).copy()).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).reshape(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states).copy()).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).reshape(-1, 1).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float64).reshape(-1, 1).to(self.device)

        p1_moves = next_states[:,0,:,:]
        p2_moves = next_states[:,1,:,:]
        # set union of player 1's and player 2's moves
        # unavailable moves are marked as 1
        next_q_masks = (p1_moves + p2_moves).reshape((-1,self.board_size[0]*self.board_size[1]))
        current_q_masks = (states[:,0,:,:] + states[:,1,:,:]).reshape((-1,self.board_size[0]*self.board_size[1]))

        ### vvvvvvvvv Single Q update  vvvvvvvvv ###
        # - This method teaches value for value on the action that was actually smapled.
        # - Other actions won't be explicitly teached

        current_q_values = self.main_network(states)
        current_q_values = current_q_values.gather(1, actions)
        next_q_values = self.target_network(next_states)
        # - next_q_values is masked with -1 where moves is unavailable
        next_q_values = next_q_values * (1-next_q_masks) - next_q_masks
        # - Maximum values of next states are extracted. If a move was a last move,the value is set to 0.
        next_q_values = torch.max(next_q_values, dim=1)[0].unsqueeze(-1) * (1 - dones)
        target_q_values = rewards + self.gamma * next_q_values.detach().clone()

        ### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ###
        ### vvvvvvvvv Whole V update vvvvvvvvv ###
        # - This method teaches every action on board at states.
        # - Invalid actions are explicitly teached negatively, and actions that were not sampled will be taught the current value.

        # current_q_values = self.main_network(states)
        # next_q_values = torch.max(self.target_network(next_states) * (1-next_q_masks) - next_q_masks, dim=1)[0].unsqueeze(-1) * (1 - dones)
        # next_q_values = rewards + self.gamma * next_q_values.detach().clone()

        # # - For the "correct answer" 1. prepare current_q_values as template
        # target_q_values = current_q_values.detach().clone()
        # # - 2. Fill the action-indexed q of it as next_states' best
        # target_q_values[np.arange(len(mini_batch)), actions.squeeze()] = next_q_values.squeeze().detach().clone()
        # # - 3. Now punish invalid moves in current state
        # target_q_values = target_q_values.detach().clone() * (1-current_q_masks)

        ### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ###

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step % self.update_step:
            self.update_target()
        self.step += 1

        return loss.item()

    def update_target(self):
        update_model(self.main_network, self.target_network, tau=self.tau)

    def save_model(self, model_file):
        torch.save(self.state_dict(), model_file)
    
    def load_model(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def __str__(self):
        return "DQN {}".format(self.player)