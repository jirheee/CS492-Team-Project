import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from functools import reduce

def calculate_input_size(array):
    return reduce(lambda x, y: x * y, array)

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class NeuralNet(nn.Module):
    def __init__(self, board_width, board_height, nn_information):
        super(NeuralNet, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        # Consturct Neural Network
        self.layers = nn.ModuleList()
        prev_channels = 4
        input_sizes = [(4, self.board_width, self.board_height)]
        for i in range(nn_information['n_layers']):
            layer_information = nn_information.get(f"layer_{i}")
            if layer_information["layer_name"] == "Conv":
                channels = layer_information["channels"]
                kernel_size = layer_information["kernel_size"]
                stride = layer_information["stride"]
                padding = layer_information["padding"]
                self.layers.append(nn.Conv2d(prev_channels, channels,
                                             kernel_size=kernel_size, stride=stride, padding=padding))
                input_sizes.append((channels, (self.board_width - kernel_size + 1 + padding * 2) // stride,
                                              (self.board_height - kernel_size + 1 + padding * 2) // stride))
                prev_channels = channels
            else:
                NotImplementedError()
        
        if nn_information["activ_func"] == "ReLU":
            self.activ_func = nn.ReLU()
        elif nn_information["activ_func"] == "LeakyReLU":
            self.activ_func = nn.LeakyReLU()
        elif nn_information["activ_func"] == "Sigmoid":
            self.activ_func = nn.Sigmoid()
        elif nn_information["activ_func"] == "Tanh":
            self.activ_func = nn.Tanh()
        else:
            NotImplementedError()

        self.final_input_size = calculate_input_size(input_sizes[-1])
        
        # action policy layers
        self.act_fc1 = nn.Linear(self.final_input_size, 64)
        self.act_fc2 = nn.Linear(64, self.board_width * self.board_height)        
        
        # state value layers
        self.val_fc1 = nn.Linear(self.final_input_size, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activ_func(x)

        x_act = x.view(-1, self.final_input_size)
        x_act = F.relu(self.act_fc1(x_act))
        x_act = F.log_softmax(self.act_fc2(x_act), dim=-1)

        x_val = x.view(-1, self.final_input_size)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    def __init__(self, board_width, board_height, nn_information, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = NeuralNet(board_width, board_height, nn_information).cuda()
        else:
            self.policy_value_net = NeuralNet(board_width, board_height, nn_information)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)