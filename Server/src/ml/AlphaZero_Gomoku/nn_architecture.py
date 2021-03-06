import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torch_geometric.nn.conv.message_passing as gnn_parent
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, ChebConv, GINConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import numpy as np
from functools import reduce

import os
from mcts_alphaZero import MCTSPlayer
import json

def calculate_input_size(array):
    return reduce(lambda x, y: x * y, array)

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class NeuralNet(nn.Module):
    def __init__(self, board_width, board_height, nn_type, nn_layers):
        super(NeuralNet, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        # Consturct Neural Network
        self.layers = nn.ModuleList()
        self.nn_type = nn_type
        if self.nn_type == "CNN":
            prev_channels = 4
            prev_width = self.board_width
            prev_height = self.board_height
            input_sizes = [(prev_channels, prev_width, prev_height)]
            for layer in nn_layers:
                layer_information = layer
                layer_type = layer_information["layer_name"]
                if layer_type == "Conv":
                    channels = int(layer_information["channels"])
                    kernel_size = int(layer_information["kernel_size"])
                    stride = int(layer_information["stride"])
                    padding = int(layer_information["padding"])
                    bias = True if layer_information['bias'] == "True" else False
                    self.layers.append(nn.Conv2d(prev_channels, channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
                    width = (prev_width - kernel_size + 1 + padding * 2) // stride
                    height = (prev_height - kernel_size + 1 + padding * 2) // stride
                    input_sizes.append((channels, width, height))
                    prev_channels = channels
                    prev_width = width
                    prev_height = height
                elif layer_type == "BatchNorm":
                    self.layers.append(nn.BatchNorm2d(input_sizes[-1][0]))
                elif layer_type == "ReLU":
                    self.layers.append(nn.ReLU())
                elif layer_type == "LeakyReLU":
                    self.layers.append(nn.LeakyReLU())
                elif layer_type == "Sigmoid":
                    self.layers.append(nn.Sigmoid())
                elif layer_type == "Tanh":
                    self.layers.append(nn.Tanh())
                else:
                    NotImplementedError()
            self.final_input_size = calculate_input_size(input_sizes[-1])
        elif self.nn_type == "GNN":
            prev_channels = 4
            for layer in nn_layers:
                layer_information = layer
                layer_type = layer_information["layer_name"]
                if layer_type == "GCNConv":
                    channels = int(layer_information["channels"])
                    bias = True if layer_information['bias'] == "True" else False
                    self.layers.append(GCNConv(prev_channels, channels, bias=bias))
                    prev_channels = channels
                elif layer_type == "ChebConv":
                    channels = int(layer_information["channels"])
                    bias = True if layer_information['bias'] == "True" else False
                    self.layers.append(ChebConv(prev_channels, channels, bias=bias))
                    prev_channels = channels
                elif layer_type == "SAGEConv":
                    channels = int(layer_information["channels"])
                    bias = True if layer_information['bias'] == "True" else False
                    self.layers.append(SAGEConv(prev_channels, channels, bias=bias))
                    prev_channels = channels
                elif layer_type == "GATConv":
                    channels = int(layer_information["channels"])
                    bias = True if layer_information['bias'] == "True" else False
                    self.layers.append(GATConv(prev_channels, channels, bias=bias))
                    prev_channels = channels
                elif layer_type == "GINConv":
                    channels = int(layer_information["channels"])
                    bias = True if layer_information['bias'] == "True" else False
                    self.layers.append(GINConv(prev_channels, channels, bias=bias))
                    prev_channels = channels
                elif layer_type == "SGConv":
                    channels = int(layer_information["channels"])
                    bias = True if layer_information['bias'] == "True" else False
                    self.layers.append(SGConv(prev_channels, channels, bias=bias))
                    prev_channels = channels      
                elif layer_type == "ReLU":
                    self.layers.append(nn.ReLU())
                elif layer_type == "LeakyReLU":
                    self.layers.append(nn.LeakyReLU())
                elif layer_type == "Sigmoid":
                    self.layers.append(nn.Sigmoid())
                elif layer_type == "Tanh":
                    self.layers.append(nn.Tanh())
                else:
                    NotImplementedError()
            self.final_input_size = channels * self.board_width * self.board_height
        else:
            NotImplementedError()
        
        # action policy layers
        self.act_fc1 = nn.Linear(self.final_input_size, 64)
        self.act_fc2 = nn.Linear(64, self.board_width * self.board_height)        
        
        # state value layers
        self.val_fc1 = nn.Linear(self.final_input_size, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x, edge_index=None, batch=None):
        if self.nn_type == "CNN":
            for layer in self.layers:
                x = layer(x)
        else:
            for layer in self.layers:
                if isinstance(layer, gnn_parent.MessagePassing):
                    x = layer(x, edge_index)
                else:
                    x=layer(x) 
                    

        x_act = x.view(-1, self.final_input_size)
        x_act = F.relu(self.act_fc1(x_act))
        x_act = F.log_softmax(self.act_fc2(x_act), dim=-1)

        x_val = x.view(-1, self.final_input_size)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    def __init__(self, board_width, board_height, nn_type, nn_layers, model_file=None, force_cpu = False):
        self.device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        
        # the policy value net module
        self.policy_value_net = NeuralNet(board_width, board_height, nn_type, nn_layers).to(self.device)
        self.nn_type = self.policy_value_net.nn_type
        if self.nn_type == "GNN":
            self.edge_index = self.get_index()

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file,map_location=torch.device(self.device))
            self.policy_value_net.load_state_dict(net_params)

    def get_index(self):
        edge_index = [[], []]
        for i in range(self.board_width * self.board_height):
            if i % self.board_width == 0:
                if i // self.board_height == 0:
                    edge_index[0] += [i, i, i+1, i+self.board_width]
                    edge_index[1] += [i+1, i+self.board_width, i, i]
                elif i // self.board_height == self.board_height - 1:
                    edge_index[0] += [i, i, i+1, i-self.board_width]
                    edge_index[1] += [i+1, i-self.board_width, i, i]
                else:
                    edge_index[0] += [i, i, i, i-self.board_width, i+1, i+self.board_width]
                    edge_index[1] += [i-self.board_width, i+1, i+self.board_width, i, i, i]
            elif i % self.board_width == self.board_width-1:
                if i // self.board_height == 0:
                    edge_index[0] += [i, i, i-1, i+self.board_width]
                    edge_index[1] += [i-1, i+self.board_width, i, i]
                elif i // self.board_height == self.board_height - 1:
                    edge_index[0] += [i, i, i-1, i-self.board_width]
                    edge_index[1] += [i-1, i-self.board_width, i, i]
                else:
                    edge_index[0] += [i, i, i, i-self.board_width, i-1, i+self.board_width]
                    edge_index[1] += [i-self.board_width, i-1, i+self.board_width, i, i, i]
            elif i // self.board_height == 0:
                edge_index[0] += [i, i, i, i-1, 1+self.board_width, i+1]
                edge_index[1] += [i-1, 1+self.board_width, i+1, i, i, i]
            elif i // self.board_height == self.board_height - 1:
                edge_index[0] += [i, i, i, i-1, i-self.board_width, i+1]
                edge_index[1] += [i-1, i-self.board_width, i+1, i, i, i]
            else:
                edge_index[0] += [i, i, i, i, i-1, i-self.board_width, i+1, i+self.board_width]
                edge_index[1] += [i-1, i-self.board_width, i+1, i+self.board_width, i, i, i, i]
        edge_index = torch.LongTensor(edge_index)
        return edge_index

    def policy_value(self, state_batch):
        if self.nn_type == "CNN":
            state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
            log_act_probs, value = self.policy_value_net(state_batch)
        else: # GNN
            state_loader = DataLoader(dataset=[Data(x=Variable(torch.FloatTensor(state_batch[i].reshape(4, self.board_width*self.board_height).T).to(self.device)),
                    edge_index=self.edge_index) for i in range(len(state_batch))], batch_size=len(state_batch))
            state = next(iter(state_loader)).to(self.device)
            log_act_probs, value = self.policy_value_net(state.x, state.edge_index, state.batch)

        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        
        if self.nn_type == "CNN":
            current_state = np.ascontiguousarray(current_state.reshape(
                    -1, 4, self.board_width, self.board_height))

            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).to(self.device).float())
        else: # GNN
            current_state = np.ascontiguousarray(current_state.reshape(4, 
                    self.board_width*self.board_height).T)

            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).to(self.device).float(), 
                    self.edge_index.to(self.device), torch.LongTensor(np.array([0 for _ in range(self.board_width*self.board_height)])))
        
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        # wrap in Variable
        if self.nn_type == "CNN":
            state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
            log_act_probs, value = self.policy_value_net(state_batch)
        else: # GNN
            state_loader = DataLoader(dataset=[Data(x=Variable(torch.FloatTensor(state_batch[i].reshape(4, self.board_width*self.board_height).T).to(self.device)),
                    edge_index=self.edge_index) for i in range(len(state_batch))], batch_size=len(state_batch))
            state = next(iter(state_loader)).to(self.device)
            log_act_probs, value = self.policy_value_net(state.x, state.edge_index, state.batch)

        mcts_probs = Variable(torch.FloatTensor(mcts_probs).to(self.device))
        winner_batch = Variable(torch.FloatTensor(winner_batch).to(self.device))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

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

    def num_params(self):
        # Number of Trainable Parameters
        return sum(p.numel() for p in self.policy_value_net.parameters() if p.requires_grad)

def get_PVN_from_uuid(uuid:str,model_option:str="best",force_cpu=False):
    
    model_config = f"../models/{uuid}/model.json"
    with open(model_config, encoding='utf-8') as f:
        model_config = json.loads(f.read())
    # params of the board and the game
    board_width = model_config["board"]["board_width"]
    board_height = model_config["board"]["board_height"]
    name = model_config["name"]

    model_file_path = f"../models/{uuid}/{model_option}.model"
    if os.path.exists(model_file_path):
        print(f"Loading checkpoint from: {uuid}, {model_option}",flush=True)
    else:
        print(f"{model_option}_model from uuid of {uuid} could not be found",flush=True)
        raise

    policy_value_net = PolicyValueNet(board_width, board_height, model_config["nn_type"], model_config["layers"], model_file = model_file_path,force_cpu = force_cpu)
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                    c_puct=5,
                                    n_playout=board_width*board_height*17,name=name)
    return mcts_player