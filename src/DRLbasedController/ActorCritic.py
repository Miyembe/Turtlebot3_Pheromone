import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hid_list):
        super(Actor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_list = hid_list
        self.n_hid_layers = len(hid_list)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # === Create Layers === #
        self.layers = nn.ModuleList()
        for i in range(self.n_hid_layers+1):
            if i == 0:
                self.layers.append(nn.Linear(self.in_dim, self.hid_list[i]))
            elif i == self.n_layers:
                for j in range(self.out_dim):
                    self.layers.append(nn.Linear(self.hid_list[i-1], 1))
                    self.layers.append(nn.Linear(self.hid_list[i-1], 1))
            else: self.layers.append(nn.Linear(self.hid_list[i-1], self.hid_list[i]))

    def forward(self, x):
        for i in range(self.n_hid_layers+1):
            if i == self.n_hid_layers:
                x1 = self.layers[i](x)
                out1 = self.tanh(x1) # angular velocity
                x2 = self.layers[i+1](x)
                out2 = self.sigmoid(x2) # linear velocity
            else:
                x = self.layers[i](x)
                x = self.relu(x)
        
        return out1, out2

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, out_dim, hid_list):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = out_dim
        self.hid_list = hid_list
        self.n_hid_layers = len(hid_list)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # === Create Layers === #
        self.layers = nn.ModuleList()
        for i in range(self.n_hid_layers+1):
            if i == 0:
                self.layers.append(nn.Linear(self.obs_dim, self.hid_list[i]))
            elif i == 1:
                self.layers.append(nn.Linear(self.hid_list[i-1]+self.act_dim, self.hid_list[i]))
            elif i == self.n_layers:
                self.layers.append(nn.Linear(self.hid_list[i-1], self.out_dim))
            else: self.layers.append(nn.Linear(self.hid_list[i-1], self.hid_list[i]))

    def forward(self, xs):
        x, a = xs
        x = self.layers[0](x)
        x = self.relu(x)
        x = self.layers[1](torch.cat([x,a]),1)
        x = self.relu(x)

        for i in range(self.n_hid_layers+1):
            if i == 1:
                x = self.layers[i](torch.cat([x,a]),1)
                x = self.relu(x)
            elif i == self.n_hid_layers:
                x = self.layers[i](x)
            else:
                x = self.layers[i](x)
                x = self.relu(x)
        
        return x