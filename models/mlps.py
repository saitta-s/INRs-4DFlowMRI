# Code adapted from https://github.com/vsitzmann/siren.git

import torch
from torch import nn
import numpy as np
import math

from models.activations import *


class MLP(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        weight_init = None
        in_features, out_features = cfg.in_features, cfg.out_features
        hidden_features = cfg.hidden_features
        num_hidden_layers = cfg.num_hidden_layers
        outermost_linear = True
        if hasattr(cfg, 'hidden_omega_0'):
            hidden_omega_0 = cfg.hidden_omega_0
        else:
            hidden_omega_0 = 30.0

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(hidden_omega_0), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'swish': (Swish(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[cfg.nonlinearity]

        if weight_init is not None:
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nl))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)

    def forward(self, coords):
        return self.net(coords)


