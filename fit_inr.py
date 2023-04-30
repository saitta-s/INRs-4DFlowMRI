import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from glob import glob
import time
import h5py
import torch
import json
import math
from torch import nn
from torch.nn import functional as F
from torch import from_numpy as fnp
from torch.autograd import grad
import random
import logging
import argparse

from models.mlps import MLP


##------------------------ Configuration
parser = argparse.ArgumentParser()

# Experiment options
parser.add_argument('--file_h5', type=str, default=None, help='preprocessed .h5 file path')
parser.add_argument('--wall_batch_frac', type=int, default=1, help='ratio between Nw and Nf (see paper eq. 6.7) sampled at each iteration')
parser.add_argument('--max_steps', type=int, default=1000, help='max # of optimization steps')
parser.add_argument('--early_stop_tol', type=float, default=0., help='tolerance for early stopping')
parser.add_argument('--out_dir', type=str, default='./results', help='output directory path')
parser.add_argument('--gpus', default=[0], nargs='+', help='device ids')

# Model options
parser.add_argument('--nonlinearity', type=str, default="sine", help='type of activation function')
parser.add_argument('--in_features', type=int, default=4, help='input coordinate dimension')
parser.add_argument('--out_features', type=int, default=3, help='output signal dimension')
parser.add_argument('--hidden_features', type=int, default=300, help='number of neurons per layer')
parser.add_argument('--num_hidden_layers', type=int, default=16, help='number of layers')
parser.add_argument('--hidden_omega_0', type=float, default=30.0, help='sine function coefficient [see SIREN paper]')
parser.add_argument('--D', type=float, default=0.01, help='coordinate scaling factor (see paper eq. 6.5)')
cfg = parser.parse_args()

# Reparse options
out_dir = cfg.out_dir

os.makedirs(out_dir, exist_ok=True)


##------------------------ Logger
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
if osp.exists(osp.join(out_dir, 'training.log')): os.remove(osp.join(out_dir, 'training.log'))
output_file_handler = logging.FileHandler(osp.join(out_dir, 'training.log'))
stout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(output_file_handler)
logger.addHandler(stout_handler)


##------------------------ Load data
data = h5py.File(cfg.file_h5, 'r')

# Time coordinates
obs_t = np.round(data['obs'].get('t')[:], 4)
obs_dt = np.round(data['meta'].get('dt')[()], 4)
Nt = len(obs_t)

# Spatial coordinates
obs_xyz = data['obs'].get('xyz')[()]
N_obs = len(obs_xyz)
print('N_obs', N_obs)

# To tensors
x_, y_, z_ = obs_xyz[:, 0], obs_xyz[:, 1], obs_xyz[:, 2]
x = fnp(np.repeat(x_, (M_tau,))).to(torch.float32)
y = fnp(np.repeat(y_, (M_tau,))).to(torch.float32)
z = fnp(np.repeat(z_, (M_tau,))).to(torch.float32)
t = fnp(np.tile(obs_tau, (N_obs,))).to(torch.float32)
obs_coords = torch.stack([x, y, z, t], dim=1)
NM = obs_coords.shape[0]

##------------------------ VELOCITIES
# Velocity observations
u_obs = fnp(data['obs'].get('u')[:, frames].astype(np.float32).flatten('C'))
v_obs = fnp(data['obs'].get('v')[:, frames].astype(np.float32).flatten('C'))
w_obs = fnp(data['obs'].get('w')[:, frames].astype(np.float32).flatten('C'))
vel = torch.stack([u_obs, v_obs, w_obs], dim=1)

##------------------------ WALL
# ---> Wall: tile xyz to match time dimension
wall_xyz = data['wall'].get('xyz')[()]
wall_num_spatial = wall_xyz.shape[0]
x_wall, y_wall, z_wall = wall_xyz[:, 0], wall_xyz[:, 1], wall_xyz[:, 2]
x_wall = fnp(np.repeat(x_wall, (M_tau,)))
y_wall = fnp(np.repeat(y_wall, (M_tau,)))
z_wall = fnp(np.repeat(z_wall, (M_tau,)))
t_wall = fnp(np.tile(obs_tau, (wall_num_spatial,)))
wall_coords = torch.stack([x_wall, y_wall, z_wall, t_wall], dim=1)
Nwall = wall_coords.shape[0]
wall_vel = torch.zeros((Nwall, 3), dtype=torch.float32).to(device)

##------------------------ Normalization (equal spacings)
Xall = torch.cat([obs_coords, wall_coords], dim=0)
DELTA = cfg.DELTA
cfg.mins = torch.amin(Xall, dim=0).tolist()
cfg.maxs = torch.amax(Xall, dim=0).tolist()
cfg.DELTA = DELTA
xn0 = (obs_coords[:, 0] - cfg.mins[0]) * DELTA / data['meta']['spacing'][0]
yn0 = (obs_coords[:, 1] - cfg.mins[1]) * DELTA / data['meta']['spacing'][1]
zn0 = (obs_coords[:, 2] - cfg.mins[2]) * DELTA / data['meta']['spacing'][2]
tn0 = (obs_coords[:, 3] - cfg.mins[3]) * DELTA / obs_dt
X0 = torch.stack([xn0, yn0, zn0, tn0], dim=1)
xnw = (wall_coords[:, 0] - cfg.mins[0]) * DELTA / data['meta']['spacing'][0]
ynw = (wall_coords[:, 1] - cfg.mins[1]) * DELTA / data['meta']['spacing'][1]
znw = (wall_coords[:, 2] - cfg.mins[2]) * DELTA / data['meta']['spacing'][2]
tnw = (wall_coords[:, 3] - cfg.mins[3]) * DELTA / obs_dt
Xw = torch.stack([xnw, ynw, znw, tnw], dim=1)

##------------------------ MODEL, LOSS, OPTIMIZER
model = MLP(cfg).to(device)
model.train()
print("Number of model parameters: ", sum(p.numel() for p in model.parameters()))
# model = nn.DataParallel(model, device_ids=[0, 1])

diff = lambda y, x: grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
loss_fn = lambda x, y: F.mse_loss(x, y, reduction='mean')

optimizer = torch.optim.LBFGS(model.parameters(),
                              lr=1,
                              history_size=50,
                              max_iter=50,
                              line_search_fn="strong_wolfe")

##------------------------ FITTING LOOP
batch_size = int(torch.min(torch.tensor([cfg.batch_size, NM])))
n_wall = int(torch.min(torch.tensor([batch_size * cfg.wall_batch_frac, Nwall])))
tot_batch_size = batch_size + n_wall
print('batch_obs: {0}/{1}'.format(batch_size, NM))
print('batch_wall: {0}/{1}'.format(n_wall, Nwall))

X0 = X0.to(device)
Y0 = vel.to(device)

tic = time.time()
step = -1
loss_hist = [100, 50]
while step < cfg.max_steps:
    step += 1
    wall_idx = random.sample(range(0, Nwall), n_wall)
    X = torch.cat([X0, Xw[wall_idx, :].to(device)], dim=0)


    def closure():
        optimizer.zero_grad()
        outputs = model(X)
        u = outputs[:-n_wall, 0:1].reshape((-1, cfg.t_factor))
        v = outputs[:-n_wall, 1:2].reshape((-1, cfg.t_factor))
        w = outputs[:-n_wall, 2:3].reshape((-1, cfg.t_factor))
        u_aggr, v_aggr, w_aggr = u.mean(dim=1), v.mean(dim=1), w.mean(dim=1)
        vel_aggr = torch.stack([u_aggr, v_aggr, w_aggr], dim=1)
        obs_loss = loss_fn(vel_aggr, Y0)
        wall_loss = loss_fn(outputs[-n_wall:, :], wall_vel[wall_idx, :])
        loss = obs_loss + wall_loss
        loss.backward()
        return loss


    loss = optimizer.step(closure)
    logger.debug('step: {0}, loss: {1:.5e}'.format(step, loss.item()))

    loss_hist[0] = loss_hist[1]
    loss_hist[1] = loss.item()
    if abs(loss_hist[1] - loss_hist[0]) < cfg.early_stop_tol:
        print(f'{abs(loss_hist[1] - loss_hist[0])} tolerance reached. Early stopping.')
        break

print('Time elapsed time: {:.1f} minutes.'.format((time.time() - tic) / 60))

## Saving
torch.save({'model_state_dict': model.state_dict()},
           osp.join(outDir, f'{pid}_INRww_weights.pth.tar'))

jsonCfg = json.dumps(cfg.__dict__, indent=4)
with open(osp.join(outDir, f'{pid}_cfg.json'), 'w') as fp:
    fp.write(jsonCfg)

sys.exit()



