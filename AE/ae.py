from typing import Sequence, Tuple
import os
import sys
import tqdm
import torch
import datetime
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    # Encoder([3, 64, 128, 256, 128])
    def __init__(self, dimensions: Sequence[int]):
        super().__init__()

        layers = []
        for i, (in_channels, out_channels) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Conv1d(in_channels, out_channels, 1))
            layers.append(nn.BatchNorm1d(out_channels))

            if i < len(dimensions) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.AdaptiveMaxPool1d(1))

        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expects tensor of shape (batch_size, 3, num_points).
        """
        gfv = self.net(x)  # (batch_size, gfv_dimension, 1)
        return gfv.view(-1, gfv.size(1))


class Decoder(nn.Module):
    # Decoder([128, 256, 256, 3], 2048)
    def __init__(self, dimensions: Sequence[int], num_points: int):
        super().__init__()

        layers = []
        for in_channels, out_channels in zip(dimensions[:-2], dimensions[1:-1]):
            layers.append(nn.Linear(in_channels, out_channels))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dimensions[-2], dimensions[-1]*num_points))
        self.net = nn.Sequential(*layers)

        self.output_dim = dimensions[-1]
        self.num_points = num_points
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expects tensor of shape (batch_size, 3, dimensions[0]).
        """
        pointclouds = self.net(x)
        return pointclouds.view(-1, self.output_dim, self.num_points)


class AutoEncoder(nn.Module):
    # AutoEncoder([3, 64, 128, 256, 128], [128, 256, 256, 3], 2048)
    def __init__(
        self,
        encoder_dimensions: Sequence[int],
        decoder_dimensions: Sequence[int],
        num_points: int,
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(encoder_dimensions)
        self.decoder = Decoder(decoder_dimensions, num_points)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expects tensor of shape (batch_size, 3, num_points).
        """
        gfv = self.encoder(x)
        out = self.decoder(gfv)
        
        return out, gfv


class ChamferLoss(nn.Module):
    def __init__(self, num_points: int):
        super().__init__()
        self.num_points = num_points

    def forward(self, predict_pc: torch.Tensor, gt_pc: torch.Tensor) -> torch.Tensor:
        """Expects tensors of shape (batch_size, 3, num_points).
        """
        z1, _ = torch.min(torch.norm(gt_pc.unsqueeze(-2) - predict_pc.unsqueeze(-1), dim=1), dim=-2)
        loss = z1.sum() / (len(gt_pc)*self.num_points)

        z_2, _ = torch.min(torch.norm(predict_pc.unsqueeze(-2) - gt_pc.unsqueeze(-1), dim=1), dim=-2)
        loss += z_2.sum() / (len(gt_pc)*self.num_points)
        return loss


def robust_norm(var):
    '''
    :param var: Variable of BxCxHxW
    :return: p-norm of BxCxW
    '''
    result = ((var**2).sum(dim=2) + 1e-8).sqrt() # TODO try infinity norm
    # result = (var ** 2).sum(dim=2)

    # try to make the points less dense, caused by the backward loss
    # result = result.clamp(min=7e-3, max=None)
    return result



# DATA_DIR = 'path_to_data'

X_train  = 'path_to_X_train'
X_test  = 'path_to_X_test'


class PointcloudDatasetAE(Dataset):
    def __init__(self, root, list_point_clouds):
        self.root = root
        self.list_files = list_point_clouds
        
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        points = PyntCloud.from_file(self.list_files[index])
        points = np.array(points.points)
        points_normalized = (points - (-0.5)) / (0.5 - (-0.5))
        points = points_normalized.astype(np.float)
        points = torch.from_numpy(points)
        
        return points


train_dataset = PointcloudDatasetAE(DATA_DIR, X_train)
train_dataloader = DataLoader(train_dataset, num_workers=2, shuffle=False, batch_size=48)

test_dataset = PointcloudDatasetAE(DATA_DIR, X_test)
test_dataloader = DataLoader(test_dataset, num_workers=2, shuffle=False, batch_size=1)

# for i, data in enumerate(train_dataloader):
#     data = data.permute([0,2,1])
#     print(data.shape)
#     break



autoencoder = AutoEncoder(2048).to(device)
# chamfer_loss = ChamferLossOrig(0).to(device)
chamfer_loss = ChamferLoss(2048).to(device)


# ROOT_DIR = './ae_out_copy/'
now =   str(datetime.datetime.now())

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

if not os.path.exists(ROOT_DIR + now):
    os.makedirs(ROOT_DIR + now)

LOG_DIR = ROOT_DIR + now + '/logs/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

OUTPUTS_DIR = ROOT_DIR  + now + '/outputs/'
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)

MODEL_DIR = ROOT_DIR + now + '/models/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

summary_writer = SummaryWriter(LOG_DIR)



lr = 1.0e-4
momentum = 0.95
epochs = 1000
optimizer_AE = torch.optim.Adam(autoencoder.parameters(), lr=lr, betas=(momentum, 0.999))



print('Training')
for epoch in range(epochs):
    autoencoder.train()
    for i, data in enumerate(train_dataloader):
        data = data.permute([0,2,1]).float().to(device)

        optimizer_AE.zero_grad()
        out_data, gfv = autoencoder(data)

        loss = chamfer_loss(out_data, data)
        loss.backward()
        optimizer_AE.step()
        
        print('Epoch: {}, Iteration: {}, Content Loss: {}'.format(epoch, i, loss.item()))
        summary_writer.add_scalar('Content Loss', loss.item())
    
    torch.save(autoencoder.state_dict(), MODEL_DIR+'{}_ae_.pt'.format(epoch))
