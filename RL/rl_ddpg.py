import os
import sys
import tqdm
import torch
import random
import datetime

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pyntcloud import PyntCloud
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

######### Parameters
max_action = 2 # To be cheched
z_dim = 5 # To be cheched
max_steps = 1e6
batch_size_actor = 100
start_time = 1e3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(15)


# # All Models

# ### Replay Buffer


class ReplayBuffer():
    def __init__(self, size):
        self.episodes = []
        self.buffer_size = size

    def add_to_buffer(self, state, action, reward, next_state):
        if len(self.episodes) == self.buffer_size:
            self.episodes = self.episodes[1:]
        self.episodes.append((state.detach().cpu().numpy(), action.detach().cpu().numpy(), reward.detach().cpu().numpy(), next_state.detach().cpu().numpy()))

    def get_batch(self, batch_size=10):
        states = []
        actions = []
        rewards = []
        next_state = []
        done = []

        for i in range(batch_size):
            epi = random.choice(self.episodes)
            states.append(epi[0])
            actions.append(epi[1])
            rewards.append(epi[2])
            next_state.append(epi[3])
        
        rewards = np.array(rewards)
        rewards = rewards.reshape((rewards.shape[0],1))
        return torch.Tensor(states), torch.Tensor(actions), torch.Tensor(rewards), torch.Tensor(next_state)


# ### Critic Network


class CriticNet(nn.Module):
    def __init__(self, state_dim, z_shape):
        super(CriticNet, self).__init__()
        self.state_dim = state_dim
        self.num_actions = z_shape
        

        self.linear1 = nn.Linear(self.state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.linear2 = nn.Linear(400 + z_shape, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.linear3 = nn.Linear(300, 300)
        self.linear4 = nn.Linear(300, 1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, state, z):
        out = (F.relu(self.linear1(state)))
        out = (F.relu(self.linear2(torch.cat([out, z], dim=1))))
        out = self.linear3(out)
        out = self.linear4(out)

        return out


# ### Actor Network


class ActorNet(nn.Module):
    def __init__(self, state_dim,  z_shape, max_action=10):
        super(ActorNet, self).__init__()
        self.state_dim = state_dim
        self.num_actions = z_shape

        self.linear1 = nn.Linear(self.state_dim, 400)
        self.bn1 = nn.BatchNorm1d(100)

        self.linear2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(300)

        self.linear3 = nn.Linear(400, 300)
        self.linear4 = nn.Linear(300, self.num_actions)

        self.max_action = max_action

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = F.leaky_relu((self.linear1(x)))
        out = F.leaky_relu((self.linear2(out)))
        out = F.tanh(self.linear3(out))
        out = self.max_action * F.tanh(self.linear4(out))
        return out


# ### Autoencoder


# ### Generator


# ### Self-Attention


# ### Discriminator


class DDPG(nn.Module):
    def __init__(self, max_action):
        super(DDPG, self).__init__()
        self.actor = ActorNet(128, z_dim, max_action)
        self.critic = CriticNet(128, z_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.replay_buffer = ReplayBuffer(int(1e6))

    def get_optimal_action(self, state):
        return self.actor(state)

    def forward(self):
        state, action, reward, next_state = self.replay_buffer.get_batch(batch_size_actor)
        
        state = state[:,0,:].float()
        next_state = next_state[:,0,:].float()
        action = action[:,0,:].float()
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        
        target_q = reward

        q_batch = self.critic(state, action)

        self.critic_optimizer.zero_grad()

        value_loss = F.mse_loss(q_batch, target_q)
        value_loss.backward()
        
        self.critic_optimizer.step() 

        self.actor_optimizer.zero_grad()

        policy_loss = - self.critic(state, self.actor(state)).mean()
        policy_loss.backward()
        
        self.actor_optimizer.step()

        return value_loss, policy_loss



# weights_ae = # add string
# weights_gen = # add string
# weight_disc = # add string


autoencoder = # AutoEncoder(2048).to(device) Add our autoencoder
autoencoder.load_state_dict(torch.load(weights_ae))

generator = # GenSAGAN(z_dim=z_dim).to(device) # Add our GAN
generator.load_state_dict(torch.load(weights_gen))

discriminator = # DiscSAGAN().to(device) # Add ours
discriminator.load_state_dict(torch.load(weight_disc))

ddpg = DDPG(max_action).to(device)


# autoencoder.eval() # to be checked
# generator.eval()
# discriminator.eval()


# # Dataloader
# train_dataset = PointcloudDatasetNoisy(DATA_DIR, X_train) # use our data
train_dataloader = DataLoader(train_dataset, num_workers=0, shuffle=True, batch_size=1)
train_loader_iterator = iter(train_dataloader)

# test_dataset = PointcloudDatasetNoisy(DATA_DIR, X_test) # use our data
test_dataloader = DataLoader(test_dataset, num_workers=0, shuffle=True, batch_size=1)
test_loader_iterator = iter(test_dataloader)

for i, data in enumerate(train_dataloader):
    data = data.permute([0,2,1])
    print(data.shape)
    break


# # RL Agent Training
# to be checked

ROOT_DIR = './rl_out_no_def/'
now =   str(datetime.datetime.now())+'_start_4_max_2_f'

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


# chamferloss = ChamferLoss(2048).to(device)


for tsteps in range(0,int(max_steps)):
    try:
        data = next(train_loader_iterator) # from Zack
    except StopIteration:
        train_loader_iterator = iter(train_dataloader)
        data = next(train_loader_iterator)
    data = data.permute([0,2,1]).float().to(device)
            
    if tsteps != 0:
        losses = ddpg()
            
    state_t = # autoencoder.encode(data) # import from AE
    
    if tsteps < start_time:
        action_t = -2 * max_action * torch.rand(1, z_dim) + max_action
        action_t = action_t.to(device)
    else:
        action_t = (ddpg.get_optimal_action(state_t).detach() + 0.1 * torch.randn(1, z_dim).to(device)).clamp(-max_action, max_action)

    next_state, _ = # generator(action_t) # to be used from pretrained gan
    
    reward_gfv = -F.mse_loss(next_state, state_t)
    reward_chamfer = # -chamferloss(autoencoder.decode(next_state), autoencoder.decode(state_t)) # to be imported
    reward_disc, _ = # discriminator(next_state) # to be imported
    reward_disc = torch.mean(reward_disc)
    reward = reward_gfv * 0.1 + reward_chamfer * 5.0 + reward_disc * 0.1 + (-torch.norm(action_t)) * 0.1
    ddpg.replay_buffer.add_to_buffer(state_t, action_t, reward, next_state)

    if tsteps % 10:
        print('Iter : {}, Reward : {:.4f}, GFV: {:.4f}, Chamfer: {:.4f}, Disc: {:.4f}, Action: {}'.format(tsteps, reward, reward_gfv, reward_chamfer, reward_disc, action_t))
    
    summary_writer.add_scalar('train total reward', reward)
    summary_writer.add_scalar('train gfv rewards', reward_gfv)
    summary_writer.add_scalar('train reward_chamfer', reward_chamfer)
    summary_writer.add_scalar('train reward_disc', reward_disc)

    # if tsteps % 1 == 0 and tsteps > start_time:
    if tsteps % 1000 <= 10 and tsteps > start_time:
        optimal_action = ddpg.get_optimal_action(state_t).detach()
        new_state, _ = # generator(optimal_action) # to be used from pretrained gan
        
        out_data = # autoencoder.decode(new_state) # c

        output = out_data[0,:,:]
        output = output.permute([1,0]).detach().cpu().numpy()

        fig = plt.figure()
        ax_x = fig.add_subplot(111, projection='3d')
        x_ = output
        ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
        ax_x.set_xlim([0,1])
        ax_x.set_ylim([0,1])
        ax_x.set_zlim([0,1])
        fig.savefig(OUTPUTS_DIR+'/{}_{}_{}.png'.format(tsteps, i, 'val_out'))

        output = # autoencoder.decode(state_t) # generator
        output = output[0,:,:]
        output = output.permute([1,0]).detach().cpu().numpy()

        fig = plt.figure()
        ax_x = fig.add_subplot(111, projection='3d')
        x_ = output
        ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
        ax_x.set_xlim([0,1])
        ax_x.set_ylim([0,1])
        ax_x.set_zlim([0,1])
        fig.savefig(OUTPUTS_DIR+'/{}_{}_{}.png'.format(tsteps, i, 'val_in'))

        plt.close('all')

        torch.save(ddpg.state_dict(), MODEL_DIR+'{}_ddpg_.pt'.format(tsteps))
    
torch.save(ddpg.state_dict(), MODEL_DIR+'{}_ddpg_.pt'.format('final'))
