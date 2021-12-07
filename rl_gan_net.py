from typing import Optional
import os
import torch
import random
from datetime import datetime
from collections import deque

import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from dataset import DentalArchesDataset
from AE.ae import AutoEncoder, chamfer_loss
from GAN.gan import GAN


class ReplayBuffer:
    def __init__(self, size: int, device: Optional[torch.device] = None):
        self.episodes = deque(maxlen=size)
        self.device = device

    def add_to_buffer(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor):
        """All tensors are expected to be batched, i.e. with shape (batch_size, ...)."""
        self.episodes.extend([(s, a, r, s_) for s, a, r, s_ in zip(state.detach().cpu(), action.detach().cpu(), reward.detach().cpu(), next_state.detach().cpu())])

    def get_batch(self, batch_size: int):
        states = []
        actions = []
        rewards = []
        next_state = []

        for i in range(batch_size):
            epi = random.choice(self.episodes)
            states.append(epi[0])
            actions.append(epi[1])
            rewards.append(epi[2])
            next_state.append(epi[3])

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_state = torch.stack(next_state).to(self.device)

        return states, actions, rewards, next_state


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, z_shape: int):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = z_shape
        
        self.linear1 = nn.Linear(self.state_dim, 400)
        # self.bn1 = nn.BatchNorm1d(400)
        self.linear2 = nn.Linear(400 + z_shape, 300)
        # self.bn2 = nn.BatchNorm1d(300)
        self.linear3 = nn.Linear(300, 300)
        self.linear4 = nn.Linear(300, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, state, z):
        out = (F.relu(self.linear1(state)))
        out = (F.relu(self.linear2(torch.cat([out, z], dim=1))))
        out = self.linear3(out)  # Is this really necessary? Two linear layers without an activation fn in between.
        out = self.linear4(out)

        return out


class ActorNet(nn.Module):
    def __init__(self, state_dim: int,  z_shape: int, max_action: float = 10):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = z_shape

        self.linear1 = nn.Linear(self.state_dim, 400)
        # self.bn1 = nn.BatchNorm1d(400)

        self.linear2 = nn.Linear(400, 400)
        # self.bn2 = nn.BatchNorm1d(400)

        self.linear3 = nn.Linear(400, 300)
        self.linear4 = nn.Linear(300, self.num_actions)

        self.max_action = max_action  # Check in original paper

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = F.leaky_relu((self.linear1(x)))
        out = F.leaky_relu((self.linear2(out)))
        out = torch.tanh(self.linear3(out))
        out = self.max_action * torch.tanh(self.linear4(out))
        return out


class DDPG(nn.Module):
    def __init__(self, max_action: float, z_dim: int, replay_buffer_device: Optional[torch.device] = None):
        super().__init__()
        self.actor = ActorNet(128, z_dim, max_action)
        self.critic = CriticNet(128, z_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.replay_buffer = ReplayBuffer(int(1e6), replay_buffer_device)

    def forward(self, batch_size: int):
        state, action, reward, next_state = self.replay_buffer.get_batch(batch_size)
        
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


def main():
    # Parameters
    max_action = 2
    z_dim = 32
    max_steps, start_time = 1e6, 1e3
    # max_steps, start_time = 10, 5
    batch_size_actor = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(15)

    split = 1

    autoencoder = AutoEncoder(
        encoder_dimensions=[3, 64, 128, 256, 128],
        decoder_dimensions=[128, 256, 256, 3],
        num_points=2048,
        split=1,
    ).to(device)
    # autoencoder.load_from_checkpoint("path/to/autoencoder/checckpoint.ckpt")

    gan = GAN(
        z_dim=32,
        generator_dimensions=[32, 128, 256, 256, 128],
        critic_dimensions=[128, 128, 128, 1],
        encoder_dimensions=[3, 64, 128, 256, 128],
        decoder_dimensions=[128, 256, 256, 3],
        num_points=2048,
        split=split,
        batch_size=8,
    ).to(device)
    # gan.load_from_checkpoint("path/to/gan/checckpoint.ckpt")

    # gan.generator
    # gan.critic

    ddpg = DDPG(max_action, z_dim, replay_buffer_device=device).to(device)

    autoencoder.eval()  # to be checked
    gan.eval()

    # Dataloader
    batch_size = 10

    # num_workers = os.cpu_count()
    num_workers = 0
    train_dataset = DentalArchesDataset(
        csv_filepath=f"data/kfold_split/split_{split}_train.csv",
        context_directory="data/preprocessed_partitions",
        opposing_directory="data/opposing_partitions",
        crown_directory="data/crowns",
        num_points=2048,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_dataset = DentalArchesDataset(
        csv_filepath=f"data/kfold_split/split_{split}_val.csv",
        context_directory="data/preprocessed_partitions",
        opposing_directory="data/opposing_partitions",
        crown_directory="data/crowns",
        num_points=2048,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader_iterator = iter(train_dataloader)
    test_loader_iterator = iter(test_dataloader)

    ROOT_DIR = './results/'
    now = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())

    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    if not os.path.exists(ROOT_DIR + now):
        os.makedirs(ROOT_DIR + now)

    LOG_DIR = ROOT_DIR + now + '/logs/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    OUTPUTS_DIR = ROOT_DIR + now + '/outputs/'
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    MODEL_DIR = ROOT_DIR + now + '/models/'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    summary_writer = SummaryWriter(LOG_DIR)

    for tsteps in range(int(max_steps)):
        try:
            data = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_dataloader)
            data = next(train_loader_iterator)

        if tsteps != 0:  # Skip the first iteration, start filling the replay buffer first
            losses = ddpg(batch_size_actor)

        # Fill replay buffer
        input_clouds, output_clouds = data
        input_clouds = input_clouds.to(device)
        output_clouds = output_clouds.to(device)
        state_t = autoencoder.encoder(input_clouds)

        if tsteps < start_time:
            # Generate random actions to fill buffer during warmup period
            action_t = -2 * max_action * torch.rand(state_t.size(0), z_dim) + max_action
            action_t = action_t.to(device)
        else:
            action_t = (ddpg.actor(state_t).detach() + 0.1 * torch.randn(state_t.size(0), z_dim).to(device)).clamp(-max_action, max_action)
            action_t.to(device)

        next_state = gan.generator(action_t)

        reward_gfv = -torch.mean(torch.pow(next_state - state_t, 2), dim=-1, keepdim=True)
        reward_chamfer = -chamfer_loss(autoencoder.decoder(next_state), autoencoder.decoder(state_t), reduction="none")
        reward_disc = -gan.critic(next_state)
        reward = reward_gfv * 0.1 + reward_chamfer * 5.0 + reward_disc * 0.1 + (-torch.norm(action_t, dim=-1, keepdim=True)) * 0.1
        ddpg.replay_buffer.add_to_buffer(state_t, action_t, reward, next_state)

        # Logging
        if tsteps % 10:
            print(
                f'Iter : {tsteps}, '
                f'Reward : {reward.mean():.4f}, '
                f'GFV: {reward_gfv.mean():.4f}, '
                f'Chamfer: {reward_chamfer.mean():.4f}, '
                f'Disc: {reward_disc.mean():.4f}'
            )

        summary_writer.add_scalar('train total mean reward', reward.mean())
        summary_writer.add_scalar('train gfv mean rewards', reward_gfv.mean())
        summary_writer.add_scalar('train mean reward_chamfer', reward_chamfer.mean())
        summary_writer.add_scalar('train mean reward_disc', reward_disc.mean())

        if tsteps % 1 == 0 and tsteps > start_time:
            if tsteps % 1000 <= 10 and tsteps > start_time:
                optimal_action = ddpg.actor(state_t).detach()
                new_state = gan.generator(optimal_action)

                out_data = autoencoder.decoder(new_state)

                output = out_data[0, :, :]
                output = output.permute([1, 0]).detach().cpu().numpy()

                fig = plt.figure()
                ax_x = fig.add_subplot(111, projection='3d')
                x_ = output
                ax_x.scatter(x_[:, 0], x_[:, 1], x_[:, 2])
                ax_x.set_xlim([0, 1])
                ax_x.set_ylim([0, 1])
                ax_x.set_zlim([0, 1])
                fig.savefig(OUTPUTS_DIR+'/{}_{}.png'.format(tsteps, 'val_out'))

                output = autoencoder.decoder(state_t)  # generator
                output = output[0, :, :]
                output = output.permute([1, 0]).detach().cpu().numpy()

                fig = plt.figure()
                ax_x = fig.add_subplot(111, projection='3d')
                x_ = output
                ax_x.scatter(x_[:, 0], x_[:, 1], x_[:, 2])
                ax_x.set_xlim([0, 1])
                ax_x.set_ylim([0, 1])
                ax_x.set_zlim([0, 1])
                fig.savefig(OUTPUTS_DIR + '/{}_{}.png'.format(tsteps, 'val_in'))

                plt.close('all')

                torch.save(ddpg.state_dict(), MODEL_DIR + '{}_ddpg_.pt'.format(tsteps))

    torch.save(ddpg.state_dict(), MODEL_DIR + '{}_ddpg_.pt'.format('final'))


if __name__ == "__main__":
    main()
