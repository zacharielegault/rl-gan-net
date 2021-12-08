from types import SimpleNamespace
from typing import Optional, Dict
import os
import torch
import random
from datetime import datetime
from collections import deque

import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import DentalArchesDataset
from AE.ae import AutoEncoder, chamfer_loss
from GAN.gan import GAN
from utils import dict_to_namespace


class ReplayBuffer:
    def __init__(self, capacity: int, device: Optional[torch.device] = None):
        self.episodes = deque(maxlen=capacity)
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
    def __init__(self, gfv_dim: int, z_dim: int):
        super().__init__()
        self.gfv_dim = gfv_dim
        self.z_dim = z_dim
        
        self.linear1 = nn.Linear(gfv_dim, 400)
        # self.bn1 = nn.BatchNorm1d(400)
        self.linear2 = nn.Linear(400 + z_dim, 300)
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
    def __init__(self, gfv_dim: int, z_dim: int, max_action: float = 10):
        super().__init__()
        self.gfv_dim = gfv_dim
        self.z_dim = z_dim

        self.linear1 = nn.Linear(gfv_dim, 400)
        # self.bn1 = nn.BatchNorm1d(400)

        self.linear2 = nn.Linear(400, 400)
        # self.bn2 = nn.BatchNorm1d(400)

        self.linear3 = nn.Linear(400, 300)
        self.linear4 = nn.Linear(300, z_dim)

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
    def __init__(
            self,
            max_action: float,
            gfv_dim: int,
            z_dim: int,
            w_gfv: float,
            w_chamfer: float,
            w_disc: float,
            regularization: float,
            start_time: int,
            replay_buffer_capacity: int,
            replay_buffer_device: Optional[torch.device] = None
    ):
        super().__init__()
        self.actor = ActorNet(gfv_dim, z_dim, max_action)
        self.critic = CriticNet(gfv_dim, z_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity, replay_buffer_device)

        self.w_gfv = w_gfv
        self.w_chamfer = w_chamfer
        self.w_disc = w_disc
        self.regularization = regularization
        self.start_time = start_time

    def forward(self, x: torch.Tensor):
        return self.actor(x)

    def training_step(self, batch_size: int):
        state, action, reward, next_state = self.replay_buffer.get_batch(batch_size)

        # Critic
        self.critic_optimizer.zero_grad()
        value_loss = F.mse_loss(self.critic(state, action), reward)
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor
        self.actor_optimizer.zero_grad()
        policy_loss = - self.critic(state, self.actor(state)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        return value_loss, policy_loss

    @torch.no_grad()
    def generate_examples(self, step: int, input_clouds, autoencoder: AutoEncoder, gan: GAN, device: Optional[torch.device]) -> Dict[str, torch.Tensor]:
        state_t = autoencoder.encoder(input_clouds.to(device))

        if step < self.start_time:
            # Generate random actions to fill buffer during warmup period
            action_t = -2 * self.actor.max_action * torch.rand(state_t.size(0), self.actor.z_dim) + self.actor.max_action
            action_t = action_t.to(device)
        else:
            action_t = (self.actor(state_t).detach() + 0.1 * torch.randn(state_t.size(0), self.actor.z_dim).to(device)).clamp(-self.actor.max_action, self.actor.max_action)

        next_state = gan.generator(action_t)
        reward_gfv = -torch.mean(torch.pow(next_state - state_t, 2), dim=-1, keepdim=True)
        reward_chamfer = -chamfer_loss(autoencoder.decoder(next_state), autoencoder.decoder(state_t), reduction="none")
        reward_disc = -gan.critic(next_state)
        reward = reward_gfv * self.w_gfv \
            + reward_chamfer * self.w_chamfer \
            + reward_disc * self.w_disc \
            + (-torch.norm(action_t, dim=-1, keepdim=True)) * self.regularization
        self.replay_buffer.add_to_buffer(state_t, action_t, reward, next_state)

        return {"total": reward, "gfv": reward_gfv, "chamfer": reward_chamfer, "disc": reward_disc}

    def to(self, device: torch.device):
        self.replay_buffer.device = device
        return super().to(device)


def main(config: SimpleNamespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(15)

    autoencoder = AutoEncoder(
        encoder_dimensions=config.autoencoder.encoder_dimensions,
        decoder_dimensions=config.autoencoder.decoder_dimensions,
        num_points=config.num_points,
        split=config.split,
    ).to(device)
    autoencoder.load_from_checkpoint(config.autoencoder.checkpoint)

    gan = GAN(
        z_dim=32,
        generator_dimensions=config.gan.generator_dimensions,
        critic_dimensions=config.gan.critic_dimensions,
        encoder_dimensions=config.autoencoder.encoder_dimensions,
        decoder_dimensions=config.autoencoder.decoder_dimensions,
        num_points=config.num_points,
        split=config.split,
        autoencoder=autoencoder,
    ).to(device)
    gan.load_from_checkpoint(config.gan.checkpoint)

    ddpg = DDPG(
        max_action=config.ddpg.max_action,
        gfv_dim=config.gfv_dim,
        z_dim=config.z_dim,
        w_gfv=config.ddpg.w_gfv,
        w_chamfer=config.ddpg.w_chamfer,
        w_disc=config.ddpg.w_disc,
        regularization=config.ddpg.regularization,
        start_time=int(config.ddpg.start_time),
        replay_buffer_capacity=int(1e6),
    ).to(device)

    autoencoder.eval()
    gan.eval()

    # num_workers = os.cpu_count()
    num_workers = 0
    train_dataset = DentalArchesDataset(
        csv_filepath=f"data/kfold_split/split_{config.split}_train.csv",
        context_directory="data/preprocessed_partitions",
        opposing_directory="data/opposing_partitions",
        crown_directory="data/crowns",
        num_points=config.num_points,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,  # RL-GAN-Net paper uses batch size of 1
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader_iterator = iter(train_dataloader)

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

    for tsteps in range(int(config.ddpg.max_steps)):
        try:
            data = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_dataloader)
            data = next(train_loader_iterator)

        if tsteps != 0:  # Skip the first iteration, start filling the replay buffer first
            ddpg.training_step(config.ddpg.batch_size_actor)

        # Fill replay buffer
        input_clouds, _ = data
        reward = ddpg.generate_examples(tsteps, input_clouds, autoencoder, gan, device)

        # Logging
        if tsteps % 10:
            print(
                f'Iter : {tsteps}, '
                f'Reward : {reward["total"].mean():.4f}, '
                f'GFV: {reward["gfv"].mean():.4f}, '
                f'Chamfer: {reward["chamfer"].mean():.4f}, '
                f'Disc: {reward["disc"].mean():.4f}'
            )

        summary_writer.add_scalar('train total mean reward', reward["total"].mean(), tsteps)
        summary_writer.add_scalar('train gfv mean rewards', reward["gfv"].mean(), tsteps)
        summary_writer.add_scalar('train mean reward_chamfer', reward["chamfer"].mean(), tsteps)
        summary_writer.add_scalar('train mean reward_disc', reward["disc"].mean(), tsteps)

        if tsteps % 1 == 0 and tsteps > config.ddpg.start_time:
            if tsteps % 1000 <= 10 and tsteps > config.ddpg.start_time:
                state_t = autoencoder.encoder(input_clouds.to(device))

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
                fig.savefig(OUTPUTS_DIR + '/{}_{}.png'.format(tsteps, 'val_out'))

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
    import yaml
    from utils import dict_to_namespace, config_is_valid

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config = dict_to_namespace(config)
    assert config_is_valid(config)
    main(config)
