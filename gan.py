from typing import Optional, Sequence
from torch.nn.modules.pooling import AvgPool1d
from torch.utils import data
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader


class Generator(nn.Module):
    def __init__(self, dimensions: Sequence[int]):
        super().__init__()
        self.z_dim = dimensions[0]

        layers = []
        for i, (in_channels, out_channels) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(in_channels, out_channels))

            if i < len(dimensions) - 2:
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.Mish())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, dimensions: Sequence[int]):
        super().__init__()

        layers = []
        for i, (in_channels, out_channels) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.utils.parametrizations.spectral_norm(nn.Linear(in_channels, out_channels)))

            if i < len(dimensions) - 2:
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.Mish())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.net(x)


def compute_gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1).expand_as(real).to(device)

    interp = alpha * real + (1 - alpha) * fake
    interp = torch.autograd.Variable(interp, requires_grad=True)

    pred = critic(interp)

    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interp,
        grad_outputs=torch.ones_like(pred, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradient penalty computed following Petzka et al. (ICLR 2018), with only gradients larger than 1 penalized instead
    # of those different from 1 (as in Gulrajani et al., NeurIPS 2017)
    gradient_penalty = torch.mean(F.relu(gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


def train_epoch(
    dataloader,
    autoencoder: nn.Module,
    generator: Generator,
    critic: Critic,
    critic_optimizer: torch.optim.Optimizer,
    generator_optimizer: torch.optim.Optimizer,
    lambda_gp: float = 10,
    device: Optional[torch.device] = None,
    critic_iterations: int = 5,
):
    autoencoder.to(device)
    generator.to(device)
    critic.to(device)

    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        with torch.no_grad():
            gfv = autoencoder.encode(batch.to(device))
        
        z = torch.randn(batch.shape[0], generator.z_dim).to(device)

        generated = generator(z)
        fake_crit = critic(generated)
        real_crit = critic(gfv)
        critic_loss = -(real_crit.mean() - fake_crit.mean())
        critic_loss = critic_loss + lambda_gp * compute_gradient_penalty(critic, gfv, generated, device)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        if i % critic_iterations == 0:
            # Train generator every `critic_iterations` steps to allow critic to reach convergence (sort of)
            fake_crit = critic(generator(z))
            generator_loss = -torch.mean(fake_crit)

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()
        
        pbar.set_description(
            f'Critic loss = {critic_loss.detach().cpu().numpy():.3f}, ' \
            f'Generator loss = {generator_loss.detach().cpu().numpy():.3f}')


class DummyAE(nn.Module):
    def __init__(self, n_points: int = 2048, latent_dim: int = 128):
        super().__init__()

        self.conv = nn.Conv1d(3, latent_dim, 1)
        self.pool = nn.AvgPool1d(n_points)
        self.flatten = nn.Flatten()
        self.lin = nn.Linear(latent_dim, 3*n_points)
        self.unflatten = nn.Unflatten(1, (n_points, 3))
    
    def encode(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x
    
    def decode(self, x: torch.Tensor):
        x = self.lin(x)
        x = self.unflatten(x)
        return x

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))


def main():
    n_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_points = 2048
    gfv_dim = 128
    z_dim = 32

    autoencoder = DummyAE(n_points, gfv_dim)

    generator = Generator([z_dim, gfv_dim, gfv_dim, gfv_dim])  # [z_dim, ..., gfv_dim]
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=3e-4)

    critic = Critic([gfv_dim, gfv_dim, gfv_dim, 1])  # [gfv_dim, ..., 1]
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    dataloader = DataLoader(torch.randn(10000, 3, n_points), batch_size=64)

    for i in range(n_epochs):
        print(f'Epoch {i+1}/{n_epochs}')
        train_epoch(dataloader, autoencoder, generator, critic, critic_optimizer, generator_optimizer, device=device)

    # TODO: add validation loop

    return generator, critic, critic_optimizer, generator_optimizer


if __name__ == "__main__":
    main()
