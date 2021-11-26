from typing import Sequence
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gan import Generator, Critic, compute_gradient_penalty


class GAN(pl.LightningModule):
    def __init__(
            self,
            z_dim: int,
            generator_dimensions: Sequence[int],
            critic_dimensions: Sequence[int],
            autoencoder: nn.Module,
            lambda_gp: float = 10,
            lr: float = 3e-4,
            batch_size: int = 16,
            critic_optimizer_frequency: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        self.lr = lr
        self.batch_size = batch_size
        self.critic_optimizer_frequency = critic_optimizer_frequency

        # Models
        self.autoencoder = autoencoder
        self.generator = Generator(generator_dimensions)
        self.critic = Critic(critic_dimensions)

    def forward(self, z: torch.Tensor):
        return self.generator(z)

    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:  # Train generator
            return self._generator_training_step(batch)
        elif optimizer_idx == 1:  # Train critic
            return self._critic_training_step(batch)

    def _generator_training_step(self, batch: torch.Tensor):
        z = torch.randn(batch.shape[0], self.z_dim)
        fake_crit = self.critic(self.generator(z))
        loss = -torch.mean(fake_crit)
        return {'loss': loss, 'progress_bar': {'generator_loss': loss}, 'log': {'generator_loss': loss}}

    def _critic_training_step(self, batch: torch.Tensor):
        with torch.no_grad():
            gfv = self.autoencoder.encode(batch)

        z = torch.randn(batch.shape[0], self.z_dim)

        generated = self.generator(z)
        fake_crit = self.critic(generated)
        real_crit = self.critic(gfv)
        loss = -(real_crit.mean() - fake_crit.mean())
        loss = loss + self.lambda_gp * compute_gradient_penalty(self.critic, gfv, generated, self.device)
        return {'loss': loss, 'progress_bar': {'critic_loss': loss}, 'log': {'critic_loss': loss}}

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        critic_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return (
            {'optimizer': generator_optimizer, 'frequency': 1},
            {'optimizer': critic_optimizer, 'frequency': self.critic_optimizer_frequency}
        )

    def train_dataloader(self):
        dataset = ...
        return DataLoader(dataset, batch_size=self.batch_size)
