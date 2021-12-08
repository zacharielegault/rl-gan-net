from typing import Optional, Sequence
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset import DentalArchesDataset
from AE.ae import AutoEncoder


class GAN(pl.LightningModule):
    def __init__(
            self,
            z_dim: int,
            generator_dimensions: Sequence[int],
            critic_dimensions: Sequence[int],
            encoder_dimensions: Sequence[int],
            decoder_dimensions: Sequence[int],
            num_points: int,
            split: int,
            autoencoder: Optional[AutoEncoder] = None,
            autoencoder_checkpoint: Optional[str] = None,  # Checkpoint is ignored if autoencoder is given
            lambda_gp: float = 10,
            lr: float = 3e-4,
            batch_size: int = 1,
            critic_optimizer_frequency: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.split = split

        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        self.lr = lr
        self.batch_size = batch_size
        self.critic_optimizer_frequency = critic_optimizer_frequency

        # Models
        if autoencoder is not None:
            self.autoencoder = autoencoder
        else:
            self.autoencoder = AutoEncoder(encoder_dimensions, decoder_dimensions, num_points, split)
            if autoencoder_checkpoint is not None:
                self.autoencoder.load_from_checkpoint(autoencoder_checkpoint)

        self.generator = Generator(generator_dimensions)
        self.critic = Critic(critic_dimensions)

    def forward(self, z: torch.Tensor):
        return self.generator(z)

    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int):
        input_clouds, target_clouds = batch
        if optimizer_idx == 0:  # Train generator
            return self._generator_training_step(target_clouds)
        elif optimizer_idx == 1:  # Train critic
            return self._critic_training_step(target_clouds)

    def _generator_training_step(self, batch: torch.Tensor):
        z = torch.randn(batch.shape[0], self.z_dim).type_as(batch)
        fake_crit = self.critic(self.generator(z))
        loss = -torch.mean(fake_crit)
        self.log('generator_loss', loss, on_step=False, on_epoch=True)
        return {'loss': loss, 'progress_bar': {'generator_loss': loss}, 'log': {'generator_loss': loss}}

    def _critic_training_step(self, batch: torch.Tensor):
        with torch.no_grad():
            gfv = self.autoencoder.encoder(batch)

        z = torch.randn(batch.shape[0], self.z_dim).type_as(batch)

        generated = self.generator(z)
        fake_crit = self.critic(generated)
        real_crit = self.critic(gfv)
        loss = -(real_crit.mean() - fake_crit.mean())
        loss = loss + self.lambda_gp * compute_gradient_penalty(self.critic, gfv, generated, self.device)
        self.log('critic_loss', loss, on_step=False, on_epoch=True)
        return {'loss': loss, 'progress_bar': {'critic_loss': loss}, 'log': {'critic_loss': loss}}

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        return (
            {'optimizer': generator_optimizer, 'frequency': 1},
            {'optimizer': critic_optimizer, 'frequency': self.critic_optimizer_frequency}
        )

    def train_dataloader(self):
        num_workers = 0  # os.cpu_count()
        train_dataset = DentalArchesDataset(
            csv_filepath=f"data/kfold_split/split_{self.split}_train.csv",
            context_directory="data/preprocessed_partitions",
            opposing_directory="data/opposing_partitions",
            crown_directory="data/crowns",
            num_points=2048,
        )

        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


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


def main():
    split = 1

    model = GAN(
        z_dim=32,
        generator_dimensions=[32, 128, 256, 256, 128],
        critic_dimensions=[128, 128, 128, 1],
        encoder_dimensions=[3, 64, 128, 256, 128],
        decoder_dimensions=[128, 256, 256, 3],
        num_points=2048,
        autoencoder_checkpoint="lightning_logs/version_4/checkpoints/epoch=2182-step=10914.ckpt",
        split=split,
        batch_size=8,
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10000,
        log_every_n_steps=1,
        precision=16,
        # auto_scale_batch_size="binsearch",
        callbacks=[
            ModelCheckpoint(monitor="critic_loss", verbose=True),
            EarlyStopping(monitor="critic_loss", patience=500, verbose=True)
        ]
    )

    trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    main()
