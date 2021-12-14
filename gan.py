import argparse
from typing import Optional, Sequence
import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset import write_pointcloud, DentalArchesDataModule, ShapeNetCoreDataModule
from ae import AutoEncoder


class GAN(pl.LightningModule):
    def __init__(
            self,
            z_dim: int,
            generator_dimensions: Sequence[int],
            critic_dimensions: Sequence[int],
            encoder_dimensions: Sequence[int],
            decoder_dimensions: Sequence[int],
            num_points: int,
            autoencoder: Optional[AutoEncoder] = None,
            autoencoder_checkpoint: Optional[str] = None,  # Checkpoint is ignored if autoencoder is given
            lambda_gp: float = 10,
            lr: float = 3e-4,
            critic_optimizer_frequency: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_points = num_points
        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        self.lr = lr
        self.critic_optimizer_frequency = critic_optimizer_frequency

        # Models
        if autoencoder is not None:
            self.autoencoder = autoencoder
        else:
            self.autoencoder = AutoEncoder(encoder_dimensions, decoder_dimensions, num_points)
            if autoencoder_checkpoint is not None:
                self.autoencoder = self.autoencoder.load_from_checkpoint(autoencoder_checkpoint)

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


def main(args: argparse.Namespace):
    # Load config file
    import yaml
    from utils import dict_to_namespace, config_is_valid

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    config = dict_to_namespace(config)
    assert config_is_valid(config)

    # Define model
    model = GAN(
        z_dim=config.z_dim,
        generator_dimensions=config.gan.generator_dimensions,
        critic_dimensions=config.gan.critic_dimensions,
        encoder_dimensions=config.autoencoder.encoder_dimensions,
        decoder_dimensions=config.autoencoder.decoder_dimensions,
        num_points=config.num_points,
        autoencoder_checkpoint=config.autoencoder.checkpoint,
    )

    # Define dataset
    if config.dataset == "dental":
        datamodule = DentalArchesDataModule(
            num_points=config.num_points,
            split=config.split,
            batch_size=config.gan.batch_size,
        )
    else:  # config.dataset == "shapenet"
        datamodule = ShapeNetCoreDataModule(num_points=config.num_points, batch_size=512, root_path="data/shape_net_core_uniform_samples_2048")

    # Train model
    checkpoint_callback = ModelCheckpoint(monitor="critic_loss", verbose=True)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        max_epochs=config.gan.max_epochs,
        log_every_n_steps=1,
        # precision=16,
        # auto_scale_batch_size="binsearch",
        default_root_dir="GAN",
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="critic_loss", patience=500, verbose=True)
        ]
    )

    print("GAN training")
    print(f"\ttensorboard --logdir {trainer.log_dir}")

    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    if args.save_checkpoint:
        # Add checkpoint to config file
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)

        config["gan"]["checkpoint"] = os.path.relpath(checkpoint_callback.best_model_path, os.getcwd())

        with open(args.config_file, "w") as f:
            yaml.safe_dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains the GAN for RL-GAN-Net.")
    parser.add_argument("--config", dest="config_file", help="Path to the YAML configuration file.")
    parser.add_argument("--save_checkpoint", action="store_true", default=False,
                        help="Creates a new configuration file that includes the checkpoint path of the best model.")
    args = parser.parse_args()
    main(args)
