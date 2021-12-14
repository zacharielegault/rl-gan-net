import argparse
from typing import Sequence, Tuple
import os
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset import write_pointcloud, DentalArchesDataModule, ShapeNetCoreDataModule


class AutoEncoder(pl.LightningModule):
    def __init__(
            self,
            encoder_dimensions: Sequence[int],
            decoder_dimensions: Sequence[int],
            num_points: int,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.encoder = Encoder(encoder_dimensions)
        self.decoder = Decoder(decoder_dimensions, num_points)
        self.num_points = num_points

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expects tensor of shape (batch_size, 3, num_points).
        """
        gfv = self.encoder(x)
        out = self.decoder(gfv)

        return out, gfv

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_clouds, target_clouds = batch
        predicted_clouds = self.decoder(self.encoder(input_clouds))
        loss = chamfer_loss(predicted_clouds, target_clouds)
        self.log("loss/train", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_clouds, target_clouds = batch
        predicted_clouds = self.decoder(self.encoder(input_clouds))
        loss = chamfer_loss(predicted_clouds, target_clouds)
        self.log("loss/val", loss, on_step=False, on_epoch=True)

        try:
            if batch_idx == 0:
                write_pointcloud(
                    predicted_clouds[0].permute([1, 0]).cpu().numpy(),
                    os.path.join(self.logger.log_dir, "outputs", f"epoch{self.current_epoch}.ply")
                )
        except AttributeError as e:
            if "object has no attribute 'log_dir'" in str(e):
                pass

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.02)


class Encoder(nn.Module):
    # Encoder([3, 64, 128, 256, 128])
    def __init__(self, dimensions: Sequence[int]):
        super().__init__()

        layers = []
        for i, (in_channels, out_channels) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Conv1d(in_channels, out_channels, (1,)))
            layers.append(nn.BatchNorm1d(out_channels))

            if i < len(dimensions) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.AdaptiveAvgPool1d(1))

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


def chamfer_loss(predicted_clouds: torch.Tensor, target_clouds: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Expects tensors of shape (batch_size, 3, num_points).
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"Reduction type must be either 'mean' or 'none', but received {reduction}")

    z1, _ = torch.min(torch.norm(target_clouds.unsqueeze(-2) - predicted_clouds.unsqueeze(-1), dim=1), dim=-2)
    loss = z1.mean() if reduction == "mean" else z1.mean(dim=-1, keepdim=True)

    z_2, _ = torch.min(torch.norm(predicted_clouds.unsqueeze(-2) - target_clouds.unsqueeze(-1), dim=1), dim=-2)
    loss += z_2.mean() if reduction == "mean" else z_2.mean(dim=-1, keepdim=True)
    return loss


def main(args: argparse.Namespace):
    # Load config file
    import yaml
    from utils import dict_to_namespace, config_is_valid

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    config = dict_to_namespace(config)
    assert config_is_valid(config)

    # Define model
    model = AutoEncoder(
        encoder_dimensions=config.autoencoder.encoder_dimensions,
        decoder_dimensions=config.autoencoder.decoder_dimensions,
        num_points=config.num_points,
    )

    # Define dataset
    if config.dataset == "dental":
        datamodule = DentalArchesDataModule(num_points=config.num_points, split=config.split)
    else:  # config.dataset == "shapenet"
        datamodule = ShapeNetCoreDataModule(
            num_points=config.num_points,
            root_path="data/shape_net_core_uniform_samples_2048",
            subset_file="data/shape_net_core_uniform_samples_2048/subset_shapenet_5000.txt",
        )

    # Train model
    checkpoint_callback = ModelCheckpoint(monitor="loss/val", verbose=True)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10000,
        log_every_n_steps=1,
        # precision=16,
        auto_scale_batch_size="binsearch",
        default_root_dir="AE",
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="loss/val", patience=config.autoencoder.patience, verbose=True)
        ]
    )

    print("Autoencoder training")
    print(f"\ttensorboard --logdir {trainer.log_dir}")

    output_dir = os.path.join(trainer.log_dir, "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    if args.save_checkpoint:
        # Add checkpoint to config file
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)

        config["autoencoder"]["checkpoint"] = os.path.relpath(checkpoint_callback.best_model_path, os.getcwd())

        with open(args.config_file, "w") as f:
            yaml.safe_dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains the autoencoder for RL-GAN-Net.")
    parser.add_argument("--config", dest="config_file", help="Path to the YAML configuration file.")
    parser.add_argument("--save_checkpoint", action="store_true", default=False,
                        help="Creates a new configuration file that includes the checkpoint path of the best model.")
    args = parser.parse_args()
    main(args)
