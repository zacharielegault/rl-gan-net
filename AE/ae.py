from typing import Sequence, Tuple
import os
import torch
import datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import DentalArchesDataset


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
        super().__init__()
        self.encoder = Encoder(encoder_dimensions)
        self.decoder = Decoder(decoder_dimensions, num_points)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expects tensor of shape (batch_size, 3, num_points).
        """
        gfv = self.encoder(x)
        out = self.decoder(gfv)
        
        return out, gfv


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = os.cpu_count()
    split = 1

    train_dataset = DentalArchesDataset(
        csv_filepath=f"data/kfold_split/split_{split}_train.csv",
        context_directory="data/preprocessed_partitions",
        opposing_directory="data/opposing_partitions",
        crown_directory="data/crowns",
        num_points=2048,
    )

    val_dataset = DentalArchesDataset(
        csv_filepath=f"data/kfold_split/split_{split}_val.csv",
        context_directory="data/preprocessed_partitions",
        opposing_directory="data/opposing_partitions",
        crown_directory="data/crowns",
        num_points=2048,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=24,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    autoencoder = AutoEncoder([3, 64, 128, 256, 128], [128, 256, 256, 3], 2048).to(device)

    ROOT_DIR = './autoencoder_training/'
    now = str(datetime.datetime.now())

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
    print(f"Tensorbord logdir: {LOG_DIR}")

    lr = 1.0e-4
    momentum = 0.95
    epochs = 1000
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, betas=(momentum, 0.999))

    for epoch in range(epochs):
        # Training loop
        autoencoder.train()
        cum_train_loss = 0
        for i, (input_clouds, target_clouds) in enumerate(train_dataloader):
            input_clouds = input_clouds.to(device)
            target_clouds = target_clouds.to(device)
            optimizer.zero_grad()
            predicted_clouds, gfv = autoencoder(input_clouds)
            loss = chamfer_loss(predicted_clouds, target_clouds)
            loss.backward()
            optimizer.step()

            cum_train_loss += loss * train_dataloader.batch_size

        summary_writer.add_scalar('Train loss', cum_train_loss / len(train_dataset), epoch)

        # Validation loop
        autoencoder.eval()
        cum_val_loss = 0
        for i, (input_clouds, target_clouds) in enumerate(val_dataloader):
            with torch.no_grad():
                input_clouds = input_clouds.to(device)
                target_clouds = target_clouds.to(device)
                predicted_clouds, gfv = autoencoder(input_clouds)
                loss = chamfer_loss(predicted_clouds, target_clouds)

                cum_val_loss += loss * val_dataloader.batch_size

        summary_writer.add_scalar('Val loss', cum_val_loss / len(val_dataset), epoch)


if __name__ == "__main__":
    main()
