from typing import Sequence, Tuple
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset import DentalArchesDataset
from AE.ae import Encoder, Decoder, chamfer_loss


class AutoEncoder(pl.LightningModule):
    def __init__(
            self,
            encoder_dimensions: Sequence[int],
            decoder_dimensions: Sequence[int],
            num_points: int,
            split: int,
            batch_size: int = 1,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.encoder = Encoder(encoder_dimensions)
        self.decoder = Decoder(decoder_dimensions, num_points)
        self.split = split
        self.batch_size = batch_size

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
        loss = self.training_step(batch, batch_idx)
        self.log("loss/val", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        num_workers = os.cpu_count()
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

    def val_dataloader(self):
        num_workers = os.cpu_count()
        val_dataset = DentalArchesDataset(
                csv_filepath=f"data/kfold_split/split_{self.split}_val.csv",
                context_directory="data/preprocessed_partitions",
                opposing_directory="data/opposing_partitions",
                crown_directory="data/crowns",
                num_points=2048,
        )

        return DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


def main():
    split = 1
    model = AutoEncoder([3, 64, 128, 256, 128], [128, 256, 256, 3], 2048, split)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10000,
        log_every_n_steps=1,
        precision=16,
        auto_scale_batch_size="binsearch",
        callbacks=[
            ModelCheckpoint(monitor="loss/val", verbose=True),
            EarlyStopping(monitor="loss/val", patience=500, verbose=True)
        ]
    )

    trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    main()
