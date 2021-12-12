from typing import Optional, Tuple, Union
import numpy as np
import open3d as o3d
import os
from glob import glob
from enum import IntEnum, auto
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class Phase(IntEnum):
    train = auto()
    val = auto()
    test = auto()


class ShapeNetCoreDataModule(pl.LightningDataModule):
    def __init__(self, num_points: int, batch_size: int = 1, num_workers: Optional[int] = None):
        super().__init__()
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

    def train_dataloader(self) -> DataLoader:
        train_dataset = ShapeNetCoreDataset(
            root_path="data/shape_net_core_uniform_samples_2048",
            phase="train",
            num_points=self.num_points,
        )

        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = ShapeNetCoreDataset(
            root_path="data/shape_net_core_uniform_samples_2048",
            phase="val",
            num_points=self.num_points
        )

        return DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class ShapeNetCoreDataset(Dataset):
    def __init__(self, root_path: str, phase: Union[str, Phase], num_points: int):
        self.root_path = root_path
        self.phase = phase if isinstance(phase, Phase) else Phase[phase]
        self.num_points = num_points

        # Split train-val-test 80-10-10
        all_files = sorted(glob(os.path.join(root_path, "*/*.ply")))
        train, val_and_test = train_test_split(all_files, test_size=0.2, random_state=42)
        val, test = val_and_test[:len(val_and_test)//2], val_and_test[len(val_and_test)//2:]
        self._data = {Phase.train: train, Phase.val: val, Phase.test: test}  # Lists of paths to the PLY files

    def __len__(self) -> int:
        return len(self._data[self.phase])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cloud = read_pointcloud(self._data[self.phase][idx])

        # Standardize data to 0 mean and unit variance
        cloud = (cloud - cloud.mean(0)) / cloud.std(0)

        # Subsample point clouds
        cloud = cloud[np.random.choice(len(cloud), self.num_points, replace=False)]

        # Swap axes to have channels dimension first
        cloud = np.swapaxes(cloud, -1, -2)  # (3, num_point)

        return torch.from_numpy(cloud), torch.from_numpy(cloud)


class DentalArchesDataModule(pl.LightningDataModule):
    def __init__(self, num_points: int, split: int, batch_size: int = 1, num_workers: Optional[int] = None):
        super().__init__()
        self.num_points = num_points
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

    def train_dataloader(self):
        train_dataset = DentalArchesDataset(
            csv_filepath=f"data/kfold_split/split_{self.split}_train.csv",
            context_directory="data/preprocessed_partitions",
            opposing_directory="data/opposing_partitions",
            crown_directory="data/crowns",
            num_points=self.num_points,
        )

        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        val_dataset = DentalArchesDataset(
            csv_filepath=f"data/kfold_split/split_{self.split}_val.csv",
            context_directory="data/preprocessed_partitions",
            opposing_directory="data/opposing_partitions",
            crown_directory="data/crowns",
            num_points=self.num_points,
        )

        return DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class DentalArchesDataset(Dataset):
    def __init__(
            self,
            csv_filepath: str,
            context_directory: str,
            opposing_directory: str,
            crown_directory: str,
            num_points: int
    ):
        self.df = pd.read_csv(csv_filepath, usecols=["context_file", "opposing_file", "crown_file"])
        self.context_directory = context_directory
        self.opposing_directory = opposing_directory
        self.crown_directory = crown_directory
        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Read point clouds from disk
        context, opposing, crown = self.df.iloc[idx]
        context = read_pointcloud(os.path.join(self.context_directory, context))
        opposing = read_pointcloud(os.path.join(self.opposing_directory, opposing))
        crown = read_pointcloud(os.path.join(self.crown_directory, crown))

        # Concatenate point clouds
        input_cloud = np.concatenate([context, opposing], axis=0)
        target_cloud = np.concatenate([context, opposing, crown], axis=0)

        # Standardize data to 0 mean and unit variance
        input_cloud = (input_cloud - input_cloud.mean(0)) / input_cloud.std(0)
        target_cloud = (target_cloud - target_cloud.mean(0)) / target_cloud.std(0)

        # Subsample point clouds
        input_cloud = input_cloud[np.random.choice(len(input_cloud), self.num_points, replace=False)]
        target_cloud = target_cloud[np.random.choice(len(target_cloud), self.num_points, replace=False)]

        # Swap axes to have channels dimension first
        input_cloud = np.swapaxes(input_cloud, -1, -2)  # (3, num_point)
        target_cloud = np.swapaxes(target_cloud, -1, -2)

        return torch.from_numpy(input_cloud), torch.from_numpy(target_cloud)


def read_pointcloud(ply_file_path: str, subsample: Optional[int] = None, dtype: type = np.float32) -> np.ndarray:
    """Returns array with shape (n_points, 3)
    """
    cloud = o3d.io.read_point_cloud(ply_file_path)
    cloud = np.asarray(cloud.points, dtype=dtype)  # np.ndarray with shape (n, 3)

    if subsample is not None:
        indices = np.random.choice(len(cloud), subsample, replace=False)
        cloud = cloud[indices]  # np.ndarray with shape (subsample, 3)

    return cloud


def write_pointcloud(points: np.ndarray, ply_file_path: str) -> o3d.geometry.PointCloud:
    """Expects array with shape (n_points, 3)
    """
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_file_path, pointcloud)
    return pointcloud
