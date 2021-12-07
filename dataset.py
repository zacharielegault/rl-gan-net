from typing import Optional, Tuple
import numpy as np
import open3d as o3d
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


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

        # Subsample point clouds
        input_cloud = input_cloud[np.random.choice(len(input_cloud), self.num_points, replace=False)]
        target_cloud = target_cloud[np.random.choice(len(target_cloud), self.num_points, replace=False)]

        # Swap axes to have channels dimension first
        input_cloud = np.swapaxes(input_cloud, -1, -2)  # (3, num_point)
        target_cloud = np.swapaxes(target_cloud, -1, -2)

        return torch.from_numpy(input_cloud), torch.from_numpy(target_cloud)


def read_pointcloud(ply_file_path: str, subsample: Optional[int] = None, dtype: type = np.float32) -> np.ndarray:
    """
    Example usage to gather and subsample all PLY files in a directory:
    >>> import os
    >>> directory = ...
    >>> crowns = np.asarray([read_pointcloud(os.path.join(directory, f), 2048) for f in os.listdir(directory)])
    """
    cloud = o3d.io.read_point_cloud(ply_file_path)
    cloud = np.asarray(cloud.points, dtype=dtype)  # np.ndarray with shape (n, 3)

    if subsample is not None:
        indices = np.random.choice(len(cloud), subsample, replace=False)
        cloud = cloud[indices]  # np.ndarray with shape (subsample, 3)

    return cloud


def write_pointcloud(points: np.ndarray, ply_file_path: str) -> None:
    """Expects array with shape (n_points, 3)
    """
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_file_path, pointcloud)
