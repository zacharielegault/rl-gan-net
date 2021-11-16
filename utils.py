from typing import Optional
import numpy as np
import open3d as o3d

def read_pointcloud(ply_file_path: str, subsample: Optional[int] = None) -> np.ndarray:
    """
    Example usage to gather and subsample all PLY files in a directory:
    >>> import os
    >>> directory = ...
    >>> crowns = np.asarray([read_pointcloud(os.path.join(directory, f), 2048) for f in os.listdir(directory)])
    """
    cloud = o3d.io.read_point_cloud(ply_file_path)
    cloud = np.asarray(cloud.points)  # np.ndarray with shape (n, 3)

    if subsample is not None:
        indices = np.random.choice(len(cloud), subsample, replace=False)
        cloud = cloud[indices]  # np.ndarray with shape (subsample, 3)

    return cloud
