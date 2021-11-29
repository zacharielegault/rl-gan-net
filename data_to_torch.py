import os
import itertools
import numpy as np
import torch
import pandas as pd

from utils import read_pointcloud


# All clouds
directories = {
    "crowns": "crowns",
    "opposing": "opposing_partitions",
    "contexts": "preprocessed_partitions",
}

for name, directory in directories.items():
    crowns = np.asarray([read_pointcloud(f"data/{directory}/{f}", 2048) for f in os.listdir(f"data/{directory}")])
    crowns = np.swapaxes(crowns, 1, 2)
    crowns = torch.from_numpy(crowns)
    torch.save(crowns, f"data/{name}.pt")

# Individual splits
os.makedirs("data/individual_splits")

for split in range(1, 6 + 1):
    for phase in ["train", "val", "test"]:
        df = pd.read_csv(
            f"data/kfold_split/split_{split}_{phase}.csv", usecols=["context_file", "opposing_file", "crown_file"]
        )

        # Contexts
        contexts = np.asarray([read_pointcloud(f"data/preprocessed_partitions/{f}", 2048) for f in df["context_file"]])
        contexts = np.swapaxes(contexts, 1, 2)
        contexts = torch.from_numpy(contexts)
        torch.save(contexts, f"data/individual_splits/contexts_{split}_{phase}.pt")

        # Opposing
        opposing = np.asarray([read_pointcloud(f"data/opposing_partitions/{f}", 2048) for f in df["opposing_file"]])
        opposing = np.swapaxes(opposing, 1, 2)
        opposing = torch.from_numpy(opposing)
        torch.save(opposing, f"data/individual_splits/opposing_{split}_{phase}.pt")

        # Crowns
        crowns = np.asarray([read_pointcloud(f"data/crowns/{f}", 2048) for f in df["crown_file"]])
        crowns = np.swapaxes(crowns, 1, 2)
        crowns = torch.from_numpy(crowns)
        torch.save(crowns, f"data/individual_splits/crowns_{split}_{phase}.pt")
