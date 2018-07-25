import numpy as np
import torch
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm

import H36M
from Model import load_model
from util.visualize import draw
from util import config

data = DataLoader(
    H36M.Dataset(
        root=config.root['Human3.6M'],
        task=config.task,
    ),
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.num_workers,
)

with tqdm(total=len(data)) as progress:
    for in_image_space, in_camera_space in data:

        print(in_image_space[0])
        print(in_camera_space[0])

        progress.update(1)
        break