import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import H36M
import Model.bilinear
from H36M.annotation import Annotation
from util import config

assert config.task == 'valid'

dataset = H36M.Dataset(
    root=config.root['Human3.6M'],
    task=config.task,
)
data = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=config.num_workers,
)

device = torch.device(config.device)
bilinear, _, _, _, pretrained_epoch, _ = Model.bilinear.load_model(device, config.pretrained['bilinear'])

total_dist = torch.zeros(17)
total = 0

dim = 3
mean = torch.Tensor(dataset.mean[dim]).to(device).view(-1, 3 * 17)
stddev = torch.Tensor(dataset.stddev[dim]).to(device).view(-1, 3 * 17)

with tqdm(total=len(data), desc='%d epoch' % pretrained_epoch) as progress:
    with torch.set_grad_enabled(False):
        for in_image_space, in_camera_space, center, scale, _, _ in data:

            in_image_space = in_image_space.to(device).view(-1, 2 * 17)
            in_camera_space = in_camera_space.to(device).view(-1, 3 * 17)

            n_batch = in_image_space.shape[0]
            total = total + n_batch

            prediction = bilinear(in_image_space)
            prediction_in_camera_space = stddev * prediction + mean

            ground_truth = in_camera_space.view(-1, 17, 3)
            prediction = prediction_in_camera_space.view(-1, 17, 3)

            dist = torch.sum(torch.sqrt(torch.sum((prediction - ground_truth) ** 2, dim=2)), dim=0)
            total_dist = total_dist + dist.cpu()

            progress.update(1)

print(total_dist / total)