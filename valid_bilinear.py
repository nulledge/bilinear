import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import H36M
import model.bilinear
from util import config

data = DataLoader(
    H36M.Dataset(
        data_dir=config.bilinear.data_dir,
        task=H36M.Task.Valid,
    ),
    batch_size=config.bilinear.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=config.bilinear.num_workers,
)

bilinear, _, step, train_epoch = model.bilinear.load(config.bilinear.parameter_dir, config.bilinear.device)
bilinear.eval()

total_dist = torch.zeros(17)
total = 0

dim = 3
task = H36M.Task.Train
mean = torch.Tensor(data.dataset.mean[task][dim]).to(config.bilinear.device).view(-1, dim * 17)
stddev = torch.Tensor(data.dataset.stddev[task][dim]).to(config.bilinear.device).view(-1, dim * 17)

with tqdm(total=len(data), desc='%d epoch' % train_epoch) as progress:
    with torch.set_grad_enabled(False):
        for in_image_space, in_camera_space, center, scale, _, _ in data:
            in_image_space = in_image_space.to(config.bilinear.device).view(-1, 2 * 17)
            in_camera_space = in_camera_space.to(config.bilinear.device).view(-1, 3 * 17)

            n_batch = in_image_space.shape[0]
            total = total + n_batch

            # Restore from normalized unit to mm unit
            prediction = bilinear(in_image_space)
            prediction_in_camera_space = stddev * prediction + mean
            in_camera_space = stddev * in_camera_space + mean

            # # Root-centered align
            ground_truth = in_camera_space.view(-1, 17, 3)
            root = ground_truth[:, 0].view(-1, 1, 3)
            ground_truth = ground_truth - root

            prediction = prediction_in_camera_space.view(-1, 17, 3)
            root = prediction[:, 0].view(-1, 1, 3)
            prediction = prediction - root

            # Compute distance between ground-truth and prediction
            dist = torch.sum(torch.sqrt(torch.sum((prediction - ground_truth) ** 2, dim=2)), dim=0)
            total_dist = total_dist + dist.cpu()

            progress.update(1)

print(total_dist / total)
print(sum(total_dist) / (total * len(total_dist)))
