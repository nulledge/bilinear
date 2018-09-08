import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

import MPII
import model.hourglass
from util import config
from util.visualize import colorize, overlap
from util.log import get_logger

time_stamp_to_load = None
assert time_stamp_to_load is not None

logger, log_dir, time_stamp = get_logger(time_stamp=time_stamp_to_load)

hourglass, optimizer, step, train_epoch = model.hourglass.load(
    device=config.hourglass.device,
    parameter_dir='{log_dir}/parameter'.format(log_dir=log_dir),
)
criterion = nn.MSELoss()

# Reset statistics of batch normalization
hourglass.reset_statistics()
hourglass.train()

train_loader = DataLoader(
    MPII.Dataset(
        root=config.hourglass.data_dir,
        task=MPII.Task.Train,
        augment=False,
    ),
    batch_size=config.hourglass.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.hourglass.num_workers,
)

# Compute statistics of batch normalization from the train subset
with tqdm(total=len(train_loader), desc='%d epoch' % train_epoch) as progress:
    with torch.set_grad_enabled(False):
        for images, _, _, _, _, _ in train_loader:
            images = images.to(config.hourglass.device)
            outputs = hourglass(images)

            progress.update(1)

hourglass = hourglass.eval()

valid_data = DataLoader(
    MPII.Dataset(
        root=config.hourglass.data_dir,
        task=MPII.Task.Valid,
        augment=False,
    ),
    batch_size=config.hourglass.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.hourglass.num_workers,
)

total = torch.zeros((14,)).int()
hit = torch.zeros((14,)).int()


writer = SummaryWriter(log_dir='{log_dir}/visualize'.format(
    log_dir=log_dir,
))
resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=[256, 256]),
    transforms.ToTensor(),
])
upscale = lambda heatmaps: torch.stack([resize(heatmap) for heatmap in heatmaps.cpu()]).to(config.hourglass.device)

with tqdm(total=len(valid_data), desc='%d epoch' % train_epoch) as progress:
    with torch.set_grad_enabled(False):
        for images, heatmaps, keypoints, centers, scales, heads in valid_data:
            images = images.to(config.hourglass.device)
            heatmaps = heatmaps.to(config.hourglass.device)
            centers = centers.to(config.hourglass.device).float()
            scales = scales.to(config.hourglass.device).float()

            outputs = hourglass(images)
            outputs = outputs[-1]  # Heatmaps from the last stack in batch-channel-height-width shape.

            n_batch = outputs.shape[0]

            poses = torch.argmax(outputs.view(n_batch, 16, -1), dim=-1)
            poses = torch.stack([
                poses % 64,
                poses // 64,
            ], dim=-1).float()
            poses = poses - 32
            poses = centers.view(n_batch, 1, 2) + poses / 64 * scales.view(n_batch, 1, 1) * 200

            if step % 10 == 0:
                ground_truth = overlap(images=images, heatmaps=upscale(colorize(heatmaps)))
                prediction = overlap(images=images, heatmaps=upscale(colorize(outputs[-1])))

                writer.add_image('{time_stamp}/val/ground-truth'.format(time_stamp=time_stamp), ground_truth.data, step)
                writer.add_image('{time_stamp}/val/prediction'.format(time_stamp=time_stamp), prediction.data, step)

            dists = poses - keypoints.to(config.hourglass.device).float()
            dists = torch.sqrt(torch.sum(dists * dists, dim=-1))

            PCKh_temp = dists / heads.view(n_batch, 1).to(config.hourglass.device).float()
            PCKh_pred = torch.zeros((n_batch, 14,))

            PCKh_pred[:, 0:6] = PCKh_temp[:, 0:6]
            PCKh_pred[:, 6:12] = PCKh_temp[:, 10:16]
            PCKh_pred[:, 12:14] = PCKh_temp[:, 8:10]

            temp = (PCKh_pred <= 0.5).float()

            total = total + torch.sum((~torch.isnan(PCKh_pred)), dim=0).int()
            hit = hit + torch.sum(temp, dim=0).int()

            progress.update(1)
            step = step + 1

hit = hit.float()
total = total.float()
for idx, joint in enumerate(MPII.keypoints):
    logger.info('{joint}: {PCKh}'.format(joint=joint, PCKh=(hit / total * 100)))

logger.info('===========================================================')
writer.close()
