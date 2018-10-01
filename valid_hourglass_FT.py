import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

import H36M
import model.hourglass
from util import config
from util.visualize import colorize, overlap
from util.log import get_logger

assert config.hourglass.comment is not None
logger, log_dir, comment = get_logger(comment=config.hourglass.comment)

hourglass, optimizer, step, train_epoch = model.hourglass.load(
    device=config.hourglass.device,
    parameter_dir='{log_dir}/parameter'.format(log_dir=log_dir),
)
criterion = nn.MSELoss()

# Reset statistics of batch normalization
hourglass.reset_statistics()
hourglass.train()

train_loader = DataLoader(
    H36M.Dataset(
        data_dir=config.bilinear.data_dir,
        task=H36M.Task.Train,
        protocol=H36M.Protocol.GT,
        position_only=False,
    ),
    batch_size=config.hourglass.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.hourglass.num_workers,
)

# Compute statistics of batch normalization from the train subset
with tqdm(total=len(train_loader), desc='%d epoch' % train_epoch) as progress:
    with torch.set_grad_enabled(False):
        for _, images, _, _ in train_loader:
            images = images.to(config.hourglass.device)
            outputs = hourglass(images)

            progress.update(1)

del train_loader

hourglass = hourglass.eval()

valid_data = DataLoader(
    H36M.Dataset(
        data_dir=config.bilinear.data_dir,
        task=H36M.Task.Valid,
        protocol=H36M.Protocol.GT,
        position_only=False,
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
        for subset, images, heatmaps, action in valid_data:
            images = images.to(config.hourglass.device)
            heatmaps = heatmaps.to(config.hourglass.device)
            centers = centers.to(config.hourglass.device).float()
            scales = scales.to(config.hourglass.device).float()

            outputs = hourglass(images)
            outputs = outputs[-1]  # Heatmaps from the last stack in batch-channel-height-width shape.

            flip_images = images.flip(3).to(config.hourglass.device)
            flip_outputs = hourglass(flip_images)
            flip_outputs = flip_outputs[-1]

            swap = torch.Tensor([5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]).long().to(config.hourglass.device)
            flip_outputs = torch.index_select(flip_outputs, 1, swap)
            flip_outputs = flip_outputs.flip(3).to(config.hourglass.device)

            outputs = (outputs + flip_outputs)/2

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
                prediction = overlap(images=images, heatmaps=upscale(colorize(outputs)))

                writer.add_image('{comment}/val/ground-truth'.format(comment=config.hourglass.comment), ground_truth.data, step)
                writer.add_image('{comment}/val/prediction'.format(comment=config.hourglass.comment), prediction.data, step)

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
PCKh = hit / total * 100
reordered = MPII.keypoints[0:6] + MPII.keypoints[10:16] + MPII.keypoints[8:10]

logger.info('===========================================================')
for idx, joint in enumerate(reordered):
    logger.info('{joint}: {PCKh}'.format(joint=joint, PCKh=PCKh[idx]))
logger.info('avg: {PCKh}'.format(PCKh=torch.sum(hit) / torch.sum(total) * 100))
logger.info('===========================================================')

writer.close()
