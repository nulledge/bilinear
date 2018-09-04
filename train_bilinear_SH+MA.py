import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

import H36M
import MPII
import model
from util import config

hourglass, optimizer, step, train_epoch = model.hourglass.load(config.hourglass.parameter_dir, config.hourglass.device)
criterion = nn.MSELoss()

# train_epoch equals -1 means that training is over
if train_epoch != -1:

    # Reset statistics of batch normalization
    hourglass.reset_statistics()
    hourglass.train()

    train_loader = DataLoader(
        MPII.Dataset(
            root=config.hourglass.data_dir,
            task='train',
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

    # epoch equals -1 means that training is over
    epoch = -1
    torch.save(
        {
            'epoch': epoch,
            'step': step,
            'state': hourglass.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        '{parameter_dir}/{epoch}.save'.format(parameter_dir=config.hourglass.parameter_dir, epoch=epoch)
    )

    del train_loader

data = DataLoader(
    H36M.Dataset(
        data_dir=config.bilinear.data_dir,
        task=H36M.Task.Train,
        position_only=False,
    ),
    batch_size=config.hourglass.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.bilinear.num_workers,
)

bilinear, optimizer, step, train_epoch = model.bilinear.load(config.bilinear.parameter_dir, config.bilinear.device)
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir=config.bilinear.log_dir)

from_MPII_to_H36M = [6, 2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 12, 11, 10]
from_MPII_to_H36M = torch.Tensor(from_MPII_to_H36M).long().to(config.bilinear.device)

hourglass.eval()
bilinear.train()

for epoch in range(train_epoch + 1, train_epoch + 200 + 1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:
        with torch.set_grad_enabled(True):
            for subset, image, _ in data:

                # Learning rate decay
                if config.bilinear.lr_decay.activate and config.bilinear.lr_decay.condition(step):
                    lr = config.bilinear.lr_decay.function(step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                in_camera_space = subset[H36M.Annotation.S]
                center = subset[H36M.Annotation.Center]
                scale = subset[H36M.Annotation.Scale]
                mean = subset[H36M.Annotation.Mean_Of + H36M.Annotation.Part]
                stddev = subset[H36M.Annotation.Stddev_Of + H36M.Annotation.Part]

                in_camera_space = in_camera_space.to(config.bilinear.device)
                image = image.to(config.hourglass.device)
                center = center.to(config.hourglass.device)
                scale = scale.to(config.hourglass.device)
                mean = mean.to(config.bilinear.device)
                stddev = stddev.to(config.bilinear.device)

                with torch.set_grad_enabled(False):
                    output = hourglass(image)
                    output = output[-1]  # Heatmaps from the last stack in batch-channel-height-width shape.

                    n_batch = output.shape[0]

                    pose = torch.argmax(output.view(n_batch, 16, -1), dim=-1)
                    pose = torch.stack([
                        pose % 64,
                        pose // 64,
                    ], dim=-1).float()
                    pose = pose - 32
                    pose = center.view(n_batch, 1, 2) + pose / 64 * scale.view(n_batch, 1, 1) * 200

                pose = pose.to(config.bilinear.device)
                pose = torch.index_select(pose, dim=1, index=from_MPII_to_H36M)

                root = pose[:, 0, :].view(n_batch, 1, 2)
                root_centered = pose - root
                root_removed = root_centered[:, 1:, :]

                normalized = (root_removed - mean) / stddev

                in_image_space = normalized.view(-1, 2 * (17 - 1))
                in_camera_space = in_camera_space.view(-1, 3 * (17 - 1))

                optimizer.zero_grad()
                prediction = bilinear(in_image_space)

                loss = criterion(prediction, in_camera_space)
                loss.backward()

                nn.utils.clip_grad_norm_(bilinear.parameters(), max_norm=1)

                optimizer.step()

                # Too frequent update reduces the performance.
                writer.add_scalar('loss', loss, step)

                progress.set_postfix(loss=float(loss.item()))
                progress.update(1)
                step = step + 1

        torch.save(
            {
                'epoch': epoch,
                'step': step,
                'state': bilinear.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            '{parameter_dir}/{epoch}.save'.format(parameter_dir=config.bilinear.parameter_dir, epoch=epoch)
        )
