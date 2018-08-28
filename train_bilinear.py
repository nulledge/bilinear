import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

import H36M
import model.bilinear
from util import config

data = DataLoader(
    H36M.Dataset(
        data_dir=config.bilinear.data_dir,
        task='train',
    ),
    batch_size=config.bilinear.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.bilinear.num_workers,
)

bilinear, optimizer, step, train_epoch = model.bilinear.load(config.bilinear.parameter_dir, config.bilinear.device)
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir=config.bilinear.log_dir)

bilinear.train()

for epoch in range(train_epoch+ 1, train_epoch + 200 + 1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:
        with torch.set_grad_enabled(True):
            for in_image_space, in_camera_space, center, scale, _, _ in data:

                # Learning rate decay
                if step % 100000 == 0 or step == 1:
                    lr = 1.0e-3 * 0.96 ** (step / 100000)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                in_image_space = in_image_space.to(config.bilinear.device).view(-1, 2 * 17)
                in_camera_space = in_camera_space.to(config.bilinear.device).view(-1, 3 * 17)

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
