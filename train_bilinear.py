import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import H36M
import Model.bilinear
from util.visualize import draw_line
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

device = torch.device(config.device)
bilinear, optimizer, criterion, step, pretrained_epoch, lr = Model.bilinear.load_model(device,
                                                                                   config.pretrained['bilinear'])

loss_window = None
windows = [loss_window]

for epoch in range(pretrained_epoch + 1, pretrained_epoch + 100 + 1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:

        with torch.set_grad_enabled(True):
            for in_image_space, in_camera_space in data:

                if step % 100000 == 0 or step == 1:
                    lr = 1.0e-3 * 0.96 ** (step / 100000)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                in_image_space = in_image_space.to(device).view(-1, 2 * 17)
                in_camera_space = in_camera_space.to(device).view(-1, 3 * 17)

                optimizer.zero_grad()
                prediciton = bilinear(in_image_space)

                loss = criterion(prediciton, in_camera_space)
                loss.backward()

                nn.utils.clip_grad_norm_(bilinear.parameters(), max_norm=1)

                optimizer.step()

                if step % 50 == 0:
                    loss_window = draw_line(x=np.asarray([step]),
                                            y=np.array([float(loss.data)]),
                                            window=loss_window)

                progress.set_postfix(loss=float(loss.item()))
                progress.update(1)
                step = step + 1

    torch.save(
        {
            'epoch': epoch,
            'step': step,
            'state': bilinear.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr,
        },
        '{pretrained}/{epoch}.save'.format(pretrained=config.pretrained['bilinear'], epoch=epoch)
    )
