import numpy as np
import torch
import torch.nn as nn
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm

import MPII
import Model.hourglass
from util.visualize import draw, draw_line
from util import config

assert config.task == 'train'

data = DataLoader(
    MPII.Dataset(
        root=config.root['MPII'],
        task=config.task,
    ),
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.num_workers,
)

device = torch.device(config.device)
hourglass, optimizer, criterion, step, pretrained_epoch = Model.hourglass.load_model(device,
                                                                                     config.pretrained['hourglass'])

hourglass.train()

loss_window, gt_image_window, out_image_window = None, None, None
max_mean_1_window, max_var_1_window = None, None
max_mean_2_window, max_var_2_window = None, None
windows = [loss_window, gt_image_window, out_image_window]

for epoch in range(pretrained_epoch + 1, pretrained_epoch + 100 + 1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:

        with torch.set_grad_enabled(True):

            for images, heatmaps, keypoints in data:
                images_cpu = images
                images = images.to(device)
                heatmaps = heatmaps.to(device)

                optimizer.zero_grad()
                outputs = hourglass(images)

                loss = sum([criterion(output, heatmaps) for output in outputs])
                loss.backward()

                nn.utils.clip_grad_norm_(hourglass.parameters(), max_norm=1)

                optimizer.step()

                mean_and_var = [
                    float(hourglass.hourglass[1].res.conv[0][0].running_mean.max()),
                    float(hourglass.hourglass[1].res.conv[0][0].running_var.max()),
                    float(hourglass.hourglass[7].upscale[0][0].conv[0][0].running_mean.max()),
                    float(hourglass.hourglass[7].upscale[0][0].conv[0][0].running_var.max()),
                ]

                max_mean_1_window = draw_line(x=step,
                                              y=np.array([mean_and_var[0]]),
                                              window=max_mean_1_window)
                max_var_1_window = draw_line(x=step,
                                             y=np.array([mean_and_var[1]]),
                                             window=max_var_1_window)
                max_mean_2_window = draw_line(x=step,
                                              y=np.array([mean_and_var[2]]),
                                              window=max_mean_2_window)
                max_var_2_window = draw_line(x=step,
                                             y=np.array([mean_and_var[3]]),
                                             window=max_var_2_window)

                if config.visualize:
                    windows = draw(step, loss, images_cpu, heatmaps, outputs, windows)

                progress.set_postfix(loss=float(loss.item()))
                progress.update(1)
                step = step + 1

    torch.save(
        {
            'epoch': epoch,
            'step': step,
            'state': hourglass.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        '{pretrained}/{epoch}.save'.format(pretrained=config.pretrained['hourglass'], epoch=epoch)
    )
