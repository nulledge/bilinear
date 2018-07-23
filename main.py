import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import MPII
from Model import load_model
from util.path import safe_path
from util.visualize import draw
from util import config

data = DataLoader(
    MPII.Dataset(
        root=config.root,
        task=config.task,
    ),
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.num_workers,
)

device = torch.device(config.device)
hourglass, optimizer, criterion, step, pretrained_epoch = load_model(device)

loss_window, gt_image_window, out_image_window = None, None, None
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

                optimizer.step()

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
        safe_path('./pretrained/{epoch}.save'.format(epoch=epoch))
    )
