import numpy as np
import torch
import torch.nn as nn
import imageio
import skimage.transform
from torch.utils.data import DataLoader
from tqdm import tqdm

import MPII
import Model.hourglass
from Model.end2end import softargmax
from util.visualize import merge_to_color_heatmap
from util import config

assert config.task == 'valid'

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


mean_and_var = [
    float(hourglass.hourglass[1].res.conv[0][0].running_mean.max()),
    float(hourglass.hourglass[1].res.conv[0][0].running_var.max()),
    float(hourglass.hourglass[7].upscale[0][0].conv[0][0].running_mean.max()),
    float(hourglass.hourglass[7].upscale[0][0].conv[0][0].running_var.max()),
]
print('before reset:', mean_and_var)

for key, value in hourglass.state_dict().items():
    if 'running_mean' in key:

        layer = hourglass
        modules = key.split('.')[:-1]
        for module in modules:
            if module.isdigit():
                layer = layer[int(module)]
            else:
                layer = getattr(layer, module)
        layer.reset_running_stats()
        layer.momentum = None


mean_and_var = [
    float(hourglass.hourglass[1].res.conv[0][0].running_mean.max()),
    float(hourglass.hourglass[1].res.conv[0][0].running_var.max()),
    float(hourglass.hourglass[7].upscale[0][0].conv[0][0].running_mean.max()),
    float(hourglass.hourglass[7].upscale[0][0].conv[0][0].running_var.max()),
]
print('after reset', mean_and_var)

train_loader = DataLoader(
    MPII.Dataset(
        root=config.root['MPII'],
        task='train',
        augment=False,
    ),
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.num_workers,
)

hourglass.train()

with tqdm(total=len(train_loader), desc='%d epoch' % pretrained_epoch) as progress:

    with torch.set_grad_enabled(False):

        for images, heatmaps, keypoints in train_loader:
            images_cpu = images
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            optimizer.zero_grad()
            outputs = hourglass(images)

            progress.update(1)

hourglass = hourglass.eval()

total = torch.zeros(hourglass.joints).cuda()
hit = torch.zeros(hourglass.joints).cuda()

mean_and_var = [
    float(hourglass.hourglass[1].res.conv[0][0].running_mean.max()),
    float(hourglass.hourglass[1].res.conv[0][0].running_var.max()),
    float(hourglass.hourglass[7].upscale[0][0].conv[0][0].running_mean.max()),
    float(hourglass.hourglass[7].upscale[0][0].conv[0][0].running_var.max()),
]
print('after train:', mean_and_var)

with tqdm(total=len(data), desc='%d epoch' % pretrained_epoch) as progress:
    with torch.set_grad_enabled(False):
        for images, heatmaps, keypoints in data:

            images_device = images.to(device)

            outputs = hourglass(images_device)
            outputs = outputs[-1]  # Heatmaps from the last stack in batch-channel-height-width shape.

            n_batch = outputs.shape[0]

            # joint_map = [13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0, 6, 7, 8, 9]
            joint_map = [x for x in range(16)]

            for batch in range(n_batch):
                for joint, heatmap in enumerate(outputs[batch]):

                    # The empty heatmap means not-annotated.
                    if np.count_nonzero(heatmaps[batch][joint]) == 0:
                        continue

                    total[joint] = total[joint] + 1

                    pose = torch.argmax(heatmap.view(-1))
                    pose = torch.Tensor([int(pose) % 64, int(pose) // 64])

                    dist = pose - keypoints[batch][joint_map[joint]]
                    dist = torch.sqrt(torch.sum(dist * dist))

                    if torch.le(dist, 64 * 0.1):
                        hit[joint] = hit[joint] + 1

            progress.update(1)

print(hit)
print(total)
print(hit / total * 100)  # In percentage.
