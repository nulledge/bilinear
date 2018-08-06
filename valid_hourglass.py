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


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


data = DataLoader(
    MPII.Dataset(
        root=config.root['MPII'],
        task=config.task,
    ),
    batch_size=config.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=config.num_workers,
)

device = torch.device(config.device)
hourglass, _, _, _, pretrained_epoch = Model.hourglass.load_model(device, config.pretrained['hourglass'])

hit = np.zeros(shape=(16,), dtype=np.uint32)
total = np.zeros(shape=(16,), dtype=np.uint32)

with tqdm(total=len(data), desc='%d epoch' % pretrained_epoch) as progress:
    with torch.set_grad_enabled(False):
        for images, heatmaps, keypoints in data:

            images_cpu = images
            images_device = images.to(device)

            outputs = hourglass(images_device)
            outputs = outputs[-1]  # Heatmaps from the last stack in batch-channel-height-width shape.

            n_batch = outputs.shape[0]

            joint_map = [13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0, 6, 7, 8, 9]
            poses_2D = torch.zeros(n_batch, 16, 2)  # MPII has 16 joints.

            for batch in range(n_batch):
                image_soft = np.asarray(images[batch].data)

                diff = torch.zeros(16, 2)
                valid = np.zeros(16)

                for joint, heatmap in enumerate(outputs[batch]):
                    poses_2D[batch, joint, :] = softargmax(heatmap)
                    diff[joint] = poses_2D.data[batch][joint] - keypoints[batch][joint_map[joint]]

                    if np.count_nonzero(heatmaps[batch][joint]) != 0:
                        total[joint] = total[joint] + 1
                        valid[joint] = True

                dist = torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)  # shape=(16)

                for joint in torch.nonzero(torch.le(dist, 64 * 0.1)):
                    if not valid[joint]:
                        continue
                    hit[joint] = hit[joint] + 1

            progress.update(1)

            if False:
                heatmaps = merge_to_color_heatmap(outputs)
                heatmaps = heatmaps.permute(0, 2, 3, 1).cpu()  # NHWC

                resized_heatmaps = list()
                for idx, ht in enumerate(heatmaps):
                    color_ht = skimage.transform.resize(ht.numpy(), (256, 256), mode='constant')
                    resized_heatmaps.append(color_ht.transpose(2, 0, 1))

                resized_heatmaps = np.stack(resized_heatmaps, axis=0)

                images = np.asarray(images_cpu).transpose(0, 2, 3, 1) * 0.6
                heatmaps = np.asarray(resized_heatmaps).transpose(0, 2, 3, 1) * 0.4
                overlayed_image = np.clip(images + heatmaps, 0, 1.)

                for idx, image in enumerate(overlayed_image):
                    # for joint in range(16):
                    #     x, y = poses_2D[idx][joint] * 4
                    #     for tx in range(-5, 5):
                    #         for ty in range(-5, 5):
                    #             xx, yy = (x + tx, y + ty)
                    #             if not (0 <= xx <= 255) or not (0 <= yy <= 255):
                    #                 continue
                    #             image[int(yy), int(xx), :] = [1, 0, 0]
                    imageio.imwrite('{idx}.jpg'.format(idx=idx), image)

                break

print(hit / total * 100)  # In percentage.
