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
total = 0

with tqdm(total=len(data), desc='%d epoch' % pretrained_epoch) as progress:
    with torch.set_grad_enabled(False):
        for images, _, keypoints in data:
            images_cpu = images
            images = images.to(device)

            outputs = hourglass(images)

            heatmaps = merge_to_color_heatmap(outputs[-1].data)
            heatmaps = heatmaps.permute(0, 2, 3, 1).cpu()  # NHWC

            resized_heatmaps = list()
            for idx, ht in enumerate(heatmaps):
                color_ht = skimage.transform.resize(ht.numpy(), (256, 256), mode='constant')
                resized_heatmaps.append(color_ht.transpose(2, 0, 1)) # CHW

            resized_heatmaps = np.stack(resized_heatmaps, axis=0)

            overlayed_image = np.clip(np.asarray(images.data) * 0.6 + resized_heatmaps * 0.4, 0, 1.)

            for idx, image in enumerate(overlayed_image):
                imageio.imwrite('prediction_{idx}.jpg'.format(idx=idx), image.transpose(1, 2, 0))

            outputs = outputs[-1]
            n_batch = outputs.shape[0]
            total = total + n_batch

            poses_2D = torch.zeros(n_batch, 16, 2)
            for batch in range(n_batch):
                image_soft = np.asarray(images[batch].data)
                image_hard = np.asarray(images[batch].data)
                for joint, heatmap in enumerate(outputs[batch]):
                    poses_2D[batch][joint] = softargmax(heatmap)

                    for tx in range(-5, 5):
                        for ty in range(-5, 5):
                            image_soft[:, int(poses_2D[batch][joint][1])*4 + ty, int(poses_2D[batch][joint][0])*4 + tx] = [1, 0, 0]

                    pose = int(torch.argmax(heatmap.view(-1)).data)
                    poses_2D[batch][joint][:] = torch.Tensor([pose % 64, pose // 64])

                    for tx in range(-5, 5):
                        for ty in range(-5, 5):
                            image_hard[:, int(poses_2D[batch][joint][1])*4 + ty, int(poses_2D[batch][joint][0])*4 + tx] = [1, 0, 0]
                imageio.imwrite('soft_argmax{idx}.jpg'.format(idx=batch), image_soft.transpose(1, 2, 0))
                imageio.imwrite('argmax{idx}.jpg'.format(idx=batch), image_hard.transpose(1, 2, 0))

                # diff = poses_2D.data[batch] - keypoints[batch]  # shape=(16, 2)
                # dist = torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)  # shape=(16)
                #
                # for joint in torch.nonzero(torch.le(dist, 64 * 0.1)):
                #     hit[joint] = hit[joint] + 1

            progress.update(1)

            break

print(total)
print(hit)
print(hit / total * 100)  # In percentage.
