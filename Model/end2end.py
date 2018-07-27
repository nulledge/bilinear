import torch
import torch.nn as nn
import numpy as np

from Model import hourglass, bilinear
from util import config


def softargmax(in_tensor):
    keypoint = np.full(shape=(2,), fill_value=-1)
    for dim in [0, 1]:
        data_along_axis = 10 * torch.sum(in_tensor, dim=dim)
        softmax = nn.Softmax(dim=0)(data_along_axis)
        soft_argmax = torch.sum(softmax * torch.arange(0, 64).cuda())

        keypoint[dim] = soft_argmax

    return torch.Tensor(keypoint)


class End2End(nn.Module):
    def __init__(self):
        super().__init__()

        device = torch.device(config.device)

        self.criterion = {}
        self.hourglass, self.opt_hg, self.loss_hg, self.step_hg, self.epoch_hg = \
            hourglass.load_model(device, config.pretrained['hourglass'])
        self.bilinear, self.opt_bi, self.loss_bi, self.step_bi, self.epoch_bi, self.lr = \
            bilinear.load_model(device, config.pretrained['bilinear'])

    def __forward__(self, in_tensor):
        out_tensor = in_tensor

        heatmaps = self.hourglass(out_tensor)
        poses_2D = torch.zeros(17, 2)
        for idx, heatmap in enumerate(heatmaps):
            poses_2D[idx] = softargmax(heatmap)

        poses_3D = self.bilinear(poses_2D)

        return heatmaps, poses_3D, self.opt_hg, self.opt_bi, self.loss_hg, self.loss_bi
