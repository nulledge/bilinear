import torch
import torch.nn as nn
import numpy as np
import os


def light_conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, momentum=None),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
    )


def heavy_conv(in_channels, out_channels):
    return nn.Sequential(
        light_conv(in_channels, out_channels // 2, kernel_size=1),
        light_conv(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
        light_conv(out_channels // 2, out_channels, kernel_size=1),
    )


class IdentityUnit(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        out_tensor = in_tensor

        return out_tensor


class ResUnit(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(ResUnit, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.conv = heavy_conv(in_channels, out_channels)

        self.skip = IdentityUnit()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, in_tensor):
        out_tensor = self.conv(in_tensor)
        out_tensor = out_tensor + self.skip(in_tensor)

        return out_tensor


class Hourglass(nn.Module):
    def __init__(self, in_channels, compression_time):
        super(Hourglass, self).__init__()

        self.skip_connection = nn.ModuleList([
            ResUnit(in_channels) for _ in range(compression_time)
        ])
        self.downscale = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(2, 2),
                ResUnit(in_channels),
            ) for _ in range(compression_time)
        ])
        self.res = ResUnit(in_channels)
        self.upscale = nn.ModuleList([
            nn.Sequential(
                ResUnit(in_channels),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ) for _ in range(compression_time)
        ])

    def forward(self, in_tensor):
        out_tensor = in_tensor
        skip_tensor = list()

        for skip_connection, downscale in zip(self.skip_connection, self.downscale):
            skip_tensor.append(skip_connection(out_tensor))
            out_tensor = downscale(out_tensor)

        out_tensor = self.res(out_tensor)

        for skip_tensor, upscale in zip(reversed(skip_tensor), self.upscale):
            out_tensor = upscale(out_tensor) + skip_tensor

        return out_tensor


class StackedHourglass(nn.Module):
    def __init__(self, stacks, joints, out_channels=256, compression_time=4):
        super(StackedHourglass, self).__init__()

        self.stacks = stacks
        self.joints = joints
        self.out_channels = out_channels
        self.compression_time = compression_time

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            # nn.BatchNorm2d(num_features=64, momentum=None),
            # nn.ReLU(),
            ResUnit(in_channels=64, out_channels=128),
            nn.MaxPool2d(2),
            ResUnit(in_channels=128, out_channels=128),
            ResUnit(in_channels=128, out_channels=self.out_channels),
        )
        self.hourglass = nn.ModuleList([
            Hourglass(in_channels=self.out_channels, compression_time=self.compression_time) for _ in range(self.stacks)
        ])
        self.prev_heatmap = nn.ModuleList([
            nn.Sequential(
                ResUnit(self.out_channels, self.out_channels),
                light_conv(self.out_channels, self.out_channels, kernel_size=1),
            ) for _ in range(self.stacks)
        ])
        self.heatmap_intermediate = nn.ModuleList([
            light_conv(self.out_channels, self.joints, kernel_size=1, bias=True) for _ in range(self.stacks)
        ])
        self.after_heatmap = nn.ModuleList([
            light_conv(self.joints, self.out_channels, kernel_size=1) for _ in range(self.stacks)
        ])
        self.skip_intermediate = nn.ModuleList([
            light_conv(self.out_channels, self.out_channels, kernel_size=1) for _ in range(self.stacks)
        ])

    def forward(self, in_tensor):
        out_tensor = in_tensor
        out_heatmap = list()
        stack = zip(
            self.hourglass,
            self.prev_heatmap,
            self.heatmap_intermediate,
            self.after_heatmap,
            self.skip_intermediate,
        )

        out_tensor = self.feature_extraction(out_tensor)
        for hourglass, prev, heatmap, after, skip in stack:
            prev_tensor = out_tensor
            out_tensor = hourglass(out_tensor)
            out_tensor = prev(out_tensor)
            skip_tensor = skip(out_tensor)
            prediction = heatmap(out_tensor)
            out_tensor = after(prediction) + skip_tensor + prev_tensor

            out_heatmap.append(prediction.unsqueeze(0))

        return torch.cat(out_heatmap, 0)


def load_model(device, pretrained):
    hourglass = StackedHourglass(stacks=8, joints=16)
    step = np.zeros([1], dtype=np.uint32)

    pretrained_epoch = 0
    eval_data = False

    for _, _, files in os.walk(pretrained):
        for file in files:
            name, extension = file.split('.')
            epoch = int(name)
            if epoch > pretrained_epoch:
                pretrained_epoch = epoch
            if epoch == -1:
                eval_data = True

    if pretrained_epoch > 0:
        if eval_data:
            pretrained_epoch = -1

        pretrained_model = os.path.join(
            '{pretrained}/{epoch}.save'.format(pretrained=pretrained, epoch=pretrained_epoch))
        pretrained_model = torch.load(pretrained_model)

        hourglass.load_state_dict(pretrained_model['state'])
        step[0] = pretrained_model['step']

    else:
        pretrained_model = None

    hourglass = hourglass.to(device)
    optimizer = torch.optim.RMSprop(hourglass.parameters(), lr=2.5e-4)
    if pretrained_model is not None:
        optimizer.load_state_dict(pretrained_model['optimizer'])
    criterion = nn.MSELoss()

    return hourglass, optimizer, criterion, step, pretrained_epoch
