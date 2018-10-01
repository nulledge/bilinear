import torch
import torch.nn as nn
import os


class CONFIG:
    nStacks = 8
    nFeatures = 256
    nModules = 1
    nJoints = 16
    nDepth = 4


class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.resSeq = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            skip = self.conv_skip(x)
        else:
            skip = x

        return skip + self.resSeq(x)


class Hourglass(nn.Module):
    def __init__(self, hg_depth, nFeatures):
        super(Hourglass, self).__init__()
        self.hg_depth = hg_depth
        self.nFeatures = nFeatures
        res1list = [ResModule(nFeatures, nFeatures) for _ in range(CONFIG.nModules)]
        res2list = [ResModule(nFeatures, nFeatures) for _ in range(CONFIG.nModules)]
        res3list = [ResModule(nFeatures, nFeatures) for _ in range(CONFIG.nModules)]
        self.res1 = nn.Sequential(*res1list)
        self.res2 = nn.Sequential(*res2list)
        self.res3 = nn.Sequential(*res3list)
        self.subHourglass = None
        self.resWaist = None
        if self.hg_depth > 1:
            self.subHourglass = Hourglass(self.hg_depth - 1, nFeatures)
        else:
            res_waist_list = [ResModule(nFeatures, nFeatures) for _ in range(CONFIG.nModules)]
            self.resWaist = nn.Sequential(*res_waist_list)

    def forward(self, x):
        up = self.res1(x)
        low1 = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        low1 = self.res2(low1)

        if self.hg_depth > 1:
            low2 = self.subHourglass(low1)
        else:
            low2 = self.resWaist(low1)

        low3 = self.res3(low2)

        low = nn.UpsamplingNearest2d(scale_factor=2)(low3)

        return up + low


class MainModel(nn.Module):
    def __init__(self, in_channels=3):
        super(MainModel, self).__init__()

        self.beforeHourglass = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            ResModule(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResModule(128, 128),
            ResModule(128, CONFIG.nFeatures)
        )

        self.hgArray = nn.ModuleList([])
        self.linArray = nn.ModuleList([])
        self.htmapArray = nn.ModuleList([])
        self.llBarArray = nn.ModuleList([])
        self.htmapBarArray = nn.ModuleList([])

        for i in range(CONFIG.nStacks):
            self.hgArray.append(Hourglass(CONFIG.nDepth, CONFIG.nFeatures))
            self.linArray.append(self.lin(CONFIG.nFeatures, CONFIG.nFeatures))
            self.htmapArray.append(nn.Conv2d(CONFIG.nFeatures, CONFIG.nJoints, kernel_size=1, stride=1, padding=0))

        for i in range(CONFIG.nStacks - 1):
            self.llBarArray.append(nn.Conv2d(CONFIG.nFeatures, CONFIG.nFeatures, kernel_size=1, stride=1, padding=0))
            self.htmapBarArray.append(nn.Conv2d(CONFIG.nJoints, CONFIG.nFeatures, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        inter = self.beforeHourglass(x)
        outHeatmap = []

        for i in range(CONFIG.nStacks):
            ll = self.hgArray[i](inter)
            ll = self.linArray[i](ll)
            htmap = self.htmapArray[i](ll)
            outHeatmap.append(htmap)

            if i < CONFIG.nStacks - 1:
                ll_ = self.llBarArray[i](ll)
                htmap_ = self.htmapBarArray[i](htmap)
                inter = inter + ll_ + htmap_

        return outHeatmap

    def lin(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )


def load(device, parameter_dir=None):
    hourglass = MainModel().to(device)
    optimizer = torch.optim.RMSprop(hourglass.parameters(), lr=2.5e-4)
    step = 1

    epoch_to_load = 0
    if parameter_dir is not None:
        for _, _, files in os.walk(parameter_dir):
            for parameter_file in files:
                # The name of parameter file is {epoch}.save
                name, extension = parameter_file.split('.')
                epoch = int(name)

                if epoch > epoch_to_load:
                    epoch_to_load = epoch

    if epoch_to_load != 0:
        parameter_file = '{parameter_dir}/{epoch}.save'.format(parameter_dir=parameter_dir, epoch=epoch_to_load)
        parameter = torch.load(parameter_file)

        hourglass.load_state_dict(parameter['state'])
        optimizer.load_state_dict(parameter['optimizer'])
        step = parameter['step']

    return hourglass, optimizer, step, epoch_to_load
