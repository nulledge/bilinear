import torch
import torch.nn as nn
import numpy as np
import os


# convolutional block: full pre-activation
def conv_block(in_channel, out_channel, last_bias=False):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel // 2, 1, bias=False),
        nn.BatchNorm2d(out_channel // 2),
        nn.ReLU(),
        nn.Conv2d(out_channel // 2, out_channel // 2, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channel // 2),
        nn.ReLU(),
        nn.Conv2d(out_channel // 2, out_channel, 1, bias=last_bias))


# covlutional 1x1 block
def conv1_block(in_channel, out_channel, is_bias=False):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1, bias=is_bias))


# residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv_block = conv_block(in_channels, out_channels)

        self.skip_layer = None
        if in_channels != out_channels:
            self.skip_layer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv_block(x)
        out = torch.add(out, self.skip_layer(x) if self.skip_layer is not None else x)
        return out


# hourglass
class Hourglass(nn.Module):
    def __init__(self, size, channels):
        super(Hourglass, self).__init__()
        self.size = size
        self.up_layers = nn.ModuleList([ResBlock(channels) for _ in range(self.size)])
        self.low1_layers = nn.ModuleList([ResBlock(channels) for _ in range(self.size)])
        self.low2 = ResBlock(channels)
        self.low3_layers = nn.ModuleList([ResBlock(channels) for _ in range(self.size)])
        self.max_pool = nn.MaxPool2d(2, 2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        l1 = x

        up1_outputs = list()
        for up, low1 in zip(self.up_layers, self.low1_layers):
            u = up(l1)
            up1_outputs.append(u)
            l1 = self.max_pool(l1)
            l1 = low1(l1)

        out = self.low2(l1)

        for up1_out, low3 in zip(reversed(up1_outputs), reversed(self.low3_layers)):
            out = up1_out + self.up_sample(low3(out))

        return out


class StackedHourglass(nn.Module):
    def __init__(self, num_stacks, features, joints, internal_size=4):
        super(StackedHourglass, self).__init__()
        self.num_stacks = num_stacks
        self.features = features
        self.joints = joints
        self.internale_size = internal_size

        # initial processing
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # 128
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            ResBlock(64, 128),
            nn.MaxPool2d(2),  # 64
            ResBlock(128, 128),
            ResBlock(128, self.features))

        # residual layer
        self.hourglasses = nn.ModuleList([Hourglass(self.internale_size, self.features) for _ in range(num_stacks)])
        self.prev_intermediate_layers = nn.ModuleList([
            nn.Sequential(ResBlock(self.features, self.features),
                          conv1_block(self.features, self.features)) for _ in range(num_stacks)])
        self.heat_intermediate_layers = nn.ModuleList(
            [conv1_block(self.features, self.joints, is_bias=True) for _ in range(num_stacks)])
        self.after_intermediate_layers = nn.ModuleList(
            [conv1_block(self.joints, self.features) for _ in range(num_stacks - 1)])
        self.skip_intermediate_layers = nn.ModuleList(
            [conv1_block(self.features, self.features) for _ in range(num_stacks - 1)])

    def forward(self, x):  # x dim: [BCHW]
        init_conv = self.init_conv(x)

        inter = init_conv
        out_heatmaps = None
        layers = zip(self.hourglasses,
                     self.prev_intermediate_layers,
                     self.heat_intermediate_layers,
                     self.after_intermediate_layers,
                     self.skip_intermediate_layers)
        for hg, prev, heat, after, skip in layers:  # loop over (num_stack - 1)
            prev_inter = inter

            hg = hg(inter)
            prev = prev(hg)
            heat = heat(prev)  # [BJHW], J: #joints
            after = after(heat)
            skip = skip(prev)

            inter = prev_inter + after + skip

            heatmap = heat.unsqueeze(0)  # ['1'BJWH]

            # stacking intermediate heatmaps for intermediate supervision
            if out_heatmaps is not None:
                out_heatmaps = torch.cat([out_heatmaps, heatmap], 0)  # [SBJHW], S: #stacks
            else:
                out_heatmaps = heatmap

        hg = self.hourglasses[-1](inter)
        prev = self.prev_intermediate_layers[-1](hg)
        heat = self.heat_intermediate_layers[-1](prev)
        heatmap = heat.unsqueeze(0)  # ['1'BJHW]
        out_heatmaps = torch.cat([out_heatmaps, heatmap], 0)  # [SBJHW]

        return out_heatmaps


def load_model(device, pretrained):
    hourglass = StackedHourglass(num_stacks=8, features=256, joints=16)
    step = np.zeros([1], dtype=np.uint32)

    pretrained_epoch = 0
    for _, _, files in os.walk(pretrained):
        for file in files:
            name, extension = file.split('.')
            epoch = int(name)
            if epoch > pretrained_epoch:
                pretrained_epoch = epoch

    if pretrained_epoch > 0:
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
