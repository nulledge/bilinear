import torch
import torch.nn as nn
import torch.nn.functional as F


# convolutional block: full pre-activation
def conv_block(in_channel, out_channel, last_bias=False):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel, momentum=None),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel // 2, 1, bias=False),
        nn.BatchNorm2d(out_channel // 2, momentum=None),
        nn.ReLU(),
        nn.Conv2d(out_channel // 2, out_channel // 2, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channel // 2, momentum=None),
        nn.ReLU(),
        nn.Conv2d(out_channel // 2, out_channel, 1, bias=last_bias))


# covlutional 1x1 block
def conv1_block(in_channel, out_channel, is_bias=False):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel, momentum=None),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1, bias=is_bias))


def identity(x):
    return x


# residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv_block = conv_block(in_channels, out_channels)

        self.skip_layer = identity
        if in_channels != out_channels:
            self.skip_layer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv_block(x)
        out = torch.add(out, self.skip_layer(x))  # if self.skip_layer is not None else x)

        return out


class ConcatBlocks(nn.Module):
    def __init__(self, size, in_channels, out_channels, feature):
        if size < 2:
            raise ValueError("size of ConcatBlocks must be greater than 2")

        super(ConcatBlocks, self).__init__()

        self.layer = nn.ModuleList([ResBlock(in_channels, feature)])
        for _ in range(size - 2):
            self.layer.append(ResBlock(feature, feature))
        self.layer.append(ResBlock(feature, out_channels))

    def forward(self, x):
        for ly in self.layer:
            x = ly(x)

        return x


# hourglass
class Hourglass(nn.Module):
    def __init__(self, size, in_channels, out_channels, feature=256):
        super(Hourglass, self).__init__()
        self.size = size

        self.up_layers = nn.ModuleList([ConcatBlocks(3, in_channels, out_channels, feature)])
        for _ in range(self.size - 1):
            self.up_layers.append(ConcatBlocks(3, feature, out_channels, feature))

        self.low1_layers = nn.ModuleList([ConcatBlocks(3, in_channels, feature, feature)])
        for _ in range(self.size - 1):
            self.low1_layers.append(ConcatBlocks(3, feature, feature, feature))

        self.low2 = ResBlock(feature, out_channels)
        self.low3_layers = nn.ModuleList([ResBlock(out_channels, out_channels) for _ in range(self.size)])

        self.max_pool = nn.MaxPool2d(2, 2)
        self.up_sample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')

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


class CoarseToFine(nn.Module):
    def __init__(self, voxel_z_resolutions, num_parts, internal_size=4):
        super(CoarseToFine, self).__init__()
        self.voxel_z_resolutions = voxel_z_resolutions
        self.num_parts = num_parts
        self.internale_size = internal_size

        # initial processing
        self.init_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # 128
            ResBlock(64, 128),
            nn.MaxPool2d(2))  # 64
        self.init_conv2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256))

        # residual layer
        self.hourglasses = nn.ModuleList([
            Hourglass(self.internale_size, in_channels=256, out_channels=512),
            Hourglass(self.internale_size, in_channels=256 + 128, out_channels=512),
            Hourglass(self.internale_size, in_channels=256 + 256, out_channels=512),
            Hourglass(self.internale_size, in_channels=256 + 256, out_channels=512)])

        self.prev_intermediate_layers = nn.ModuleList([
            nn.Sequential(conv1_block(512, 512), conv1_block(512, 256)),
            nn.Sequential(conv1_block(512, 512), conv1_block(512, 256)),
            nn.Sequential(conv1_block(512, 512), conv1_block(512, 256)),
            nn.Sequential(conv1_block(512, 512), conv1_block(512, 512))])

        self.voxel_intermediate_layers = nn.ModuleList([
            conv1_block(256, self.num_parts * self.voxel_z_resolutions[0], is_bias=True),
            conv1_block(256, self.num_parts * self.voxel_z_resolutions[1], is_bias=True),
            conv1_block(256, self.num_parts * self.voxel_z_resolutions[2], is_bias=True),
            conv1_block(512, self.num_parts * self.voxel_z_resolutions[3], is_bias=True)])

        self.after_intermediate_layers = nn.ModuleList([
            conv1_block(self.num_parts * self.voxel_z_resolutions[0], 256 + 128),
            conv1_block(self.num_parts * self.voxel_z_resolutions[1], 256 + 256),
            conv1_block(self.num_parts * self.voxel_z_resolutions[2], 256 + 256)])

        self.skip_intermediate_layers = nn.ModuleList([
            conv1_block(256 + 128, 256 + 128),
            conv1_block(256 + 256, 256 + 256),
            conv1_block(256 + 256, 256 + 256)])

    def forward(self, x):  # x dim: [BCHW]
        init_conv1 = self.init_conv1(x)
        init_conv2 = self.init_conv2(init_conv1)

        out_voxels = list()
        layers = zip(self.hourglasses,
                     self.prev_intermediate_layers,
                     self.voxel_intermediate_layers,
                     self.after_intermediate_layers,
                     self.skip_intermediate_layers)

        _input = init_conv2
        inter = init_conv1

        for hg, prev, voxel, after, skip in layers:  # loop over (num_stack - 1)
            # prev_inter = inter

            hg = hg(_input)
            prev = prev(hg)
            voxel = voxel(prev)  # BCHW
            after = after(voxel)

            concat = torch.cat((prev, inter), 1)
            skip = skip(concat)

            _input = skip + after
            inter = prev

            # stacking intermediate heatmaps for intermediate supervision
            out_voxels.append(voxel)

        hg = self.hourglasses[-1](_input)
        prev = self.prev_intermediate_layers[-1](hg)
        voxel = self.voxel_intermediate_layers[-1](prev)
        out_voxels.append(voxel)

        return out_voxels
