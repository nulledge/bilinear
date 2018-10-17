import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D

import data.H36M

from util.euro_filter import OneEuroFilter

euro_cfg = {
    'freq': 10,  # Hz
    'mincutoff': 1.0,  # FIXME
    'beta': 0.0,  # FIXME
    'dcutoff': 1.0  # this one should be ok
}

euro_filter = OneEuroFilter(**euro_cfg)
timestamp = 0


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
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
    def __init__(self, n, numIn, numOut):
        super(Hourglass, self).__init__()

        self.up1 = Residual(numIn, 256)
        self.up2 = Residual(256, 256)
        self.up4 = Residual(256, numOut)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.low1 = Residual(numIn, 256)
        self.low2 = Residual(256, 256)
        self.low5 = Residual(256, 256)

        if n > 1:
            self.low6 = Hourglass(

                n - 1, 256, numOut)
        else:
            self.low6 = Residual(256, numOut)

        self.low7 = Residual(numOut, numOut)
        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, inp):
        up1 = self.up1(inp)
        up2 = self.up2(up1)
        up4 = self.up4(up2)

        pool = self.max_pool(inp)
        low1 = self.low1(pool)
        low2 = self.low2(low1)
        low5 = self.low5(low2)

        low6 = self.low6(low5)
        low7 = self.low7(low6)
        up5 = self.up5(low7)

        return up4 + up5


class MainModel(nn.Module):
    def __init__(self, in_channels=3):
        super(MainModel, self).__init__()

        self.cnv1_ = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.cnv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.r1 = Residual(in_channels=64, out_channels=128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, 128)
        self.r6 = Residual(128, 256)

        self.outputDim = [1 * 17, 2 * 17, 4 * 17, 64 * 17]

        self.hg1 = Hourglass(4, 256, 512)
        self.l1 = self.lin(512, 512)
        self.l2 = self.lin(512, 256)
        self.out1 = nn.Conv2d(256, self.outputDim[0], kernel_size=1, stride=1, padding=0)
        self.out1_ = nn.Conv2d(self.outputDim[0], 256 + 128, kernel_size=1, stride=1, padding=0)
        self.cat1_ = nn.Conv2d(256 + 128, 256 + 128, kernel_size=1, stride=1, padding=0)

        self.hg2 = Hourglass(4, 256 + 128, 512)
        self.l3 = self.lin(512, 512)
        self.l4 = self.lin(512, 256)
        self.out2 = nn.Conv2d(256, self.outputDim[1], kernel_size=1, stride=1, padding=0)
        self.out2_ = nn.Conv2d(self.outputDim[1], 256 + 256, kernel_size=1, stride=1, padding=0)
        self.cat2_ = nn.Conv2d(256 + 256, 256 + 256, kernel_size=1, stride=1, padding=0)

        self.hg3 = Hourglass(4, 256 + 256, 512)
        self.l5 = self.lin(512, 512)
        self.l6 = self.lin(512, 256)
        self.out3 = nn.Conv2d(256, self.outputDim[2], kernel_size=1, stride=1, padding=0)
        self.out3_ = nn.Conv2d(self.outputDim[2], 256 + 256, kernel_size=1, stride=1, padding=0)
        self.cat3_ = nn.Conv2d(256 + 256, 256 + 256, kernel_size=1, stride=1, padding=0)

        self.hg4 = Hourglass(4, 256 + 256, 512)
        self.l7 = self.lin(512, 512)
        self.l8 = self.lin(512, 512)
        self.out4 = nn.Conv2d(512, self.outputDim[3], kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        cnv1_ = self.cnv1_(inp)
        cnv1 = self.cnv1(cnv1_)
        r1 = self.r1(cnv1)
        pool = self.pool(r1)
        r4 = self.r4(pool)
        r5 = self.r5(r4)
        r6 = self.r6(r5)

        hg1 = self.hg1(r6)
        l1 = self.l1(hg1)
        l2 = self.l2(l1)
        out1 = self.out1(l2)
        out1_ = self.out1_(out1)
        cat1 = torch.cat([l2, pool], 1)
        cat1_ = self.cat1_(cat1)
        int1 = cat1_ + out1_

        hg2 = self.hg2(int1)
        l3 = self.l3(hg2)
        l4 = self.l4(l3)
        out2 = self.out2(l4)
        out2_ = self.out2_(out2)
        cat2 = torch.cat([l4, l2], 1)
        cat2_ = self.cat2_(cat2)
        int2 = cat2_ + out2_

        hg3 = self.hg3(int2)
        l5 = self.l5(hg3)
        l6 = self.l6(l5)
        out3 = self.out3(l6)
        out3_ = self.out3_(out3)
        cat3 = torch.cat([l6, l4], 1)
        cat3_ = self.cat3_(cat3)
        int3 = cat3_ + out3_

        hg4 = self.hg4(int3)
        l7 = self.l7(hg4)
        l8 = self.l8(l7)
        out4 = self.out4(l8)

        return out4

    def lin(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )


def get_coord_3d(heatmap, num_parts):
    batch, _, _, height, width = heatmap.shape  # BJDHW
    heatmap = heatmap.view(batch, num_parts, -1)
    _, idx = heatmap.max(2)

    coords = idx.to(torch.float).unsqueeze(-1).repeat(1, 1, 3)
    coords[:, :, 0] = (coords[:, :, 0] % (height * width)) % height  # x
    coords[:, :, 1] = (coords[:, :, 1] % (height * width)) / height  # y
    coords[:, :, 2] = (coords[:, :, 2] / (height * width))  # z
    # coords = coords.floor()

    return coords.floor_()


def draw_fig(pose_3d, angle):
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax = fig.gca(projection='3d')
    pelvis = pose_3d[0]
    neck = pose_3d[8]
    lim = [-1, 64]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(**angle)
    ax.plot(
        np.concatenate([[pelvis[x]], pose_3d[1:4][:, x]]),
        np.concatenate([[pelvis[y]], pose_3d[1:4][:, y]]),
        np.concatenate([[pelvis[z]], pose_3d[1:4][:, z]]),
        label='Left leg')
    ax.plot(
        np.concatenate([[pelvis[x]], pose_3d[4:7][:, x]]),
        np.concatenate([[pelvis[y]], pose_3d[4:7][:, y]]),
        np.concatenate([[pelvis[z]], pose_3d[4:7][:, z]]),
        label='Right leg')
    ax.plot(
        np.concatenate([[pelvis[x]], pose_3d[7:11][:, x]]),
        np.concatenate([[pelvis[y]], pose_3d[7:11][:, y]]),
        np.concatenate([[pelvis[z]], pose_3d[7:11][:, z]]),
        label='Spine')
    ax.plot(
        np.concatenate([[neck[x]], pose_3d[11:14][:, x]]),
        np.concatenate([[neck[y]], pose_3d[11:14][:, y]]),
        np.concatenate([[neck[z]], pose_3d[11:14][:, z]]),
        label='Left arm')
    ax.plot(
        np.concatenate([[neck[x]], pose_3d[14:17][:, x]]),
        np.concatenate([[neck[y]], pose_3d[14:17][:, y]]),
        np.concatenate([[neck[z]], pose_3d[14:17][:, z]]),
        label='Right arm')
    ax.legend()

    return fig


cfg = torch.load('batch_norm_026.ckpt')['config']
device = cfg.device

param = torch.load('torch7_c2f.save')

c2f = MainModel().to(device)
c2f.load_state_dict(param)

valid_batch_size = 1

# Reconstructed voxel z-value.
z_boundary = np.squeeze(
    scipy.io.loadmat('/media/nulledge/2nd/data/Human3.6M/converted/annot/data/voxel_limits.mat')['limits'])
z_reconstructed = (z_boundary[1:65] + z_boundary[0:64]) / 2
z_delta = z_boundary[32]
z_reconstructed = torch.tensor(z_reconstructed).to(cfg.device).float()
z_delta = torch.tensor(z_delta).to(cfg.device)

prev = None
out_stream = None

file_name = 'hummasong.mp4'
cap = cv2.VideoCapture(file_name)
euro_cfg['freq'] = cap.get(cv2.CAP_PROP_FPS)

currentTime = 18000
cap.set(cv2.CAP_PROP_POS_MSEC, currentTime)

from preprocess_img import centerCrop

with tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT), desc='valid epoch') as progress:
    while True:
        isCaptured, frame_org = cap.read()
        if not isCaptured:
            break
        frame_org = centerCrop(frame_org)
        frame_org = cv2.resize(frame_org, dsize=(256, 256))
        frame_org = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
        frame = np.swapaxes(np.swapaxes(frame_org, 1, 2), 0, 1)
        frame = np.divide(frame, 255)
        frameTensor = torch.Tensor([frame])

        images = frameTensor

        now = 'C2F Origin YouTube'

        if prev != now:
            if out_stream is not None:
                out_stream.release()
                break

            out_stream = cv2.VideoWriter(
                '{action}.avi'.format(action=now),
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                10.0, (480 + 640 + 640, 480 * 2), True)
            print('Action:', now)

        prev = now

        # rgb = imageio.imread('rgb.jpg')
        # rgb = torch.Tensor(rgb).float().to(device)
        # rgb = rgb / 255
        # rgb = rgb.permute(2, 0, 1)
        # rgb = rgb.unsqueeze(0)
        # images = rgb
        batch_size, _, _, _ = images.shape

        with torch.no_grad():
            imgs = images.to(cfg.device)
            flip_imgs = images.flip(3).to(cfg.device)

            outputs = c2f(imgs)
            flip_outputs = c2f(flip_imgs)

        out = outputs[-1].view(-1, cfg.num_parts, cfg.voxel_z_res[-1],
                               cfg.voxel_xy_res, cfg.voxel_xy_res)
        flip_out = flip_outputs[-1].view(-1, cfg.num_parts, cfg.voxel_z_res[-1],
                                         cfg.voxel_xy_res, cfg.voxel_xy_res)

        # swap left right joints
        swap_indices = torch.LongTensor([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]).to(cfg.device)
        flip_out = torch.index_select(flip_out, 1, swap_indices)
        flip_out = flip_out.flip(4)

        out = (out + flip_out) / 2.0  # normal and flip sum and div 2
        pose = get_coord_3d(out, num_parts=cfg.num_parts)

        pose = np.asarray(pose.data).reshape(-1, 3)
        x = 0
        y = 2
        z = 1
        pose[:, z] = -pose[:, z] + 64

        pose_3d = pose

        fig = draw_fig(pose_3d, {'elev': 0, 'azim': -90})
        fig.canvas.draw()
        fig1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig1 = fig1.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        fig = draw_fig(pose_3d, {'elev': 0, 'azim': -60})
        fig.canvas.draw()
        fig2 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig2 = fig2.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        canvas = np.zeros(shape=(480 * 2, 480 + 640 + 640, 3), dtype=np.uint8)

        image = np.asarray(imgs[0].data * 255, dtype=np.uint8).reshape((3, 256, 256)).transpose(1, 2, 0)
        imgs = cv2.resize(image, (480, 480))

        canvas[240:240 + 480, 0:480, :] = imgs
        canvas[:480, 480:480 + 640, :] = fig1
        canvas[:480, 480 + 640:480 + 640 + 640, :] = fig2

        pose_3d = euro_filter(pose_3d, timestamp)

        fig = draw_fig(pose_3d, {'elev': 0, 'azim': -90})
        fig.canvas.draw()
        fig1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig1 = fig1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        fig = draw_fig(pose_3d, {'elev': 0, 'azim': -60})
        fig.canvas.draw()
        fig2 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig2 = fig2.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        canvas[480:480 * 2, 480:480 + 640, :] = fig1
        canvas[480:480 * 2, 480 + 640:480 + 640 + 640, :] = fig2

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        out_stream.write(canvas)
        progress.update(1)
        timestamp += 1.0 / euro_cfg['freq']

out_stream.release()