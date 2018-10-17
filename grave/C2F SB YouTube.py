import numpy as np
import imageio
import scipy.io
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D

import data.H36M
from hourglass import CoarseToFine

from util.euro_filter import OneEuroFilter

euro_cfg = {
    'freq': 10,  # Hz
    'mincutoff': 1.0,  # FIXME
    'beta': 0.0,  # FIXME
    'dcutoff': 1.0  # this one should be ok
}

euro_filter = OneEuroFilter(**euro_cfg)
timestamp = 0


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


param = torch.load('batch_norm_026.ckpt')

cfg = param['config']
device = cfg.device

c2f = CoarseToFine(voxel_z_resolutions=[1, 2, 4, 64], num_parts=17).to(device)
c2f.load_state_dict(param['state'])

valid_batch_size = 1

# Reconstructed voxel z-value.
z_boundary = np.squeeze(
    scipy.io.loadmat('/media/nulledge/2nd/data/Human3.6M/converted/annot/data/voxel_limits.mat')['limits'])
z_reconstructed = (z_boundary[1:65] + z_boundary[0:64]) / 2
z_delta = z_boundary[32]
z_reconstructed = torch.tensor(z_reconstructed).to(cfg.device).float()
z_delta = torch.tensor(z_delta).to(cfg.device)

# Load Human3.6M for normalization and un-normalization
H36M = data.H36M.Parser(
    data_dir='/media/nulledge/3rd/data/Human3.6M',
    task=data.Task.Valid,
    protocol='GT',
    position_only=False,
)
in_stream = DataLoader(H36M, num_workers=8, batch_size=1, pin_memory=True, shuffle=False)

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

        now = 'C2F SB YouTube'

        if prev != now:
            if out_stream is not None:
                out_stream.release()
                break

            out_stream = cv2.VideoWriter(
                '{action}.avi'.format(action=now),
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                10.0, (480 + 640 + 640, 480*2), True)
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

        canvas = np.zeros(shape=(480*2, 480 + 640 + 640, 3), dtype=np.uint8)

        image = np.asarray(imgs[0].data * 255, dtype=np.uint8).reshape((3, 256, 256)).transpose(1, 2, 0)
        imgs = cv2.resize(image, (480, 480))

        canvas[240:240+480, 0:480, :] = imgs
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

        canvas[480:480*2, 480:480 + 640, :] = fig1
        canvas[480:480*2, 480 + 640:480 + 640 + 640, :] = fig2

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        out_stream.write(canvas)
        progress.update(1)
        timestamp += 1.0 / euro_cfg['freq']

out_stream.release()