# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch
import copy
import cv2
from dotmap import DotMap
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import model
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


# In[2]:
def draw_fig(pose_3d, angle):
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax = fig.gca(projection='3d')
    x = 0
    y = 2
    z = 1
    pelvis = np.asarray([0, ], dtype=pose_3d.dtype)
    neck = np.expand_dims(pose_3d[7], axis=-1)
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-1000, 1000])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(**angle)
    ax.plot(
        np.concatenate([pelvis, pose_3d[0:3][:, x]]),
        np.concatenate([pelvis, pose_3d[0:3][:, y]]),
        np.concatenate([pelvis, pose_3d[0:3][:, z]]),
        label='Left leg')
    ax.plot(
        np.concatenate([pelvis, pose_3d[3:6][:, x]]),
        np.concatenate([pelvis, pose_3d[3:6][:, y]]),
        np.concatenate([pelvis, pose_3d[3:6][:, z]]),
        label='Right leg')
    ax.plot(
        np.concatenate([pelvis, pose_3d[6:10][:, x]]),
        np.concatenate([pelvis, pose_3d[6:10][:, y]]),
        np.concatenate([pelvis, pose_3d[6:10][:, z]]),
        label='Spine')
    ax.plot(
        np.concatenate([neck[x], pose_3d[13:16][:, x]]),
        np.concatenate([neck[y], pose_3d[13:16][:, y]]),
        np.concatenate([neck[z], pose_3d[13:16][:, z]]),
        label='Right arm')
    ax.plot(
        np.concatenate([neck[x], pose_3d[10:13][:, x]]),
        np.concatenate([neck[y], pose_3d[10:13][:, y]]),
        np.concatenate([neck[z], pose_3d[10:13][:, z]]),
        label='Left arm')
    ax.legend()

    return fig


config = DotMap()

# config.GT.SH.model = None
# config.GT.SH.param = None
# config.GT.BI.model = 'bilinear'
# config.GT.BI.param = '/media/nulledge/2nd/ubuntu/bilinear/save/Bilineare GT c2f-converted/parameter/160.save'

config.IB.SH.model = 'hourglass'
config.IB.SH.param = '/media/nulledge/2nd/ubuntu/bilinear/save/SH/parameter/181.save'
config.IB.BI.model = 'bilinear'
config.IB.BI.param = '/media/nulledge/2nd/ubuntu/bilinear/save/Bilinear SH/parameter/171.save'

config.Newell.SH.model = 'hourglass_torch7'
config.Newell.SH.param = '/media/nulledge/2nd/ubuntu/sh/torch7_SH.save'
config.Newell.BI.model = 'bilinear'
config.Newell.BI.param = '/media/nulledge/2nd/ubuntu/bilinear/save/Bilinear SH Newell/parameter/280.save'

# In[3]:


selected = 'IB'
protocol = config[selected]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

# In[4]:


# Create SH and BI
SH, _, _, _ = getattr(model, protocol.SH.model).load(device=device) if protocol.SH.model is not None else None
BI, _, _, _ = getattr(model, protocol.BI.model).load(device=device)

# Load parameters of SH and BI
SH.load_state_dict(torch.load(protocol.SH.param)['state'])
BI.load_state_dict(torch.load(protocol.BI.param)['state'])

SH = SH.train()
BI = BI.eval()

# In[5]:


# Load Human3.6M for normalization and un-normalization
H36M = data.H36M.Parser(
    data_dir='/media/nulledge/3rd/data/Human3.6M',
    task=data.Task.Valid,
    protocol='SH_{protocol}'.format(protocol=selected),
    position_only=False,
)
in_stream = DataLoader(H36M, num_workers=8, batch_size=1, pin_memory=True, shuffle=False)

prev = None
out_stream = None

delta = 472
with tqdm(total=delta) as progress:
    for idx in range(delta, delta*2):
        images, _, info = H36M[idx]

        images = torch.Tensor(np.expand_dims(images, 0))

        now = info[data.Annotation.Action]

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

        batch_size, _, _, _ = images.shape
        rgb = images.to(device)

        htmps = SH(rgb)
        htmps = htmps[-1]

        scale = torch.Tensor([256 / 200]).float().to(device)
        center = torch.Tensor([256 / 2, 256 / 2]).float().to(device)

        pose = torch.argmax(htmps.view(1, 16, -1), dim=-1)
        pose = torch.stack([
            pose % 64,
            pose // 64,
        ], dim=-1).float()
        pose = pose - 32
        pose = center.view(1, 1, 2) + pose / 64 * scale.view(1, 1, 1) * 200

        from_MPII_to_H36M = [6, 3, 4, 5, 2, 1, 0, 7, 8, 9, 13, 14, 15, 12, 11, 10]
        from_MPII_to_H36M = torch.Tensor(from_MPII_to_H36M).long().to(device)
        pose = torch.index_select(pose, dim=1, index=from_MPII_to_H36M)

        pose_2d = np.asarray(pose).reshape(-1)
        mean = H36M.data[data.Task.Train][data.Annotation.Mean_Of + data.Annotation.Part]
        stddev = H36M.data[data.Task.Train][data.Annotation.Stddev_Of + data.Annotation.Part]
        pose_2d = (pose_2d - mean) / stddev
        pose_2d = np.expand_dims(pose_2d, 0)
        pose_2d = torch.Tensor(pose_2d).float().to(device)

        pose_3d = BI(pose_2d)
        pose_3d = np.asarray(pose_3d.data).reshape(-1)
        mean = H36M.data[data.Task.Train][data.Annotation.Mean_Of + data.Annotation.S]
        stddev = H36M.data[data.Task.Train][data.Annotation.Stddev_Of + data.Annotation.S]
        pose_3d = pose_3d * stddev + mean
        pose_3d = pose_3d.reshape(-1, 3)

        x = 0
        y = 2
        z = 1
        pose_3d[:, z] = -pose_3d[:, z]

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

        image = np.asarray(rgb[0].data * 255, dtype=np.uint8).reshape((3, 256, 256)).transpose(1, 2, 0)
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
