
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch
import copy
from dotmap import DotMap
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from torchvision import transforms

import model
import data.H36M


# In[2]:


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


selected = 'Newell'
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
    task=data.Task.Train,
    protocol='SH_{protocol}'.format(protocol=selected),
    position_only=False,
)


# In[6]:


# Load input RGB image
rgb = imageio.imread('sit.jpg')
tmp = copy.deepcopy(rgb)
rgb = torch.Tensor(rgb).float().to(device)
rgb = rgb / 255
rgb = rgb.permute(2, 0, 1)
rgb = rgb.unsqueeze(0)
print('rgb', rgb.shape)


# In[7]:


# Calculate 2D pose detections
htmps = SH(rgb)
htmps = htmps[-1]
print('htmps', htmps.shape)


# In[8]:
# MSDN 'Heat Map Color Gradients'
COLOR_SPECTRUM = torch.Tensor([
    [0.0, 0.0, 0.5],  # Navy
    [0.0, 0.0, 1.0],  # Blue
    [0.0, 1.0, 0.0],  # Green
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 0.0, 0.0],  # Red
])
INCANDESCENT = torch.Tensor([
    [0.0, 0.0, 0.0],  # Black
    [0.5, 0.0, 0.0],  # Dark red
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 1.0, 1.0],  # White
])


def colorize(heatmaps, color_gradient=COLOR_SPECTRUM):
    color_gradient = color_gradient.to(heatmaps.device)

    batch, _, height, width = heatmaps.shape
    heatmaps, _ = heatmaps.max(dim=1)
    heatmaps = heatmaps.view(-1)

    index = heatmaps.mul(len(color_gradient) - 1).clamp(0, len(color_gradient) - 1)
    lower_bound, upper_bound = (index.floor(), index.ceil())
    rate = (index - lower_bound).view(-1, 1)
    heatmaps = color_gradient.index_select(0, lower_bound.long()) * (1 - rate) \
               + color_gradient.index_select(0, upper_bound.long()) * rate

    return heatmaps.view(batch, height, width, 3).permute(0, 3, 1, 2)  # 3 for RGB channel


def overlap(heatmaps, images, ratio=0.5):
    assert 0.0 <= ratio <= 1.0
    return heatmaps * ratio + images * (1 - ratio)

resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=[256, 256]),
    transforms.ToTensor(),
])
upscale = lambda heatmaps: torch.stack([resize(heatmap) for heatmap in heatmaps.cpu()]).to(device)

ground_truth = overlap(images=rgb, heatmaps=upscale(colorize(htmps)))
imageio.imwrite('overlay.jpg', np.asarray(ground_truth.data)[0].transpose(1, 2, 0))


# Convert 2D pose detections from heatmaps to matrix
scale = torch.Tensor([256/200]).float().to(device)
center = torch.Tensor([256/2, 256/2]).float().to(device)

pose = torch.argmax(htmps.view(1, 16, -1), dim=-1)
pose = torch.stack([
    pose % 64,
    pose // 64,
], dim=-1).float()
pose = pose - 32
pose = center.view(1, 1, 2) + pose / 64 * scale.view(1, 1, 1) * 200
print(pose.shape)


# In[10]:


# Convert 2D pose detections from MPII format to Human3.6M format
from_MPII_to_H36M = [6, 3, 4, 5, 2, 1, 0, 7, 8, 9, 13, 14, 15, 12, 11, 10]
from_MPII_to_H36M = torch.Tensor(from_MPII_to_H36M).long().to(device)
pose = torch.index_select(pose, dim=1, index=from_MPII_to_H36M)
print(pose.shape)


# In[11]:


pose_2d = np.asarray(pose).reshape(-1)
mean = H36M.data[data.Task.Train][data.Annotation.Mean_Of + data.Annotation.Part]
stddev = H36M.data[data.Task.Train][data.Annotation.Stddev_Of + data.Annotation.Part]
pose_2d = (pose_2d - mean) / stddev
pose_2d = np.expand_dims(pose_2d, 0)
pose_2d = torch.Tensor(pose_2d).float().to(device)
print(pose_2d.shape, pose_2d.device)


# In[41]:


pose_3d = BI(pose_2d)
pose_3d = np.asarray(pose_3d.data).reshape(-1)
mean = H36M.data[data.Task.Train][data.Annotation.Mean_Of + data.Annotation.S]
stddev = H36M.data[data.Task.Train][data.Annotation.Stddev_Of + data.Annotation.S]
pose_3d = pose_3d * stddev + mean
pose_3d = pose_3d.reshape(-1, 3)
print(pose_3d.shape)


# In[42]:


fig = plt.figure()
# ax = fig.add_subplot(111)
ax = fig.gca(projection='3d')
x = 0
y = 2
z = 1
pose_3d[:, z] = -pose_3d[:, z]
pelvis = np.asarray([0, ], dtype=pose_3d.dtype)
neck = np.expand_dims(pose_3d[7], axis=-1)
ax.set_xlim([-1000, 1000])
ax.set_ylim([-1000, 1000])
ax.set_zlim([-1000, 1000])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(-30, -90)
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

plt.show()
