import numpy as np
import imageio
import torch
import copy
from dotmap import DotMap
from itertools import product

import model
import data.H36M

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

selected = 'Newell'
protocol = config[selected]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

# Create SH and BI
SH, _, _, _ = getattr(model, protocol.SH.model).load(device=device) if protocol.SH.model is not None else None
BI, _, _, _ = getattr(model, protocol.BI.model).load(device=device)

# Load parameters of SH and BI
SH.load_state_dict(torch.load(protocol.SH.param)['state'])
BI.load_state_dict(torch.load(protocol.BI.param)['state'])

# Load input RGB image
rgb = imageio.imread('rgb.jpg')
tmp = copy.deepcopy(rgb)
rgb = torch.Tensor(rgb).float().to(device)
rgb = rgb / 255
rgb = rgb.permute(2, 0, 1)
rgb = rgb.unsqueeze(0)
print('rgb', rgb.shape)

# Calculate 2D pose detections
htmps = SH(rgb)
htmps = htmps[-1]
print('htmps', htmps.shape)

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

# Convert 2D pose detections from MPII format to Human3.6M format
# One of duplicated values, 9, will be removed at H36M/data.py
from_MPII_to_H36M = [6, 3, 4, 5, 2, 1, 0, 7, 8, 9, 9, 13, 14, 15, 12, 11, 10]
from_MPII_to_H36M = torch.Tensor(from_MPII_to_H36M).long().to(device)
pose = torch.index_select(pose, dim=1, index=from_MPII_to_H36M)

# Load Human3.6M for normalization and un-normalization
data.H36M.Parser(
    data_dir='/media/nulledge/3rd/data/Human3.6M',
    task=data.Task.Train,
    protocol='SH_{protocol}'.format(protocol=selected),
)

# in_image_space = np.squeeze(np.asarray(pose.cpu()))
# print('in_image_space', in_image_space.shape)
# for x, y in in_image_space:
#     for tx, ty in product(range(-5, 5), range(-5, 5)):
#         xx, yy = int(x + tx), int(y + ty)
#         if xx < 0 or xx >= 256 or yy < 0 or yy >= 256:
#             continue
#         tmp[yy, xx, :] = [255, 0, 0]
# imageio.imwrite('pred.jpg', tmp)
#
# H36M = data.H36M.Parser(data_dir='/media/nulledge/3nd/data/Human3.6M/train_SH.bin')
