import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import MPII
import Model
from MPII.util import draw_line, draw_merged_image
from util.path import safe_path

data = DataLoader(
    MPII.Dataset(
        root='/media/nulledge/2nd/data/MPII/',
        task='train',
    ),
    batch_size=8,
    shuffle=True,
    pin_memory=True,
    num_workers=8
)

hourglass = Model.StackedHourglass(stacks=8, joints=16)
step = np.zeros([1], dtype=np.uint32)

pretrained_epoch = 0
for _, _, files in os.walk('./pretrained'):
    for file in files:
        name, extension = file.split('.')
        epoch = int(name)
        if epoch > pretrained_epoch:
            pretrained_epoch = epoch

if pretrained_epoch > 0:
    pretrained_model = os.path.join('./pretrained/{epoch}.save'.format(epoch=pretrained_epoch))
    pretrained_model = torch.load(pretrained_model)

    hourglass.load_state_dict(pretrained_model['state'])
    step[0] = pretrained_model['step']

else:
    pretrained_model = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hourglass = hourglass.to(device)
optimizer = torch.optim.RMSprop(hourglass.parameters(), lr=2.5e-4)
if pretrained_model is not None:
    optimizer.load_state_dict(pretrained_model['optimizer'])
criterion = nn.MSELoss()

loss_window, gt_image_window, out_image_window = None, None, None

for epoch in range(pretrained_epoch + 1, pretrained_epoch + 100 +1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:

        with torch.set_grad_enabled(True):

            for images, heatmaps, keypoints in data:
                images_cpu = images
                images = images.to(device)
                heatmaps = heatmaps.to(device)

                optimizer.zero_grad()
                outputs = hourglass(images)

                loss = sum([criterion(output, heatmaps) for output in outputs])
                loss.backward()

                optimizer.step()

                progress.set_postfix(loss=float(loss.data[0]))
                progress.update(1)

                loss_window = draw_line(x=step,
                                        y=np.array([float(loss.data)]),
                                        window=loss_window)
                if step % 10 == 0:
                    out = outputs[-1, :].squeeze().contiguous()
                    gt_images = images_cpu.cpu().numpy()
                    gt_image_window = draw_merged_image(out, gt_images.copy(), gt_image_window)
                    out_image_window = draw_merged_image(heatmaps, gt_images.copy(), out_image_window)
                step = step + 1

    torch.save(
        {
            'epoch': epoch,
            'step': step,
            'state': hourglass.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        safe_path('./pretrained/{epoch}.save'.format(epoch=epoch))
    )
