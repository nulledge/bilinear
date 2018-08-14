import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import MPII
import Model.hourglass
from util.visualize import draw_merged_image
from util import config
from tensorboardX import SummaryWriter

assert config.task == 'train'

writer = SummaryWriter()

data = DataLoader(
    MPII.Dataset(
        root=config.root['MPII'],
        task=config.task,
    ),
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.num_workers,
)

device = torch.device(config.device)
hourglass, optimizer, criterion, step, pretrained_epoch = Model.hourglass.load_model(device,
                                                                                     config.pretrained['hourglass'])

hourglass.train()

dummy = torch.rand(1, 3, 256, 256).to(config.device)
with SummaryWriter(comment='Hourglass') as w:
    w.add_graph(hourglass, input_to_model=dummy)

for epoch in range(pretrained_epoch + 1, pretrained_epoch + 100 + 1):
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

                nn.utils.clip_grad_norm_(hourglass.parameters(), max_norm=1)

                optimizer.step()

                writer.add_scalar('data/loss', loss.item(), step)
                if step % 300 == 0:
                    for name, param in hourglass.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

                    out = outputs[-1, :].squeeze().contiguous()
                    gt_images = images.cpu().numpy()
                    true_img = draw_merged_image(out, gt_images.copy())
                    infr_img = draw_merged_image(heatmaps, gt_images.copy())
                    writer.add_image('Ground truth image', true_img, step)
                    writer.add_image('Inference image', infr_img, step)

                progress.set_postfix(loss=float(loss.item()))
                progress.update(1)
                step = step + 1

    torch.save(
        {
            'epoch': epoch,
            'step': step,
            'state': hourglass.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        '{pretrained}/{epoch}.save'.format(pretrained=config.pretrained['hourglass'], epoch=epoch)
    )
