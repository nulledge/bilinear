import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

import MPII
import Model.hourglass
from util.visualize import colorize, overlap
from util import config

assert config.task == 'train'

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

writer = SummaryWriter(log_dir='/media/nulledge/2nd/ubuntu/bilinear/log')

for epoch in range(pretrained_epoch + 1, pretrained_epoch + 100 + 1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:

        with torch.set_grad_enabled(True):

            for images, heatmaps, keypoints in data:
                images = images.to(device)
                heatmaps = heatmaps.to(device)

                optimizer.zero_grad()
                outputs = hourglass(images)

                loss = sum([criterion(output, heatmaps) for output in outputs])
                loss.backward()

                nn.utils.clip_grad_norm_(hourglass.parameters(), max_norm=1)

                optimizer.step()

                writer.add_scalar('data/loss', loss, step)
                if step % 10 == 0:
                    resize = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(size=[64, 64]),
                        transforms.ToTensor(),
                    ])
                    images = torch.stack([resize(image) for image in images.cpu()]).to(device)

                    ground_truth = overlap(images=images, heatmaps=colorize(heatmaps))
                    prediction = overlap(images=images, heatmaps=colorize(outputs[-1]))

                    writer.add_image('ground truth', ground_truth.data, step)
                    writer.add_image('prediction', prediction.data, step)

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

writer.close()