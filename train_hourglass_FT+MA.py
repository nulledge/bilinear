import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

import H36M
import model.hourglass
from util import config
from util.visualize import colorize, overlap

writer = SummaryWriter()

data = DataLoader(
    H36M.Dataset(
        data_dir=config.bilinear.data_dir,
        task=H36M.Task.Train,
        protocol=H36M.Protocol.GT,
    ),
    batch_size=config.hourglass.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.bilinear.num_workers,
)

hourglass, optimizer, step, train_epoch = model.hourglass.load(config.hourglass.parameter_dir, config.hourglass.device)
criterion = nn.MSELoss()

# train_epoch equals -1 means that training is over
assert train_epoch is not -1

hourglass.train()

writer = SummaryWriter(log_dir=config.hourglass.log_dir + '/FT+center_moved')
resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=[64, 64]),
    transforms.ToTensor(),
])

from_H36M_to_MPII = torch.Tensor([6, 5, 4, 1, 2, 3, 0, 7, 8, 9, 15, 14, 13, 10, 11, 12]).long()

for epoch in range(train_epoch + 1, train_epoch + 100 + 1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:

        with torch.set_grad_enabled(True):

            for subset, images, heatmaps, action in data:

                images = images.to(config.hourglass.device)
                heatmaps = heatmaps.to(config.hourglass.device)

                heatmaps = torch.index_select(heatmaps, index=from_H36M_to_MPII, dim=1)

                optimizer.zero_grad()
                outputs = hourglass(images)

                loss = sum([criterion(output, heatmaps) for output in outputs])
                loss.backward()

                nn.utils.clip_grad_norm_(hourglass.parameters(), max_norm=1)

                optimizer.step()

                if config.hourglass.visualize:
                    writer.add_scalar('loss', loss, step)
                    if step % 10 == 0:
                        images = torch.stack([resize(image) for image in images.cpu()]).to(config.hourglass.device)

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
        '{parameter}/{epoch}.save'.format(parameter=config.hourglass.parameter_dir, epoch=epoch)
    )

writer.close()
