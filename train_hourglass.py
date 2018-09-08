import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

import MPII
import model.hourglass
from util import config
from util.visualize import colorize, overlap
from util.log import get_logger

logger, log_dir, comment = get_logger(comment=config.hourglass.comment)

if config.hourglass.comment is None or not os.path.exists('save/{comment}/parameter'.format(comment=comment)):
    logger.info('                                                           ')
    logger.info('                                                           ')
    logger.info('===========================================================')
    logger.info('Comment        : ' + comment + '                           ')
    logger.info('===========================================================')
    logger.info('Architecture   : ' + 'Stacked hourglass' + '               ')
    logger.info('   -task       : ' + MPII.Task.Train + '                   ')
    logger.info('   -device     : ' + str(config.hourglass.device) + '      ')
    logger.info('===========================================================')
    logger.info('Data           : ' + 'MPII' + '                            ')
    logger.info('   -directory  : ' + config.hourglass.data_dir + '         ')
    logger.info('   -mini batch : ' + str(config.hourglass.batch_size) + '  ')
    logger.info('   -shuffle    : ' + 'True' + '                            ')
    logger.info('   -worker     : ' + str(config.hourglass.num_workers) + ' ')
    logger.info('===========================================================')

data = DataLoader(
    MPII.Dataset(
        root=config.hourglass.data_dir,
        task=MPII.Task.Train,
    ),
    batch_size=config.hourglass.batch_size,
    num_workers=config.hourglass.num_workers,
    shuffle=True,
    pin_memory=True,
)

hourglass, optimizer, step, train_epoch = model.hourglass.load(
    device=config.hourglass.device,
    parameter_dir='{log_dir}/parameter'.format(log_dir=log_dir) if config.hourglass.comment is not None else None,
)
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir='{log_dir}/visualize'.format(
    log_dir=log_dir,
))

hourglass.train()

resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=[256, 256]),
    transforms.ToTensor(),
])
upscale = lambda heatmaps: torch.stack([resize(heatmap) for heatmap in heatmaps.cpu()]).to(config.hourglass.device)

for epoch in range(train_epoch + 1, train_epoch + 10 + 1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:

        with torch.set_grad_enabled(True):

            for images, heatmaps, _, _, _, _ in data:

                images = images.to(config.hourglass.device)
                heatmaps = heatmaps.to(config.hourglass.device)

                optimizer.zero_grad()
                outputs = hourglass(images)

                loss = sum([criterion(output, heatmaps) for output in outputs])
                loss.backward()

                nn.utils.clip_grad_norm_(hourglass.parameters(), max_norm=1)

                optimizer.step()

                writer.add_scalar('SH/loss', loss, step)
                if step % 100 == 0:
                    ground_truth = overlap(images=images, heatmaps=upscale(colorize(heatmaps)))
                    prediction = overlap(images=images, heatmaps=upscale(colorize(outputs[-1])))

                    writer.add_image('{comment}/ground-truth'.format(comment=config.hourglass.comment), ground_truth.data, step)
                    writer.add_image('{comment}/prediction'.format(comment=config.hourglass.comment), prediction.data, step)

                progress.set_postfix(loss=float(loss.item()))
                progress.update(1)
                step = step + 1

    save_dir = '{log_dir}/parameter'.format(log_dir=log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_to = '{save_dir}/{epoch}.save'.format(save_dir=save_dir, epoch=epoch, )
    torch.save(
        {
            'epoch': epoch,
            'step': step,
            'state': hourglass.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        save_to,
    )
    logger.info('Epoch {epoch} saved (loss: {loss})'.format(epoch=epoch, loss=float(loss.item())))

writer.close()
