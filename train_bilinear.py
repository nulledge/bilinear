import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

import H36M
import model
from util import config
from util.log import get_logger

time_stamp_to_load = None

logger, log_dir, time_stamp = get_logger(time_stamp=time_stamp_to_load)

if time_stamp_to_load is None:
    logger.info('                                                           ')
    logger.info('                                                           ')
    logger.info('===========================================================')
    logger.info('Time stamp     : ' + time_stamp + '                        ')
    logger.info('===========================================================')
    logger.info('Architecture   : ' + 'Bilinear' + '                        ')
    logger.info('   -protocol   : ' + config.bilinear.protocol + '          ')
    logger.info('   -task       : ' + H36M.Task.Train + '                   ')
    logger.info('   -device     : ' + str(config.bilinear.device) + '       ')
    logger.info('===========================================================')
    logger.info('Data           : ' + 'Human3.6M' + '                       ')
    logger.info('   -directory  : ' + config.bilinear.data_dir + '          ')
    logger.info('   -mini batch : ' + str(config.bilinear.batch_size) + '   ')
    logger.info('   -shuffle    : ' + 'True' + '                            ')
    logger.info('   -worker     : ' + str(config.bilinear.num_workers) + '  ')
    logger.info('===========================================================')

data = DataLoader(
    H36M.Dataset(
        data_dir=config.bilinear.data_dir,
        task=H36M.Task.Train,
        protocol=config.bilinear.protocol,
    ),
    batch_size=config.bilinear.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.bilinear.num_workers,
)

bilinear, optimizer, step, train_epoch = model.bilinear.load(
    device=config.bilinear.device,
    parameter_dir='{log_dir}/parameter'.format(log_dir=log_dir) if time_stamp_to_load is not None else None,
)
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir='{log_dir}/visualize'.format(
    log_dir=log_dir,
))

bilinear.train()

for epoch in range(train_epoch + 1, 200 + 1):
    with tqdm(total=len(data), desc='%d epoch' % epoch) as progress:
        with torch.set_grad_enabled(True):

            for subset, _, _, _ in data:

                in_image_space = subset[H36M.Annotation.Part]
                in_camera_space = subset[H36M.Annotation.S]

                # Learning rate decay
                if config.bilinear.lr_decay.activate and config.bilinear.lr_decay.condition(step):
                    lr = config.bilinear.lr_decay.function(step)
                    logger.info('Learning rate decay to {lr} (step: {step})'.format(lr=lr, step=step))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                in_image_space = in_image_space.to(config.bilinear.device)
                in_camera_space = in_camera_space.to(config.bilinear.device)

                optimizer.zero_grad()
                prediction = bilinear(in_image_space)

                loss = criterion(prediction, in_camera_space)
                loss.backward()

                nn.utils.clip_grad_norm_(bilinear.parameters(), max_norm=1)

                optimizer.step()

                # Too frequent update reduces the performance.
                writer.add_scalar('BI/loss', loss, step)

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
                'state': bilinear.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            save_to,
        )
        logger.info('Epoch {epoch} saved (loss: {loss})'.format(epoch=epoch, loss=float(loss.item())))

logger.info('===========================================================')
