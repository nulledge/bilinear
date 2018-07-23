from .hourglass import *

import numpy as np
import os


def load_model(device):
    hourglass = StackedHourglass(stacks=8, joints=16)
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

    hourglass = hourglass.to(device)
    optimizer = torch.optim.RMSprop(hourglass.parameters(), lr=2.5e-4)
    if pretrained_model is not None:
        optimizer.load_state_dict(pretrained_model['optimizer'])
    criterion = nn.MSELoss()

    return hourglass, optimizer, criterion, step, pretrained_epoch
