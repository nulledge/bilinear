import torch
import torch.nn as nn
import numpy as np
import os


def heavy_linear(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )


class BilinearUnit(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = heavy_linear(in_features=2 * 17, out_features=1024)
        self.linear = nn.ModuleList([
            heavy_linear(in_features=1024, out_features=1024) for _ in range(2)
        ])
        self.decode = nn.Linear(in_features=1024, out_features=3 * 17, bias=True)

    def forward(self, in_tensor):
        out_tensor = in_tensor

        out_tensor = self.encode(out_tensor)
        for linear in self.linear:
            skip_tensor = out_tensor
            out_tensor = linear(out_tensor)
            out_tensor = out_tensor + skip_tensor
        out_tensor = self.decode(out_tensor)

        return out_tensor


def load_model(device, pretrained):
    lr = 1.0e-3
    bilinear = BilinearUnit().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(bilinear.parameters(), lr=lr)
    step = 1

    pretrained_epoch = 0
    for _, _, files in os.walk(pretrained):
        for file in files:
            name, extension = file.split('.')
            epoch = int(name)
            if epoch > pretrained_epoch:
                pretrained_epoch = epoch

    if pretrained_epoch > 0:
        pretrained_model = os.path.join(
            '{pretrained}/{epoch}.save'.format(pretrained=pretrained, epoch=pretrained_epoch))
        pretrained_model = torch.load(pretrained_model)

        bilinear.load_state_dict(pretrained_model['state'])
        step = pretrained_model['step']
        lr = pretrained_model['lr']
        optimizer.load_state_dict(pretrained_model['optimizer'])

    else:

        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)

        bilinear.apply(weight_init)  # (lambda x: nn.init.kaiming_normal(x.weight)))

    return bilinear, optimizer, criterion, step, pretrained_epoch, lr
